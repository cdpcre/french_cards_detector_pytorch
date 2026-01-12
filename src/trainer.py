import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from .dataset import YOLODataset, create_weighted_sampler
from .layer_analyzer import analyze_yolo11_layers, get_two_phase_schedule
from .freezing_engine import apply_phase_unfreezing
from .optimizer_factory import create_conservative_optimizer

def setup_device_with_mixed_precision(args):
    """Setup device with proper mixed precision support"""
    if hasattr(args, 'no_amp') and args.no_amp:
        use_amp = False
    else:
        use_amp = getattr(args, 'amp', True)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        autocast_device = 'cuda' if use_amp else None
        print("Using CUDA with full AMP support" if use_amp else "Using CUDA without AMP")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        # MPS has limited AMP support in PyTorch 2.5+
        use_amp = use_amp and torch.__version__ >= '2.5.0'
        scaler = None  # MPS doesn't support GradScaler yet
        autocast_device = 'mps' if use_amp else None
        if use_amp:
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print(f"Using MPS with AMP: {use_amp}")
    else:
        device = torch.device('cpu')
        scaler = None
        autocast_device = 'cpu' if use_amp else None
        print("Using CPU with bfloat16 AMP" if use_amp else "Using CPU without AMP")

    return device, use_amp, scaler, autocast_device


def train(args):
    # Device setup with mixed precision support
    device, use_amp, scaler, autocast_device = setup_device_with_mixed_precision(args)

    # Transforms (enhanced augmentation)
    if args.no_aug:
        train_transform = v2.Compose([
            v2.Resize((args.imgsz, args.imgsz)),
        ])
    else:
        train_transform = v2.Compose([
            v2.Resize((args.imgsz, args.imgsz)),
            v2.RandomAffine(
                degrees=15,           # Rotation ±15°
                translate=(0.1, 0.1), # Translation 10%
                scale=(0.5, 1.5),     # Scale 50%-150%
                shear=10              # Shear ±10°
            ),
            v2.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.7,
                hue=0.015
            ),
            v2.RandomGrayscale(p=0.1),  # 10% chance grayscale
        ])

    val_transform = v2.Compose([
        v2.Resize((args.imgsz, args.imgsz)),
    ])

    # Datasets
    train_dataset = YOLODataset(args.data, split='train', img_size=args.imgsz, transform=train_transform, mosaic_prob=args.mosaic, disable_aug=args.no_aug, cache=args.cache)
    val_dataset = YOLODataset(args.data, split='val', img_size=args.imgsz, transform=val_transform, mosaic_prob=0.0, cache=args.cache)

    # Weighted sampling for class imbalance
    loader_kwargs = {
        'batch_size': args.batch,
        'num_workers': args.workers,
        'pin_memory': True,
        'persistent_workers': args.workers > 0,
        'prefetch_factor': 2 if args.workers > 0 else None,
        'collate_fn': train_dataset.collate_fn
    }

    if args.class_weighted_sampling:
        print("Creating weighted sampler for class imbalance...")
        train_sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, collate_fn=val_dataset.collate_fn,
                          num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Weighted sampling: {'ENABLED' if args.class_weighted_sampling else 'disabled'}")
    print(f"Disk caching: {'ENABLED' if args.cache else 'disabled'}")

    # Model
    print(f"Loading model: {args.model}")
    model_wrapper = YOLO(args.model)
    model = model_wrapper.model
    model.to(device)

    # Automatic Head Replacement for Fine-tuning
    disable_head_replacement = getattr(args, 'disable_head_replacement', False)
    force_head_replacement = getattr(args, 'force_head_replacement', False)

    if not disable_head_replacement:
        from .head_replacer import replace_yolo_head_if_needed

        try:
            print(f"\n{'='*60}")
            print("CHECKING MODEL ARCHITECTURE COMPATIBILITY")
            print(f"Dataset config: {args.data}")

            was_replaced = False
            if force_head_replacement:
                print("🔄 Force head replacement enabled")
                # Force replacement by temporarily setting up replacer
                from .head_replacer import YOLOHeadReplacer
                replacer = YOLOHeadReplacer(model, args.data, verbose=True)
                if replacer.needs_replacement() or force_head_replacement:
                    model = replacer.replace_classification_head()
                    was_replaced = True
            else:
                # Automatic replacement only if needed
                model, was_replaced = replace_yolo_head_if_needed(model, args.data, verbose=True)

            if was_replaced:
                print("✓ Model architecture updated for French playing cards dataset")
                model.to(device)  # Move to device after replacement
            else:
                print("✓ Model architecture already compatible with dataset")

        except Exception as e:
            print(f"⚠️ Head replacement failed: {e}")
            print("   Continuing with original model (may cause class mismatch issues)")
            print("   Use --disable-head-replacement to skip this check in future")
    else:
        print("⚠️ Head replacement disabled by user flag")

    # Fine-tuning setup
    fine_tune_mode = getattr(args, 'fine_tune_mode', 'full')
    phase1_epochs = getattr(args, 'phase1_epochs', 20)

    print(f"\n{'='*60}")
    print("FINE-TUNING CONFIGURATION")
    print(f"Mode: {fine_tune_mode}")
    print(f"Mixed Precision: {use_amp}")

    # Analyze model structure
    layers_info = analyze_yolo11_layers(model)
    print(f"Total parameters: {layers_info['total_params']:,}")

    # Initialize progressive unfreezing schedule if needed
    schedule = None
    current_phase = None

    if fine_tune_mode == 'progressive':
        schedule = get_two_phase_schedule(args.epochs, phase1_epochs)
        current_phase = 'head-only'
        apply_phase_unfreezing(model, current_phase, verbose=True)
        print(f"\n📅 TRAINING SCHEDULE:")
        for epoch, phase in schedule:
            print(f"  Epoch {epoch:2d}: {phase}")
    elif fine_tune_mode == 'head-only':
        current_phase = 'head-only'
        apply_phase_unfreezing(model, current_phase, verbose=True)
    else:  # full mode
        current_phase = 'full'
        apply_phase_unfreezing(model, current_phase, verbose=True)

    # Optimizer
    if fine_tune_mode == 'full':
        # Use original learning rate for full training
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
        print(f"Using full training with lr={args.lr}")
    else:
        # Use conservative learning rates for fine-tuning
        head_lr = getattr(args, 'head_lr', 1e-3)
        neck_lr = getattr(args, 'neck_lr', 1e-4)
        optimizer = create_conservative_optimizer(model, head_lr=head_lr, neck_lr=neck_lr)
    
    # Loss
    # Fix for Ultralytics loss expecting attribute access for hyperparameters
    if hasattr(model, 'args') and isinstance(model.args, dict):
        from types import SimpleNamespace
        model.args = SimpleNamespace(**model.args)
    
    # Ensure hyperparameters exist
    if not hasattr(model.args, 'box'): model.args.box = 7.5
    if not hasattr(model.args, 'cls'): model.args.cls = 0.5
    if not hasattr(model.args, 'dfl'): model.args.dfl = 1.5
        
    loss_fn = v8DetectionLoss(model_wrapper.model)

    # Training Loop
    os.makedirs(args.project, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Check if we need to switch phases for progressive unfreezing
        if schedule and fine_tune_mode == 'progressive':
            for switch_epoch, new_phase in schedule:
                if epoch == switch_epoch and current_phase != new_phase:
                    print(f"\n🔄 SWITCHING TO PHASE: {new_phase}")
                    apply_phase_unfreezing(model, new_phase, verbose=True)
                    current_phase = new_phase

                    # Recreate optimizer for new phase
                    head_lr = getattr(args, 'head_lr', 1e-3)
                    neck_lr = getattr(args, 'neck_lr', 1e-4)
                    optimizer = create_conservative_optimizer(model, head_lr=head_lr, neck_lr=neck_lr)
                    break

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} ({current_phase})")
        total_loss = 0

        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_amp and autocast_device:
                with torch.autocast(device_type=autocast_device):
                    preds = model(imgs)

                    batch_data = {
                        "batch_idx": targets[:, 0],
                        "cls": targets[:, 1].view(-1, 1),
                        "bboxes": targets[:, 2:],
                        "device": device,
                        "img": imgs
                    }

                    loss, loss_items = loss_fn(preds, batch_data)

                    # Ensure loss is scalar
                    if loss.ndim > 0:
                        loss = loss.sum() / imgs.shape[0]

                # Backward pass with gradient scaling if available
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                # Standard precision forward pass
                preds = model(imgs)

                batch_data = {
                    "batch_idx": targets[:, 0],
                    "cls": targets[:, 1].view(-1, 1),
                    "bboxes": targets[:, 2:],
                    "device": device,
                    "img": imgs
                }

                loss, loss_items = loss_fn(preds, batch_data)

                # Ensure loss is scalar
                if loss.ndim > 0:
                    loss = loss.sum() / imgs.shape[0]

                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} ({current_phase})")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                preds = model(imgs)
                batch_data = {
                    "batch_idx": targets[:, 0],
                    "cls": targets[:, 1].view(-1, 1),
                    "bboxes": targets[:, 2:],
                    "device": device,
                    "img": imgs
                }
                loss, _ = loss_fn(preds, batch_data)
                if loss.ndim > 0:
                    loss = loss.sum() / imgs.shape[0]
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"{args.project}/custom_yolo_epoch_{epoch+1}.pt")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = f"{args.project}/best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"✓ New best model saved: {best_path} (val_loss: {best_val_loss:.4f})")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    return best_val_loss
