import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO

def train(args):
    # Ensure project directory exists
    os.makedirs(args.project, exist_ok=True)

    # Load a model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

    # Train the model
    print(f"Starting FAST training for {args.epochs} epochs...")
    print(f"  - Image Size: {args.imgsz}")
    print(f"  - Workers: {args.workers}")
    print(f"  - Freeze Layres: {args.freeze}")
    
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True, # Overwrite existing run if name is same
        "pretrained": True,
        "resume": args.resume,
        "freeze": args.freeze,
        "multi_scale": args.multi_scale,
        "workers": args.workers,
        "cache": "disk" if not args.no_cache else False,
        "cos_lr": not args.no_cos_lr,
        # NMS settings to avoid timeout warnings
        "iou": args.iou,
        "conf": args.conf,
        "max_det": args.max_det,
        # Flip augmentations (disabled by default)
        "fliplr": args.fliplr,
        "flipud": args.flipud,
    }

    if args.no_aug:
        print("Disabling data augmentation...")
        train_args.update({
            "degrees": 0,
            "translate": 0,
            "scale": 0,
            "shear": 0,
            "perspective": 0,
            "flipud": 0,
            "fliplr": 0,
            "mosaic": 0,
            "mixup": 0,
            "copy_paste": 0,
            "hsv_h": 0,
            "hsv_s": 0,
            "hsv_v": 0,
        })
    
    if args.no_mosaic:
        print("Disabling MOSAIC augmentation...")
        train_args['mosaic'] = 0.0

    results = model.train(**train_args)
    print("Training completed.")

    # Validate the model
    print("Validating model...")
    metrics = model.val()
    print(f"Validation results: {metrics}")

    # Export the model
    if args.export:
        print("Exporting model...")
        success = model.export(format=args.export, device=args.device)
        print(f"Export success: {success}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Finetune YOLOv11 model locally (FAST CONFIG)')
    
    # Model and Data
    parser.add_argument('--model', type=str, default='models/yolo11n.pt', help='Path to model weights (default: models/yolo11n.pt)')
    parser.add_argument('--data', type=str, default='datasets/unified/data.yaml', help='Path to data config (default: datasets/unified/data.yaml)')
    
    # Training Hyperparameters - OPTIMIZED DEFAULTS
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size (default: 8, optimized for M4)')
    parser.add_argument('--imgsz', type=int, default=480, help='Image size (Reduced to 480 for speed)')
    parser.add_argument('--device', type=str, default='mps', help='Device to run on, i.e. 0, 0,1,2,3 or cpu')
    parser.add_argument('--freeze', type=int, default=15, help='Number of layers to freeze (15 freezes backbone+neck, faster)')
    parser.add_argument('--multi_scale', type=bool, default=False, help='Use multi scale training (Disabled for speed)')    
    
    # Run Configuration
    parser.add_argument('--project', type=str, default='runs/train', help='Directory to save results')
    parser.add_argument('--name', type=str, default='fast_exp', help='Name of the run')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers (0 for Mac/MPS optimization)')
    parser.add_argument('--no-aug', action='store_true', help='Disable data augmentation')
    
    # Export
    parser.add_argument('--export', type=str, default="onnx", help='Export format (e.g., onnx, torchscript) after training')
    parser.add_argument('--no-cache', action='store_true', help='Disable disk caching (default: cache="disk")')
    parser.add_argument('--no-cos-lr', action='store_true', help='Disable cosine learning rate scheduler (default: True)')
    parser.add_argument('--no-mosaic', action='store_true', default=True, help='Disable mosaic augmentation (default: True for faster training)')
    
    # NMS settings (to avoid NMS timeout warnings on M4/MPS)
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS (default: 0.5, lower = faster)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5, higher = faster)')
    parser.add_argument('--max-det', type=int, default=100, help='Max detections per image (default: 100, lower = faster)')
    
    # Flip augmentations (disabled by default for cards - orientation matters)
    parser.add_argument('--fliplr', type=float, default=0.0, help='Horizontal flip probability (default: 0.0, disabled)')
    parser.add_argument('--flipud', type=float, default=0.0, help='Vertical flip probability (default: 0.0, disabled)')

    args = parser.parse_args()
    
    train(args)
