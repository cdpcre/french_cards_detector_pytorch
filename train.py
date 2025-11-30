import os
import cv2
import torch
import yaml
import random
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from tqdm import tqdm

# Helper Functions
def xywhn2xyxy(x, w=640, h=640):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] absolute
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2)  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2)  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2)  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2)  # bottom right y
    return y

def xyxy2xywhn(x, w=640, h=640):
    # Convert nx4 boxes from [x1, y1, x2, y2] absolute to [x, y, w, h] normalized
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

class YOLODataset(Dataset):
    def __init__(self, yaml_path, split='train', img_size=640, transform=None, mosaic_prob=0.0):
        self.img_size = img_size
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.split = split
        
        # Load yaml
        with open(yaml_path, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
            
        self.root = Path(yaml_path).parent
        # Handle path relative to yaml or absolute
        if os.path.isabs(self.data_cfg[split]):
             img_dir = Path(self.data_cfg[split])
        else:
             img_dir = self.root / self.data_cfg[split]
             
        self.img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        
        # Cache labels
        self.labels = []
        for img_path in self.img_paths:
            label_path = img_path.parent.parent.parent / 'labels' / img_path.parent.name / (img_path.stem + ".txt")
            if label_path.exists():
                with open(label_path, 'r') as f:
                    l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    l = np.array(l, dtype=np.float32) if len(l) else np.zeros((0, 5), dtype=np.float32)
                self.labels.append(l)
            else:
                self.labels.append(np.zeros((0, 5), dtype=np.float32))

    def __len__(self):
        return len(self.img_paths)

    def load_image_and_label(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        label = self.labels[index].copy()
        bboxes = label[:, 1:] if len(label) > 0 else np.zeros((0, 4), dtype=np.float32)
        cls = label[:, 0] if len(label) > 0 else np.zeros((0,), dtype=np.float32)
        
        # Convert xywhn to xyxy absolute
        if len(bboxes) > 0:
            bboxes = xywhn2xyxy(bboxes, w, h)
            
        return img, bboxes, cls

    def load_mosaic(self, index):
        # YOLO Mosaic implementation
        s = self.img_size
        xc = int(random.uniform(-s // 2, 2 * s + s // 2))
        yc = int(random.uniform(-s // 2, 2 * s + s // 2))
        
        indices = [index] + random.choices(range(len(self)), k=3)
        random.shuffle(indices)
        
        result_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        result_bboxes = []
        result_cls = []
        
        for i, idx in enumerate(indices):
            img, bboxes, cls = self.load_image_and_label(idx)
            h, w = img.shape[:2]
            
            # Define placement coordinates (unclamped)
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = xc - w, yc - h, xc, yc
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, yc - h, xc + w, yc
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = xc - w, yc, xc, yc + h
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, xc + w, yc + h

            # Clamp to canvas
            x1a_c = max(0, min(x1a, 2 * s))
            x2a_c = max(0, min(x2a, 2 * s))
            y1a_c = max(0, min(y1a, 2 * s))
            y2a_c = max(0, min(y2a, 2 * s))
            
            w_c = x2a_c - x1a_c
            h_c = y2a_c - y1a_c
            
            if w_c <= 0 or h_c <= 0:
                continue
                
            # Calculate source coordinates
            x1b = x1a_c - x1a
            y1b = y1a_c - y1a
            x2b = x1b + w_c
            y2b = y1b + h_c
            
            result_img[y1a_c:y2a_c, x1a_c:x2a_c] = img[y1b:y2b, x1b:x2b]
            
            if len(bboxes) > 0:
                # Adjust bboxes
                bboxes[:, [0, 2]] += x1a
                bboxes[:, [1, 3]] += y1a
                result_bboxes.append(bboxes)
                result_cls.append(cls)
                
        if len(result_bboxes) > 0:
            result_bboxes = np.concatenate(result_bboxes, 0)
            result_cls = np.concatenate(result_cls, 0)
            
            # Clip boxes to image
            np.clip(result_bboxes[:, 0], 0, 2 * s, out=result_bboxes[:, 0])
            np.clip(result_bboxes[:, 1], 0, 2 * s, out=result_bboxes[:, 1])
            np.clip(result_bboxes[:, 2], 0, 2 * s, out=result_bboxes[:, 2])
            np.clip(result_bboxes[:, 3], 0, 2 * s, out=result_bboxes[:, 3])
            
            # Filter degenerate boxes
            w_box = result_bboxes[:, 2] - result_bboxes[:, 0]
            h_box = result_bboxes[:, 3] - result_bboxes[:, 1]
            keep = (w_box > 2) & (h_box > 2)
            result_bboxes = result_bboxes[keep]
            result_cls = result_cls[keep]
            
        return result_img, result_bboxes, result_cls

    def __getitem__(self, index):
        if self.split == 'train' and np.random.rand() < self.mosaic_prob:
            img, bboxes, cls = self.load_mosaic(index)
        else:
            img, bboxes, cls = self.load_image_and_label(index)
            
        # Prepare for transforms
        # Convert to torch tensors
        # Ensure CHW format
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = tv_tensors.Image(img)
        
        # BoundingBoxes requires shape [N, 4]
        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            cls = torch.zeros((0,), dtype=torch.float32)
        else:
            bboxes = torch.from_numpy(bboxes).float()
            cls = torch.from_numpy(cls).float()
            
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=img.shape[-2:])
        
        if self.transform:
            img, bboxes, cls = self.transform(img, bboxes, cls)
            
        # Normalize image 0-1
        img = img.float() / 255.0
        
        # Convert boxes back to xywhn for YOLO loss
        h, w = img.shape[-2:]
        if len(bboxes) > 0:
            bboxes_norm = xyxy2xywhn(bboxes, w, h)
            # Create target tensor [idx, cls, x, y, w, h]
            # Note: idx will be added in collate_fn
            targets = torch.cat((cls.unsqueeze(1), bboxes_norm), dim=1)
        else:
            targets = torch.zeros((0, 5))
            
        return img, targets
        
    def collate_fn(self, batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs, 0)
        
        # Add batch index to targets
        new_targets = []
        for i, t in enumerate(targets):
            if t.shape[0] > 0:
                idx = torch.full((t.shape[0], 1), i)
                new_targets.append(torch.cat((idx, t), 1))
        
        if new_targets:
            targets = torch.cat(new_targets, 0)
        else:
            targets = torch.zeros((0, 6))
            
        return imgs, targets

def train(args):
    # Device
    device = torch.device(args.device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms
    train_transform = v2.Compose([
        v2.Resize((args.imgsz, args.imgsz)),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.5, 1.5)),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015),
    ])

    val_transform = v2.Compose([
        v2.Resize((args.imgsz, args.imgsz)),
    ])
    
    # Datasets
    train_dataset = YOLODataset(args.data, split='train', img_size=args.imgsz, transform=train_transform, mosaic_prob=args.mosaic)
    val_dataset = YOLODataset(args.data, split='val', img_size=args.imgsz, transform=val_transform, mosaic_prob=0.0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, collate_fn=val_dataset.collate_fn, num_workers=args.workers)

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    
    # Model
    print(f"Loading model: {args.model}")
    model_wrapper = YOLO(args.model)
    model = model_wrapper.model
    model.to(device)
    
    # Force gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    
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
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
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
                loss = loss.sum()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
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
                    loss = loss.sum()
                val_loss += loss.item()
                
        print(f"Val Loss: {val_loss / len(val_loader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"{args.project}/custom_yolo_epoch_{epoch+1}.pt")
        
    return val_loss / len(val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv11 with custom PyTorch loop')
    parser.add_argument('--data', type=str, default='datasets/unified/data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (default: 5e-4)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic probability (default: 1.0)')
    parser.add_argument('--device', type=str, default='mps', help='Device (cuda/mps/cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Save directory')
    parser.add_argument('--workers', type=int, default=0, help='Num workers')
    parser.add_argument('--class-weighted-sampling', action='store_true', default=False, help='Use weighted sampling for class imbalance')

    args = parser.parse_args()
    train(args)
