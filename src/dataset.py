import os
import cv2
import torch
import yaml
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import tv_tensors
from .utils import xywhn2xyxy, xyxy2xywhn

class YOLODataset(Dataset):
    def __init__(self, yaml_path, split='train', img_size=640, transform=None, mosaic_prob=0.0, disable_aug=False, cache=False):
        self.img_size = img_size
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.split = split
        self.disable_aug = disable_aug
        self.cache = cache
        
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
                    lines = [x.strip().split() for x in f.read().strip().splitlines() if len(x.strip())]
                    # Filter for lines that have exactly 5 elements (class_id + 4 bbox coords)
                    valid_lines = [line for line in lines if len(line) == 5]
                    l = np.array(valid_lines, dtype=np.float32) if len(valid_lines) else np.zeros((0, 5), dtype=np.float32)
                self.labels.append(l)
            else:
                self.labels.append(np.zeros((0, 5), dtype=np.float32))

    def __len__(self):
        return len(self.img_paths)

    def load_image_and_label(self, index):
        img_path = self.img_paths[index]
        
        # Caching logic
        if self.cache:
            cache_path = img_path.with_suffix('.npy')
            if cache_path.exists():
                try:
                    img = np.load(cache_path)
                except Exception as e:
                    print(f"Warning: Failed to load cache {cache_path}: {e}. Reloading from source.")
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    np.save(cache_path, img)
            else:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                np.save(cache_path, img)
        else:
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
            
            # Clip boxes to mosaic canvas (2s x 2s)
            np.clip(result_bboxes[:, 0], 0, 2 * s, out=result_bboxes[:, 0])
            np.clip(result_bboxes[:, 1], 0, 2 * s, out=result_bboxes[:, 1])
            np.clip(result_bboxes[:, 2], 0, 2 * s, out=result_bboxes[:, 2])
            np.clip(result_bboxes[:, 3], 0, 2 * s, out=result_bboxes[:, 3])

            # Filter degenerate boxes (use larger threshold for mosaic)
            w_box = result_bboxes[:, 2] - result_bboxes[:, 0]
            h_box = result_bboxes[:, 3] - result_bboxes[:, 1]
            keep = (w_box > 4) & (h_box > 4)  # Increased threshold for mosaic
            result_bboxes = result_bboxes[keep]
            result_cls = result_cls[keep]

        # Crop mosaic to final image size (s x s from center of 2s x 2s)
        crop_x = s // 2
        crop_y = s // 2
        result_img = result_img[crop_y:crop_y+s, crop_x:crop_x+s]

        # Adjust boxes for the crop
        if len(result_bboxes) > 0:
            result_bboxes[:, [0, 2]] -= crop_x  # subtract x offset
            result_bboxes[:, [1, 3]] -= crop_y  # subtract y offset

            # Clip boxes to final image size
            np.clip(result_bboxes[:, 0], 0, s, out=result_bboxes[:, 0])
            np.clip(result_bboxes[:, 1], 0, s, out=result_bboxes[:, 1])
            np.clip(result_bboxes[:, 2], 0, s, out=result_bboxes[:, 2])
            np.clip(result_bboxes[:, 3], 0, s, out=result_bboxes[:, 3])

            # Filter boxes that are now outside or too small after crop
            w_box = result_bboxes[:, 2] - result_bboxes[:, 0]
            h_box = result_bboxes[:, 3] - result_bboxes[:, 1]
            keep = (w_box > 2) & (h_box > 2) & (result_bboxes[:, 0] < s) & (result_bboxes[:, 1] < s)
            result_bboxes = result_bboxes[keep]
            result_cls = result_cls[keep]

        return result_img, result_bboxes, result_cls

    def apply_joker_augmentation(self, img, bboxes, cls):
        """Apply aggressive augmentation to samples with joker cards."""
        # Check if there are joker cards in the image (class 52)
        if len(cls) == 0 or 52 not in cls:
            return img, bboxes, cls

        # Aggressive augmentations specifically for joker (bbox-invariant only)
        augmentations = [
            lambda x: cv2.GaussianBlur(x, (5, 5), 0),
            lambda x: np.clip(x * np.random.uniform(0.7, 1.3), 0, 255).astype(np.uint8),
            lambda x: np.clip(x + np.random.normal(0, 10, x.shape), 0, 255).astype(np.uint8),
        ]

        # Apply 1-2 random augmentations
        num_augs = np.random.randint(1, min(3, len(augmentations) + 1))
        selected_augs = np.random.choice(len(augmentations), num_augs, replace=False)

        for aug_idx in selected_augs:
            img = augmentations[aug_idx](img)

        return img, bboxes, cls

    def __getitem__(self, index):
        if self.split == 'train' and np.random.rand() < self.mosaic_prob:
            img, bboxes, cls = self.load_mosaic(index)
        else:
            img, bboxes, cls = self.load_image_and_label(index)

        # Apply joker-specific augmentation (only for training)
        if self.split == 'train' and not self.disable_aug:
            img, bboxes, cls = self.apply_joker_augmentation(img, bboxes, cls)

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


def create_weighted_sampler(dataset):
    """Create sampler that oversamples minority classes (especially joker)."""
    from torch.utils.data.sampler import WeightedRandomSampler

    class_counts = {}

    # Count samples per class
    for idx in range(len(dataset)):
        label = dataset.labels[idx]
        if len(label) > 0:
            cls_id = int(label[0, 0])
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    # Calculate weights per sample (inversely proportional to frequency)
    sample_weights = []
    for idx in range(len(dataset)):
        label = dataset.labels[idx]
        if len(label) > 0:
            cls_id = int(label[0, 0])
            # Use square root to balance without oversampling too much
            weight = 1.0 / (class_counts[cls_id] ** 0.5)
        else:
            weight = 1.0
        sample_weights.append(weight)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
