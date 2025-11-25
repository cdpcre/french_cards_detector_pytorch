import os
import shutil
import pandas as pd
import yaml
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import zipfile

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
UNIFIED_DIR = DATASETS_DIR / "unified"
HUGO_DIR = DATASETS_DIR / "hugopaigneau:playing-cards-dataset"
ANDY_DIR = DATASETS_DIR / "andy8744:playing-cards-object-detection-dataset"
VDNT_ZIP = DATASETS_DIR / "vdntdesai11:playing-cards.zip"
VDNT_DIR = DATASETS_DIR / "vdntdesai11_extracted"
JAY_DIR = DATASETS_DIR / "jaypradipshah:the-complete-playing-card-dataset"

# Canonical Class List (Alphabetical order as seen in Andy8744)
CANONICAL_CLASSES = [
    '10c', '10d', '10h', '10s', 
    '2c', '2d', '2h', '2s', 
    '3c', '3d', '3h', '3s', 
    '4c', '4d', '4h', '4s', 
    '5c', '5d', '5h', '5s', 
    '6c', '6d', '6h', '6s', 
    '7c', '7d', '7h', '7s', 
    '8c', '8d', '8h', '8s', 
    '9c', '9d', '9h', '9s', 
    'Ac', 'Ad', 'Ah', 'As', 
    'Jc', 'Jd', 'Jh', 'Js', 
    'Kc', 'Kd', 'Kh', 'Ks', 
    'Qc', 'Qd', 'Qh', 'Qs',
    'joker'
]

CLASS_TO_ID = {name: i for i, name in enumerate(CANONICAL_CLASSES)}

def setup_directories():
    """Create necessary directories for the unified dataset."""
    for split in ['train', 'val', 'test']:
        (UNIFIED_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (UNIFIED_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def convert_bbox_to_yolo(size, box):
    """Convert (xmin, ymin, xmax, ymax) to (x_center, y_center, width, height) normalized."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_hugopaigneau():
    """Process HugoPaigneau dataset (CSV to YOLO)."""
    print("Processing HugoPaigneau dataset...")
    csv_path = HUGO_DIR / "train_cards_label.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Skipping.")
        return []

    df = pd.read_csv(csv_path)
    # Group by filename to handle multiple objects per image
    grouped = df.groupby('filename')

    # We'll put all Hugo images into 'train' initially, then split later if needed.
    # Or better, we just collect all data and split at the end.
    # For now, let's just write to a temporary list of (image_path, label_content)
    
    data_items = []

    for filename, group in grouped:
        src_img_path = HUGO_DIR / "train" / filename
        if not src_img_path.exists():
            # Try 'test' folder if not in 'train'
            src_img_path = HUGO_DIR / "test" / filename
        
        if not src_img_path.exists():
            continue

        img_w = group.iloc[0]['width']
        img_h = group.iloc[0]['height']
        
        label_lines = []
        for _, row in group.iterrows():
            class_name = row['class']
            if class_name not in CLASS_TO_ID:
                continue
            
            class_id = CLASS_TO_ID[class_name]
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            
            # Convert to YOLO
            bbox = (xmin, xmax, ymin, ymax)
            yolo_bbox = convert_bbox_to_yolo((img_w, img_h), bbox)
            
            label_lines.append(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
        
        data_items.append({
            'src_img': src_img_path,
            'labels': label_lines,
            'dataset': 'hugopaigneau'
        })
        
    return data_items

def process_andy8744():
    """Process Andy8744 dataset (Already YOLO, check mapping)."""
    print("Processing Andy8744 dataset...")
    
    # Load Andy's data.yaml to check class names
    yaml_path = ANDY_DIR / "data.yaml"
    if not yaml_path.exists():
        print("Warning: Andy8744 data.yaml not found.")
        return []

    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    andy_names = data_config['names']
    # Create a mapping from Andy's ID to Canonical ID
    andy_id_map = {}
    for i, name in enumerate(andy_names):
        if name in CLASS_TO_ID:
            andy_id_map[i] = CLASS_TO_ID[name]
    
    data_items = []
    
    # Iterate over train, valid, test folders
    for split in ['train', 'valid', 'test']:
        img_dir = ANDY_DIR / split / "images"
        lbl_dir = ANDY_DIR / split / "labels"
        
        if not img_dir.exists():
            continue
            
        for img_path in img_dir.glob("*.jpg"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
                
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                cls_id = int(parts[0])
                
                if cls_id in andy_id_map:
                    new_cls_id = andy_id_map[cls_id]
                    new_lines.append(f"{new_cls_id} " + " ".join(parts[1:]))
            
            data_items.append({
                'src_img': img_path,
                'labels': new_lines,
                'dataset': 'andy8744'
            })
            
    return data_items

def process_vdntdesai11():
    """Process vdntdesai11 dataset (Unzip and Map)."""
    print("Processing vdntdesai11 dataset...")
    
    if not VDNT_ZIP.exists():
        print("vdntdesai11 zip not found.")
        return []
        
    if not VDNT_DIR.exists():
        print("Unzipping vdntdesai11...")
        with zipfile.ZipFile(VDNT_ZIP, 'r') as zip_ref:
            zip_ref.extractall(VDNT_DIR)
            
    # Hardcoded class list for vdntdesai11 based on inspection (K before Q)
    # Order: Suits (C, D, H, S), Ranks (A, 2..10, J, K, Q)
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'K', 'Q']
    suits = ['c', 'd', 'h', 's']
    names = []
    for s in suits:
        for r in ranks:
            names.append(f"{r}{s}")
            
    print("Using hardcoded class mapping for vdntdesai11 (K before Q swap).")

    # Map IDs
    id_map = {}
    for i, name in enumerate(names):
        # Normalize name (e.g., '10H' -> '10h')
        # Check if name is in CANONICAL_CLASSES
        # Try exact match first
        if name in CLASS_TO_ID:
            id_map[i] = CLASS_TO_ID[name]
        else:
            # Try lowercase
            if name.lower() in CLASS_TO_ID:
                id_map[i] = CLASS_TO_ID[name.lower()]
            else:
                print(f"Warning: Class '{name}' not found in canonical classes.")
            
    data_items = []
    # Assume standard YOLO structure inside extracted folder
    # We'll search recursively for images
    for img_path in VDNT_DIR.rglob("*.jpg"):
        # Find corresponding label
        # Assuming label is in a 'labels' folder parallel to 'images' or same folder
        # Strategy: look for .txt with same stem
        
        # Check if it's in an 'images' folder
        if 'images' in img_path.parts:
            # Try replacing 'images' with 'labels'
            parts = list(img_path.parts)
            idx = parts.index('images')
            parts[idx] = 'labels'
            lbl_path = Path(*parts).with_suffix('.txt')
        else:
            lbl_path = img_path.with_suffix('.txt')
            
        if not lbl_path.exists():
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            cls_id = int(parts[0])
            
            if cls_id in id_map:
                new_cls_id = id_map[cls_id]
                new_lines.append(f"{new_cls_id} " + " ".join(parts[1:]))
        
        if new_lines:
            data_items.append({
                'src_img': img_path,
                'labels': new_lines,
                'dataset': 'vdntdesai11'
            })

    return data_items


def process_jaypradipshah():
    """Process jaypradipshah dataset (YOLO format with specific mapping)."""
    print("Processing jaypradipshah dataset...")
    
    # Paths
    img_dir = JAY_DIR / "Images" / "Images"
    lbl_dir = JAY_DIR / "YOLO_Annotations" / "YOLO_Annotations"
    
    if not img_dir.exists() or not lbl_dir.exists():
        print("Warning: jaypradipshah directories not found. Skipping.")
        return []

    # Class mapping based on annotation.json inspection
    # The dataset uses a specific order. We map it to our CANONICAL_CLASSES.
    # The dataset classes are: AS, AC, AD, AH, 2S, 2C, ... JOKER
    # We need to map these to our canonical names (e.g., AS -> As, 2S -> 2s, JOKER -> joker)
    
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    suits = ['S', 'C', 'D', 'H']
    
    jay_classes = []
    for r in ranks:
        for s in suits:
            jay_classes.append(f"{r}{s}")
    jay_classes.append("JOKER")
    
    # Map from Jay's index (0-based) to Canonical ID
    jay_id_map = {}
    for i, name in enumerate(jay_classes):
        canonical_name = name
        
        # Handle JOKER
        if name == 'JOKER':
            canonical_name = 'joker'
        else:
            # Handle standard cards: Rank + Suit_lower
            # e.g., AS -> As, 10H -> 10h
            rank = name[:-1]
            suit = name[-1]
            canonical_name = f"{rank}{suit.lower()}"
            
        if canonical_name in CLASS_TO_ID:
            jay_id_map[i] = CLASS_TO_ID[canonical_name]
        else:
            print(f"Warning: Class '{name}' (canonical: '{canonical_name}') not found in CANONICAL_CLASSES.")

    data_items = []
    
    # Iterate over images
    # Since there are many files, we use glob
    for img_path in img_dir.glob("*.jpg"):
        # Find corresponding label
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        
        if not lbl_path.exists():
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            try:
                cls_id = int(parts[0])
                
                if cls_id in jay_id_map:
                    new_cls_id = jay_id_map[cls_id]
                    new_lines.append(f"{new_cls_id} " + " ".join(parts[1:]))
            except ValueError:
                continue
        
        if new_lines:
            data_items.append({
                'src_img': img_path,
                'labels': new_lines,
                'dataset': 'jaypradipshah'
            })
            
    return data_items

def main():
    setup_directories()
    
    all_data = []
    all_data.extend(process_hugopaigneau())
    all_data.extend(process_andy8744())
    all_data.extend(process_vdntdesai11())
    all_data.extend(process_jaypradipshah())
    
    print(f"Total images found: {len(all_data)}")
    
    # Split data
    train_data, test_val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=42)
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, items in splits.items():
        print(f"Writing {split_name} split: {len(items)} images...")
        for i, item in enumerate(items):
            # Create a unique filename to avoid collisions
            # e.g., hugopaigneau_original_name.jpg
            original_name = item['src_img'].name
            dataset_prefix = item['dataset']
            new_filename = f"{dataset_prefix}_{original_name}"
            
            dst_img = UNIFIED_DIR / 'images' / split_name / new_filename
            dst_lbl = UNIFIED_DIR / 'labels' / split_name / (Path(new_filename).stem + ".txt")
            
            # Copy image
            shutil.copy2(item['src_img'], dst_img)
            
            # Write labels
            with open(dst_lbl, 'w') as f:
                f.write("\n".join(item['labels']))
                
    # Create data.yaml for unified dataset
    yaml_content = {
        'path': str(UNIFIED_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CANONICAL_CLASSES),
        'names': CANONICAL_CLASSES
    }
    
    with open(UNIFIED_DIR / "data.yaml", 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print("Unified dataset created successfully!")

if __name__ == "__main__":
    main()
