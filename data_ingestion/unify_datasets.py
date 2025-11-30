import shutil
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
UNIFIED_DIR = DATASETS_DIR / "unified"
HUGO_DIR = DATASETS_DIR / "hugopaigneau:playing-cards-dataset"
ANDY_DIR = DATASETS_DIR / "andy8744:playing-cards-object-detection-dataset"
JAY_DIR = DATASETS_DIR / "jaypradipshah:the-complete-playing-card-dataset"
CARDS_V1I_DIR = DATASETS_DIR / "Cards.v1i.yolov11"
PLAYING_CARDS_V2I_DIR = DATASETS_DIR / "Playing Cards.v2i.yolov11"
ROBOFLOW_PLAYING_CARDS_DIR = DATASETS_DIR / "Playing Cards.v4-yolov8n.yolov11"

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
        # Ensure filename is a str before joining with Path to avoid TypeError
        filename_str = str(filename)
        src_img_path = HUGO_DIR / "train" / filename_str
        if not src_img_path.exists():
            # Try 'test' folder if not in 'train'
            src_img_path = HUGO_DIR / "test" / filename_str
        
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
                if not parts:
                    continue
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
            if not parts:
                continue
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



def process_cards_v1i():
    """Process Cards.v1i.yolov11 dataset."""
    print("Processing Cards.v1i.yolov11 dataset...")
    
    yaml_path = CARDS_V1I_DIR / "data.yaml"
    if not yaml_path.exists():
        print("Warning: Cards.v1i.yolov11 data.yaml not found.")
        return []

    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    src_names = data_config['names']
    # Create mapping
    id_map = {}
    for i, name in enumerate(src_names):
        # Format is RankSuit (e.g. 10C). Canonical is RankSuit_lower (e.g. 10c)
        # Check if last char is suit
        if len(name) >= 2:
            rank = name[:-1]
            suit = name[-1]
            canonical_name = f"{rank}{suit.lower()}"
            
            if canonical_name in CLASS_TO_ID:
                id_map[i] = CLASS_TO_ID[canonical_name]
            else:
                print(f"Warning: Class '{name}' (canonical: '{canonical_name}') not found in canonical classes.")
        else:
             print(f"Warning: Unexpected class name format '{name}'")

    data_items = []
    # Iterate over train, valid, test folders (if they exist)
    for split in ['train', 'valid', 'test', 'val']:
        img_dir = CARDS_V1I_DIR / split / "images"
        lbl_dir = CARDS_V1I_DIR / split / "labels"
        
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
                if not parts:
                    continue
                try:
                    cls_id = int(parts[0])
                    if cls_id in id_map:
                        new_cls_id = id_map[cls_id]
                        new_lines.append(f"{new_cls_id} " + " ".join(parts[1:]))
                except ValueError:
                    continue
            
            if new_lines:
                data_items.append({
                    'src_img': img_path,
                    'labels': new_lines,
                    'dataset': 'cards_v1i'
                })
                
    return data_items

def process_playing_cards_v2i():
    """Process Playing Cards.v2i.yolov11 dataset."""
    print("Processing Playing Cards.v2i.yolov11 dataset...")
    
    yaml_path = PLAYING_CARDS_V2I_DIR / "data.yaml"
    if not yaml_path.exists():
        print("Warning: Playing Cards.v2i.yolov11 data.yaml not found.")
        return []

    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    src_names = data_config['names']
    # Create mapping
    id_map = {}
    for i, name in enumerate(src_names):
        if name == 'joker':
            canonical_name = 'joker'
        elif len(name) >= 2:
            rank = name[:-1]
            suit = name[-1]
            canonical_name = f"{rank}{suit.lower()}"
        else:
            print(f"Warning: Unexpected class name format '{name}'")
            continue
            
        if canonical_name in CLASS_TO_ID:
            id_map[i] = CLASS_TO_ID[canonical_name]
        else:
            print(f"Warning: Class '{name}' (canonical: '{canonical_name}') not found in canonical classes.")

    data_items = []
    for split in ['train', 'valid', 'test', 'val']:
        img_dir = PLAYING_CARDS_V2I_DIR / split / "images"
        lbl_dir = PLAYING_CARDS_V2I_DIR / split / "labels"
        
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
                if not parts:
                    continue
                try:
                    cls_id = int(parts[0])
                    if cls_id in id_map:
                        new_cls_id = id_map[cls_id]
                        new_lines.append(f"{new_cls_id} " + " ".join(parts[1:]))
                except ValueError:
                    continue
            
            if new_lines:
                data_items.append({
                    'src_img': img_path,
                    'labels': new_lines,
                    'dataset': 'playing_cards_v2i'
                })
                
    return data_items


def process_roboflow_playing_cards():
    """Process Roboflow Playing Cards dataset (focus on joker class)."""
    print("Processing Roboflow Playing Cards dataset...")

    yaml_path = ROBOFLOW_PLAYING_CARDS_DIR / "data.yaml"

    if not yaml_path.exists():
        print(f"Warning: {yaml_path} not found. Skipping Roboflow dataset.")
        return []

    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    src_names = data_config['names']
    id_map = {}

    # Map classes from Roboflow to canonical
    for i, name in enumerate(src_names):
        if name.lower() == 'joker':
            canonical_name = 'joker'
        elif len(name) >= 2:
            rank = name[:-1]
            suit = name[-1].lower()
            canonical_name = f"{rank}{suit}"
        else:
            continue

        if canonical_name in CLASS_TO_ID:
            id_map[i] = CLASS_TO_ID[canonical_name]

    data_items = []

    # Process all splits
    for split in ['train', 'valid', 'test', 'val']:
        img_dir = ROBOFLOW_PLAYING_CARDS_DIR / split / "images"
        lbl_dir = ROBOFLOW_PLAYING_CARDS_DIR / split / "labels"

        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*.jpg"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            with open(lbl_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            has_joker = False

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    cls_id = int(parts[0])
                    if cls_id in id_map:
                        new_cls_id = id_map[cls_id]
                        new_lines.append(f"{new_cls_id} " + " ".join(parts[1:]))
                        if new_cls_id == 52:  # joker
                            has_joker = True
                except ValueError:
                    continue

            # Prioritize images with joker (oversampling)
            if new_lines:
                item = {
                    'src_img': img_path,
                    'labels': new_lines,
                    'dataset': 'roboflow_playing_cards'
                }
                data_items.append(item)

                # If contains joker, duplicate for oversampling
                if has_joker:
                    data_items.append(item)  # Add 2x for joker samples

    print(f"Processed {len(data_items)} items from Roboflow dataset (with joker oversampling)")
    return data_items


def main():
    setup_directories()

    all_data = []
    all_data.extend(process_hugopaigneau())
    all_data.extend(process_andy8744())
    all_data.extend(process_jaypradipshah())
    all_data.extend(process_cards_v1i())
    all_data.extend(process_playing_cards_v2i())
    all_data.extend(process_roboflow_playing_cards())  # New dataset with joker oversampling

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
