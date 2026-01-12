import shutil
import yaml
import random
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
SOURCE_DIR = DATASETS_DIR / "unified"
TARGET_DIR = DATASETS_DIR / "unified_small"
JOKER_CLASS_ID = 52
# KEEP_RATIO is no longer used

def main():
    print(f"Creating reduced dataset from {SOURCE_DIR} to {TARGET_DIR}...")
    
    if TARGET_DIR.exists():
        print(f"Target directory {TARGET_DIR} already exists. Removing...")
        shutil.rmtree(TARGET_DIR)
    
    TARGET_DIR.mkdir(parents=True)
    
    # Copy data.yaml and update path
    src_yaml = SOURCE_DIR / "data.yaml"
    dst_yaml = TARGET_DIR / "data.yaml"
    
    with open(src_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        
    data_config['path'] = str(TARGET_DIR.absolute())
    
    with open(dst_yaml, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)

    # Global stats
    total_original_images = 0
    total_kept_images = 0
    
    # Process splits
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        src_img_dir = SOURCE_DIR / "images" / split
        src_lbl_dir = SOURCE_DIR / "labels" / split
        
        dst_img_dir = TARGET_DIR / "images" / split
        dst_lbl_dir = TARGET_DIR / "labels" / split
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        if not src_img_dir.exists():
            continue
            
        images = sorted(list(src_img_dir.glob("*")))
        
        # 1. Analyze usage
        # img_file -> set(class_ids)
        # class_id -> list(img_files)
        img_to_classes = {}
        class_to_imgs = {cid: [] for cid in range(data_config['nc'])}
        
        print("Analyzing class distribution...")
        for img_path in tqdm(images):
            total_original_images += 1
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            
            current_img_classes = set()
            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            current_img_classes.add(cls_id)
            
            img_to_classes[img_path] = current_img_classes
            for cid in current_img_classes:
                class_to_imgs[cid].append(img_path)

        # 2. Determine target count from Joker
        joker_imgs = class_to_imgs.get(JOKER_CLASS_ID, [])
        target_count = len(joker_imgs)
        print(f"  Target instances per class (based on Joker): {target_count}")
        
        if target_count == 0:
             print("  WARNING: No jokers found in this split. Result will be empty for this split.")

        # 3. Select images
        kept_images = set()
        current_class_counts = {cid: 0 for cid in range(data_config['nc'])}
        
        # Always keep all jokers first
        for img in joker_imgs:
            if img not in kept_images:
                kept_images.add(img)
                for cid in img_to_classes[img]:
                    current_class_counts[cid] += 1
        
        # Fill other classes
        # Shuffle lists to ensure random selection
        for cid in class_to_imgs:
            random.shuffle(class_to_imgs[cid])
            
        # Iterate through all classes to satisfy target
        sorted_classes = sorted(list(class_to_imgs.keys()))
        
        for cid in sorted_classes:
            if cid == JOKER_CLASS_ID:
                continue
                
            needed = target_count - current_class_counts[cid]
            if needed <= 0:
                continue
            
            # Try to find images for this class
            potential_imgs = class_to_imgs[cid]
            for img in potential_imgs:
                if current_class_counts[cid] >= target_count:
                    break
                
                if img not in kept_images:
                    kept_images.add(img)
                    # Update counts for all classes in this new image
                    for c in img_to_classes[img]:
                        current_class_counts[c] += 1

        # 4. Copy files
        print(f"  Selected {len(kept_images)} images for {split}.")
        for img_path in kept_images:
            total_kept_images += 1
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

        # 5. Print Split Stats
        print(f"  Split Stats ({split}):")
        print(f"    Joker count: {current_class_counts.get(JOKER_CLASS_ID, 0)}")
        counts = list(current_class_counts.values())
        if counts:
            print(f"    Class counts - Min: {min(counts)}, Max: {max(counts)}, Avg: {sum(counts)/len(counts):.1f}")
        else:
            print("    Class counts - None")

    print("\nDataset Reduction Complete:")
    print(f"Total Images Saved: {total_kept_images}")
    print(f"Target Directory: {TARGET_DIR}")

if __name__ == "__main__":
    main()
