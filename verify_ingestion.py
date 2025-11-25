import glob
import os
import csv
from tqdm import tqdm
from pathlib import Path

def verify_dataset():
    label_dir = "datasets/unified/labels"
    csv_output = "dataset_details.csv"
    
    print("Verifying dataset integrity and generating CSV...")
    
    # Get all label files
    label_files = glob.glob(os.path.join(label_dir, "**", "*.txt"), recursive=True)
    
    print(f"Total label files found: {len(label_files)}")
    
    # Prepare CSV
    headers = ['image_path', 'label_path', 'class_id', 'bbox_values']
    
    rows = []
    invalid_files = []
    all_classes = set()
    
    for lf in tqdm(label_files, desc="Processing files"):
        lf_path = Path(lf)
        # Infer image path: replace 'labels' with 'images' and extension with .jpg (or check others)
        # Structure is datasets/unified/labels/split/filename.txt
        # Image should be datasets/unified/images/split/filename.jpg
        
        parts = list(lf_path.parts)
        try:
            # Find 'labels' in path and replace with 'images'
            # We look from the right to avoid issues if 'labels' appears earlier in the path
            # But here we know the structure is datasets/unified/labels/...
            idx = len(parts) - 1 - parts[::-1].index('labels')
            parts[idx] = 'images'
        except ValueError:
            print(f"Skipping file with unexpected path structure: {lf}")
            continue
            
        img_path_base = Path(*parts).with_suffix('')
        
        # Check for image existence with common extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            probe = Path(str(img_path_base) + ext)
            if probe.exists():
                image_path = str(probe.absolute())
                break
        
        if not image_path:
            image_path = "IMAGE_NOT_FOUND"

        with open(lf, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                try:
                    cls_id = int(parts[0])
                    bbox = " ".join(parts[1:])
                    
                    all_classes.add(cls_id)
                    if cls_id < 0 or cls_id > 51:
                        invalid_files.append((lf, cls_id))
                    
                    rows.append({
                        'image_path': image_path,
                        'label_path': str(lf_path.absolute()),
                        'class_id': cls_id,
                        'bbox_values': bbox
                    })
                except ValueError:
                    print(f"Error parsing line in {lf}: {line}")

    # Write CSV
    print(f"Writing {len(rows)} rows to {csv_output}...")
    with open(csv_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    if invalid_files:
        print(f"FOUND {len(invalid_files)} FILES WITH INVALID CLASS IDs!")
        for f, cid in invalid_files[:5]:
            print(f"  {f}: {cid}")
    else:
        print("All class IDs are within valid range (0-51).")
        
    print(f"Unique classes found: {sorted(list(all_classes))}")
    print(f"CSV saved to {os.path.abspath(csv_output)}")
    print("Verification complete.")

if __name__ == "__main__":
    verify_dataset()
