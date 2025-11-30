# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a French playing cards object detection project using YOLOv11. The system detects and classifies 53 card classes (52 standard cards + joker) in images.

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Data Pipeline
```bash
# Unify multiple dataset sources into YOLO format
python data_ingestion/unify_datasets.py

# Verify dataset integrity
python verify_ingestion.py
```

### Training

**Main Training Script**:
```bash
# Custom PyTorch training loop with full control
python train.py --epochs 50 --batch 16 --lr 5e-4 --mosaic 1.0 --device mps

# Common options:
# --model yolo11n.pt              # Model weights (default: yolo11n.pt)
# --data datasets/unified/data.yaml  # Dataset config (default: datasets/unified/data.yaml)
# --epochs 50                     # Training epochs (default: 50)
# --batch 16                      # Batch size (default: 16)
# --imgsz 640                     # Image size (default: 640)
# --lr 5e-4                       # Learning rate (default: 5e-4)
# --mosaic 1.0                    # Mosaic augmentation probability (default: 1.0)
# --device mps                    # Device: mps/cuda/cpu (default: mps)
# --project runs/train            # Save directory (default: runs/train)
# --workers 0                     # Num workers for DataLoader (default: 0)
# --class-weighted-sampling       # Enable weighted sampling for class imbalance
```

**Makefile shortcuts**:
```bash
make train       # Full 50-epoch training
make train-fast  # Quick 10-epoch training for experiments
```

**Full Pipeline** (data ingestion + verification + training):
```bash
python pipeline.py --epochs 50 --batch 16
# Use --skip-ingestion or --skip-verification to skip steps
```

### Testing
```bash
pytest tests/test_predict.py
```

### Running Inference API
```bash
# Local development
python predict.py
# API runs on http://localhost:9696

# Docker
make docker-build
make docker-run

# Test the API
curl -X POST -F "file=@image.jpg" http://localhost:9696/predict
```

## Architecture

### Training Pipeline

**`train.py` - Custom PyTorch Training Loop**
- Full PyTorch training loop with manual control over all aspects
- Custom `YOLODataset` class with explicit data loading from YAML config
- Mosaic augmentation implementation (YOLO's signature technique)
- Manual augmentation pipeline (affine transforms, color jitter)
- Bbox coordinate conversion utilities (`xywhn2xyxy`, `xyxy2xywhn`)
- Integrates Ultralytics' v8DetectionLoss for compatibility
- Best for: Full visibility, debugging, custom augmentation, research
- Optimized defaults: 50 epochs, 5e-4 learning rate, mosaic enabled

### Data Pipeline Architecture

**Multi-Source Dataset Unification** (`data_ingestion/unify_datasets.py`):
- Consolidates 5 different dataset formats into unified YOLO format
- Handles format conversions: CSV → YOLO, class remapping, path normalization
- Train/val/test split: 80/10/10
- Output: `datasets/unified/` with images/ and labels/ directories

**Supported Dataset Sources**:
1. HugoPaigneau (CSV format)
2. Andy8744 (YOLO format)
3. JayPradipShah (YOLO format, different class naming)
4. Cards.v1i.yolov11 (Roboflow YOLO)
5. Playing Cards.v2i.yolov11 (Roboflow YOLO)

**Canonical Class Naming**:
- Format: `[rank][suit_lowercase]` (e.g., "As", "10d", "Kh")
- 53 classes total: 10c, 10d, ..., Qs, joker
- Defined in `CANONICAL_CLASSES` list in `unify_datasets.py`

### YOLO Label Format

Labels are in YOLO format (one `.txt` file per image):
```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```
- Coordinates normalized to [0, 1]
- Multiple objects: one line per object
- Example: `10 0.5 0.5 0.3 0.4` (class 10 at center, 30% width, 40% height)

### Key Data Handling Patterns

**Coordinate Conversions** (in `train_custom.py`):
- `xywhn2xyxy()`: Normalized center format → absolute corner format
- `xyxy2xywhn()`: Absolute corner format → normalized center format

**Mosaic Augmentation** (YOLO's signature technique):
- Combines 4 random images into one training sample
- Controlled by `mosaic_prob` parameter (default: 0.0 in custom training)
- Includes degenerate bbox filtering (removes boxes < 2px)
- Critical for detecting small objects and improving robustness

**Path Resolution**:
- Dataset paths can be absolute or relative to YAML file
- Both `data.yaml` and custom training handle this automatically
- Important for Docker/cloud deployments where paths differ

## Dataset Configuration

**`data.yaml`**: YOLO dataset configuration
- `path`: Absolute path to unified dataset root
- `train/val/test`: Relative paths to image directories
- `nc`: 53 (number of classes)
- `names`: List of 53 class names

**Dataset Structure**:
```
datasets/unified/
├── images/
│   ├── train/    # 80% of images
│   ├── val/      # 10% of images
│   └── test/     # 10% of images
└── labels/
    ├── train/    # YOLO format labels
    ├── val/
    └── test/
```

## Model Architecture

- **Base Model**: YOLOv11 Nano (`yolo11n.pt`)
- **Output**: 53 classes (52 cards + joker)
- **Image Size**: 640×640 (default, configurable)
- **Best Weights**: Saved to `runs/train/<name>/weights/best.pt`

## Deployment

**Docker**:
- Flask API served via Gunicorn
- Port: 9696
- Endpoint: `POST /predict` (multipart form-data with `file` key)
- Response: JSON with detections (class_id, class_name, confidence, bbox)

**Cloud (Render)**:
- Dockerfile configured for Render deployment
- Set internal port to 9696
- Automatic deployment on git push

## Common Workflows

**Adding a New Dataset Source**:
1. Add dataset to `datasets/` directory
2. Extend `unify_datasets.py` with `process_<name>()` function
3. Map source classes to `CANONICAL_CLASSES`
4. Call from `main()` and append to `all_data_items`
5. Re-run data ingestion

**Debugging Training**:
1. `train.py` provides full visibility into training loop
2. Visualize augmentations in `visualize_augmentations.ipynb`
3. Verify data with `verify_ingestion.py`
4. Check class distribution in `dataset_details.csv`
5. Monitor best model saves for validation loss trends

**Hyperparameter Tuning**:
1. Use `tuning.ipynb` for experiments
2. Key params: learning rate, mosaic probability, batch size
3. Compare results across runs in `runs/tuning/`

**Model Evaluation**:
1. Train model with `train.py`
2. Open `evaluation.ipynb`
3. Load best weights from `runs/train/<name>/weights/best.pt`
4. Run evaluation on test set

## Advanced Training Features

### Weighted Sampling for Class Imbalance
Enable with `--class-weighted-sampling` flag:
- Uses inverse square root weighting to oversample minority classes
- Especially effective for joker class (severe imbalance 1:32.5)
- Balances training without excessive oversampling
- Automatically computes weights from training dataset

### Joker-Specific Augmentation
Automatically applied during training to samples containing joker cards:
- 1-2 random augmentations per joker sample
- Techniques: Gaussian blur, brightness variation, noise injection, horizontal flip
- Increases joker sample diversity
- Applied before general augmentation pipeline

### Best Model Tracking
Automatic during training:
- Saves `best.pt` when validation loss improves
- Tracks best_val_loss across all epochs
- Final summary shows best validation loss achieved
- Use `best.pt` for inference/deployment

## Important Notes

- **Device Selection**: Use `--device mps` on Mac, `cuda` on GPU, `cpu` as fallback
- **Weighted Sampling**: Highly recommended for joker class (`--class-weighted-sampling`)
- **Mosaic Augmentation**: Enabled by default (mosaic=1.0), critical for YOLO performance
- **Degenerate Bboxes**: Automatically filtered in mosaic augmentation (< 2px width/height)
- **Path Resolution**: Both absolute and relative paths supported in YAML configs
- **Best Model**: Always use `runs/train/best.pt` for deployment, not last epoch

## Notebooks

- `eda.ipynb`: Dataset statistics, class distribution, augmentation visualization
- `evaluation.ipynb`: Test set metrics, confusion matrix, prediction visualization
- `kaggle_training.ipynb`: Kaggle-specific training workflow
- `tuning.ipynb`: Hyperparameter experiments and comparison
- `visualize_augmentations.ipynb`: Augmentation pipeline demonstration
- `yolo_training_pipeline.ipynb`: End-to-end training demonstration
