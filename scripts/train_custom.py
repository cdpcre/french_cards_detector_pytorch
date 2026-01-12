import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv11 with custom PyTorch loop')
    parser.add_argument('--data', type=str, default='datasets/unified/data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='models/yolo11n.pt', help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (default: 5e-4) - only used for full training')

    # Fine-tuning arguments
    parser.add_argument('--fine-tune-mode', type=str, default='progressive',
                       choices=['full', 'head-only', 'progressive'],
                       help='Fine-tuning mode: full model, head only, or progressive unfreeze')
    parser.add_argument('--head-lr', type=float, default=1e-3,
                       help='Learning rate for head layers (conservative)')
    parser.add_argument('--neck-lr', type=float, default=1e-4,
                       help='Learning rate for neck layers (conservative)')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Enable automatic mixed precision (device-aware)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--phase1-epochs', type=int, default=20,
                       help='Number of epochs for phase 1 (head-only training)')

    # Head replacement control arguments
    parser.add_argument('--force-head-replacement', action='store_true', default=False,
                       help='Force replacement of classification head even if classes match')
    parser.add_argument('--disable-head-replacement', action='store_true', default=False,
                       help='Disable automatic head replacement (not recommended)')

    # Original arguments
    parser.add_argument('--mosaic', type=float, default=0.3, help='Mosaic probability (default: 0.3)')
    parser.add_argument('--device', type=str, default='mps', help='Device (cuda/mps/cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Save directory')
    parser.add_argument('--workers', type=int, default=0, help='Num workers')
    parser.add_argument('--class-weighted-sampling', action='store_true', default=False, help='Use weighted sampling for class imbalance')
    parser.add_argument('--no-aug', action='store_true', default=False, help='Disable data augmentation')
    parser.add_argument('--cache', action='store_true', default=True, help='Enable disk caching for images')

    args = parser.parse_args()
    train(args)
