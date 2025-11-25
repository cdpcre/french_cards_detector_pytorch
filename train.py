import argparse
import os
from ultralytics import YOLO

def train(args):
    # Ensure project directory exists
    os.makedirs(args.project, exist_ok=True)

    # Load a model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True, # Overwrite existing run if name is same
        pretrained=True,
        resume=args.resume,
        freeze=args.freeze,
        multi_scale=args.multi_scale
    )
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
    parser = argparse.ArgumentParser(description='Train/Finetune YOLOv11 model locally')
    
    # Model and Data
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to model weights (default: yolo11n.pt)')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data config (default: data.yaml)')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size (default: 16, use -1 for auto)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='mps', help='Device to run on, i.e. 0, 0,1,2,3 or cpu')
    parser.add_argument('--freeze', type=int, default=None, help='Number of layers to freeze')
    parser.add_argument('--multi_scale', type=bool, default=True, help='Use multi scale training')    
    
    # Run Configuration
    parser.add_argument('--project', type=str, default='runs/train', help='Directory to save results')
    parser.add_argument('--name', type=str, default='exp', help='Name of the run')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    
    # Export
    parser.add_argument('--export', type=str, default="onnx", help='Export format (e.g., onnx, torchscript) after training')

    args = parser.parse_args()
    
    # Handle auto batch size if user passes -1 (argparse parses as int)
    if args.batch == -1:
        args.batch = -1 # ultralytics treats -1 as auto

    train(args)
