import argparse
from ultralytics import YOLO
import os

def train(epochs=50, imgsz=640):
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # Note: 'data.yaml' must be configured to point to your dataset
    if os.path.exists('data.yaml'):
        results = model.train(data='data.yaml', epochs=epochs, imgsz=imgsz)
        print("Training completed.")
    else:
        print("Error: data.yaml not found. Please ensure your dataset is set up correctly.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    args = parser.parse_args()
    
    train(epochs=args.epochs, imgsz=args.imgsz)
