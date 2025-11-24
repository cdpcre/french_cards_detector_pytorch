from ultralytics import YOLO
import os

def train():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # Note: 'data.yaml' must be configured to point to your dataset
    if os.path.exists('data.yaml'):
        results = model.train(data='data.yaml', epochs=50, imgsz=640)
        print("Training completed.")
    else:
        print("Error: data.yaml not found. Please ensure your dataset is set up correctly.")

if __name__ == '__main__':
    train()
