
import os
from ultralytics import YOLO

# Path to the best model weights
model_path = 'runs/train/run_top_perf/weights/best.pt'

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Did you run the pipeline?")
else:
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded successfully.")
    
    # Validate on Test Set
    print("\nStarting evaluation on test set...")
    metrics = model.val(split='test')
    
    print("\nEvaluation Results:")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
