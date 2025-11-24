from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import cv2
import numpy as np

app = Flask(__name__)

# Load the model (ensure best.pt is present after training)
# For initial testing, we can use the pre-trained yolov8n.pt if best.pt doesn't exist
model_path = 'runs/detect/train/weights/best.pt'
if not os.path.exists(model_path):
    print(f"Warning: {model_path} not found. Using yolov8n.pt for demonstration.")
    model_path = 'yolov8n.pt'

model = YOLO(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run inference
    results = model(img)
    
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'class': int(box.cls),
                'class_name': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy.tolist()[0]
            })

    return jsonify({'detections': detections})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
