from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model
# Priority: Environment Variable -> proper location in models/ -> root -> default
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/best.pt')

if not os.path.exists(MODEL_PATH):
    # Fallback to checking root if not found in models/
    if os.path.exists('best.pt'):
        MODEL_PATH = 'best.pt'
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Trying to download or use default yolov8n.pt for demo.")
        MODEL_PATH = 'yolov8n.pt'  # Ultralytics will download this if missing

logger.info(f"Loading model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Failed to decode image. Ensure it is a valid image file.'}), 400

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

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
