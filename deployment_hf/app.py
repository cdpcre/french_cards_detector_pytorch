import os
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Set Ultralytics config dir to /tmp to avoid permission issues on HF Spaces
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

# Get the absolute path of the current directory (deployment_hf)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to resources
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
EXAMPLES = [
    os.path.join(BASE_DIR, "raw_example_1.jpg"),
    os.path.join(BASE_DIR, "raw_example_2.jpg"),
    os.path.join(BASE_DIR, "raw_example_3.jpg")
]

# Load model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    # Fallback to trying default locations just in case
    try:
        model = YOLO("best.pt")
    except:
        print("Error: best.pt not found.")
        raise

def predict(image, conf_threshold, iou_threshold):
    if image is None:
        return None
    
    # Run inference
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold
    )
    
    # Plot results
    # results[0].plot() returns a BGR numpy array
    res_plotted = results[0].plot()
    
    # Convert BGR to RGB for Gradio
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(res_rgb)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🃏 French Cards Detector")
    gr.Markdown("Upload an image of playing cards to detect their rank and suit.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            conf_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold")
            iou_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.45, label="IoU Threshold")
            predict_btn = gr.Button("Detect Cards")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Detections")
    
    predict_btn.click(
        fn=predict,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=output_image
    )
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch()
