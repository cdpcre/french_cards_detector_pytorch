---
title: French Cards Detector
emoji: 🃏
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
---

# French Cards Detector

<p align="center">
  <img src="https://github.com/cdpcre/french_cards_detector_pytorch/raw/main/assets/inference_example_1.jpg" width="30%" />
  <img src="https://github.com/cdpcre/french_cards_detector_pytorch/raw/main/assets/inference_example_2.jpg" width="30%" />
  <img src="https://github.com/cdpcre/french_cards_detector_pytorch/raw/main/assets/inference_example_3.jpg" width="30%" />
</p>

## Problem Description
This project detects and classifies French playing cards in images using a fine-tuned YOLOv11 model. It identifies both rank and suit (e.g., "Ace of Spades", "10 of Hearts") with high precision.

### 🚀 Performance
- **mAP@50**: 93.7%
- **Precision**: 95.5%
- **Recall**: 88.7%

## Deployment
This Space hosts the inference demo.


This folder contains the necessary code to deploy your model to a Hugging Face Space.

## Steps to Deploy

1. **Create a Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces).
   - Click "Create new Space".
   - Name your space (e.g., `french-cards-detector`).
   - Select **Gradio** as the SDK.
   - Choose "Public" or "Private".
   - Click "Create Space".

2. **Upload Files**:
   You can upload files directly via the web interface or clone the repository locally.

   **Files to upload:**
   - `app.py` (from this folder)
   - `requirements.txt` (from this folder)
   - `best.pt` (COPY this from your local `models/best.pt`!)

   > **Important**: You MUST upload the trained model weights `best.pt` to the root of your Space strictly alongside `app.py`.

3. **Wait for Build**:
   - Validating the requirements installation.
   - The app should start automatically.

## Local Test
To test this specific deployment code locally:

```bash
cd deployment_hf
# Copy model here for testing
cp ../models/best.pt .
# Install requirements
pip install -r requirements.txt
# Run app
python app.py
```
Open the link http://127.0.0.1:7860 in your browser.
