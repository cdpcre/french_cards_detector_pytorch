# French Cards Detector - Object Detection Project

## Problem Description
This project addresses the challenge of detecting and classifying playing cards in images. Leveraging computer vision and deep learning techniques—specifically the YOLOv8 object detection model—we aim to identify both the rank and suit of standard French playing cards (e.g., "Ace of Spades", "10 of Hearts") within a given image.

Potential applications include:
- **Automated Card Game Analysis**: Tracking game states in real-time.
- **Casino Security**: Monitoring tables for fair play.
- **Augmented Reality**: Enhancing physical card games with digital overlays.

## Dataset
We utilize a unified dataset composed of multiple sources:
- **Andy8744**: [Playing Cards Object Detection Dataset](https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset)
- **HugoPaigneau**: [Playing Cards Dataset](https://www.kaggle.com/datasets/hugopaigneau/playing-cards-dataset)
- **JayPradipShah**: [The Complete Playing Card Dataset](https://www.kaggle.com/datasets/jaypradipshah/the-complete-playing-card-dataset)

- **Classes**: 52 standard cards (Ace through King, across Spades, Hearts, Diamonds, and Clubs) + Joker.
- **Format**: The dataset is in YOLO format (images and corresponding `.txt` label files).

## Project Structure
- `notebook.ipynb`: Jupyter notebook for Exploratory Data Analysis (EDA) and initial model training experiments.
- `train.py`: A standalone Python script exported from the notebook to ensure reproducible training.
- `predict.py`: A Flask web application that serves the trained model via a REST API.
- `Dockerfile`: Configuration file for containerizing the prediction service.
- `requirements.txt`: List of all Python dependencies required for the project.
- `Makefile`: A collection of helper commands for setup, training, and execution.

## Setup and Installation

### Prerequisites
- **Python 3.9+**
- **Docker** (optional, recommended for containerization)

### Local Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd french_cards_detector_pytorch
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model
To train the model locally, execute the following command:
```bash
python train.py
```
*Note: This script will download the pre-trained `yolov8n.pt` model and fine-tune it on your dataset. Ensure your `data.yaml` is correctly configured.*

### Running the Prediction Service
To start the Flask API locally:
```bash
python predict.py
```
The service will be accessible at `http://localhost:9696`.

## Containerization
To build and run the application using Docker:

1. **Build the image**:
   ```bash
   docker build -t french-cards-detector .
   ```

2. **Run the container**:
   ```bash
   docker run -it --rm -p 9696:9696 french-cards-detector
   ```

## Cloud Deployment (Render)
This project is optimized for deployment on **Render**.

1. Create a new **Web Service** on the Render dashboard.
2. Connect your GitHub repository.
3. Select **Docker** as the runtime environment.
4. Set the internal port to `9696` in the settings.
5. Click **Deploy**!

## API Usage
You can interact with the API using `curl` or any HTTP client.

**Endpoint**: `/predict`
**Method**: `POST`
**Body**: Multipart form-data containing an image file with the key `file`.

**Example Request**:
```bash
curl -X POST -F "file=@/path/to/card_image.jpg" http://localhost:9696/predict
```

**Example Response**:
```json
{
  "detections": [
    {
      "bbox": [100.0, 150.0, 200.0, 300.0],
      "class": 10,
      "class_name": "Jack of Spades",
      "confidence": 0.95
    }
  ]
}
```
