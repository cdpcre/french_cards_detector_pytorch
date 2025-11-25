import pytest
import sys
import os
from unittest.mock import MagicMock, patch
import numpy as np

# Add the parent directory to sys.path to import predict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock ultralytics before importing predict
sys.modules['ultralytics'] = MagicMock()

from predict import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('predict.model')
def test_predict_no_file(mock_model, client):
    response = client.post('/predict')
    assert response.status_code == 400
    assert b'No file part' in response.data

@patch('predict.model')
def test_predict_valid_image(mock_model, client):
    # Mock the model's return value
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = 10
    mock_box.conf = 0.95
    mock_box.xyxy.tolist.return_value = [[100.0, 150.0, 200.0, 300.0]]
    
    mock_result.boxes = [mock_box]
    mock_model.return_value = [mock_result]
    mock_model.names = {10: 'Jack of Spades'}

    # Create a dummy image
    import io
    data = {
        'file': (io.BytesIO(b'fake image data'), 'test.jpg')
    }
    
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'detections' in json_data
    assert len(json_data['detections']) == 1
    assert json_data['detections'][0]['class_name'] == 'Jack of Spades'
