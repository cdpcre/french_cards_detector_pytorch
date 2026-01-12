"""
Unit tests for the predict module.

These tests use mocking to avoid loading the real YOLO model,
ensuring fast test execution and isolation.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def client():
    """Create a test client for the Flask app with mocked YOLO model."""
    # Temporarily mock ultralytics for this test module only
    mock_ultralytics = MagicMock()
    mock_model = MagicMock()
    mock_ultralytics.YOLO.return_value = mock_model
    
    # Store original module if it exists
    original_ultralytics = sys.modules.get('ultralytics')
    sys.modules['ultralytics'] = mock_ultralytics
    
    try:
        # Need to reload predict module with mocked ultralytics
        import importlib
        if 'predict' in sys.modules:
            # Clear cached module to force reimport with mock
            del sys.modules['predict']
        
        import predict
        predict.model = mock_model
        
        predict.app.config['TESTING'] = True
        with predict.app.test_client() as test_client:
            yield test_client
    finally:
        # Restore original ultralytics module
        if original_ultralytics is not None:
            sys.modules['ultralytics'] = original_ultralytics
        elif 'ultralytics' in sys.modules:
            del sys.modules['ultralytics']
        
        # Clear predict module from cache
        if 'predict' in sys.modules:
            del sys.modules['predict']


def test_predict_no_file(client):
    """Test that endpoint returns 400 when no file is provided."""
    with patch('predict.model'):
        response = client.post('/predict')
        assert response.status_code == 400
        assert b'No file part' in response.data


def test_predict_valid_image(client):
    """Test prediction with a valid image file."""
    with patch('predict.cv2.imdecode') as mock_imdecode, \
         patch('predict.model') as mock_model:
        
        # Mock cv2.imdecode to return a valid image
        mock_imdecode.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Mock the model's return value
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = 10
        mock_box.conf = 0.95
        mock_box.xyxy.tolist.return_value = [[100.0, 150.0, 200.0, 300.0]]
        
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        mock_model.names = {10: 'Jack of Spades'}

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
