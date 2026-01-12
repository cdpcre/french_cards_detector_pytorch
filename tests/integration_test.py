import requests
import os
import time
import pytest

# URL of the prediction service
URL = os.environ.get('PREDICT_SERVICE_URL', 'http://localhost:9696/predict')
SAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), 'sample.jpg')

def test_prediction_endpoint():
    if not os.path.exists(SAMPLE_IMAGE):
        pytest.skip(f"Sample image not found at {SAMPLE_IMAGE}")

    print(f"Testing endpoint: {URL}")
    print(f"Using image: {SAMPLE_IMAGE}")

    with open(SAMPLE_IMAGE, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post(URL, files=files, timeout=5)
        except requests.exceptions.ConnectionError:
            pytest.fail(f"Could not connect to {URL}. Is the service running?")

    assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"
    
    data = response.json()
    assert 'detections' in data, "Response JSON missing 'detections' key"
    assert isinstance(data['detections'], list), "'detections' should be a list"
    
    # If detections are found, verify structure
    if len(data['detections']) > 0:
        det = data['detections'][0]
        assert 'class' in det
        assert 'class_name' in det
        assert 'confidence' in det
        assert 'bbox' in det
        print(f"Detections found: {len(data['detections'])}")
    else:
        print("No detections found (this might be expected depending on the image and model)")

if __name__ == "__main__":
    # Simple manual run
    try:
        test_prediction_endpoint()
        print("Integration test passed!")
    except Exception as e:
        print(f"Integration test failed: {e}")
        exit(1)
