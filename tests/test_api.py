from fastapi.testclient import TestClient
from api.main import app
import io
from PIL import Image

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid_image():
    # Create a blank 224x224 test image
    img = Image.new("RGB", (224, 224), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200

    data = response.json()
    assert "prediction"    in data
    assert "confidence"    in data
    assert "gradcam_image" in data
    assert data["prediction"] in ["real", "fake"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["gradcam_image"]) > 0   # base64 string is not empty

def test_predict_invalid_file_type():
    # Send a text file — should return 400
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400

def test_predict_missing_file():
    # Send request with no file
    response = client.post("/predict")
    assert response.status_code == 422   # FastAPI validation error

def test_predict_corrupted_image_with_valid_content_type():
    # Valid JPEG bytes truncated to corrupt the file, but the content-type
    # header still claims image/jpeg — this used to raise an uncaught OSError
    # (500) instead of a clean 400.
    img = Image.new("RGB", (224, 224), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    corrupted = buf.getvalue()[: len(buf.getvalue()) // 3]

    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", corrupted, "image/jpeg")}
    )
    assert response.status_code == 400