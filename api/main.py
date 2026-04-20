import io
import base64
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from src.model import build_model
from src.gradcam import get_gradcam_image

app = FastAPI(title="Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once at startup — not on every request
model = build_model(pretrained=False)
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval().to(DEVICE)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image."
        )

    try:
        contents = await file.read()
        img      = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()

    # ImageFolder assigns labels alphabetically: fake=0, real=1
    # sigmoid output = P(real) because model was trained with real=1 as positive
    # High prob → real, Low prob → fake
    is_fake    = prob < 0.5
    label      = "fake" if is_fake else "real"
    confidence = round((1 - prob) if is_fake else prob, 4)

    # Grad-CAM — highlights regions that activated for fake class
    gradcam_img = get_gradcam_image(model, img, DEVICE)
    buf         = io.BytesIO()
    gradcam_img.save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "prediction":    label,
        "confidence":    confidence,
        "gradcam_image": gradcam_b64
    }