import os
import io
import base64
import requests
from PIL import Image
import streamlit as st

API_URL = os.getenv("API_URL",  "http://api:8000/predict")

st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Deepfake Face Detector")
st.caption(
    "Upload a face image. The model will classify it as real or fake "
    "and show a Grad-CAM heatmap of the regions it focused on."
)

uploaded = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input image")
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)

    with st.spinner("Running prediction..."):
        try:
            response = requests.post(
                API_URL,
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=30
            )
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure uvicorn is running on port 8000.")
            st.stop()

    if response.status_code == 200:
        result      = response.json()
        label       = result["prediction"]
        conf        = result["confidence"]
        gradcam_b64 = result.get("gradcam_image", "")

        with col2:
            st.subheader("Grad-CAM heatmap")
            if gradcam_b64:
                gradcam_bytes = base64.b64decode(gradcam_b64)
                gradcam_img   = Image.open(io.BytesIO(gradcam_bytes))
                st.image(gradcam_img, use_column_width=True)
            else:
                st.info("Grad-CAM not available.")

        st.divider()

        if label == "fake":
            st.error(f"Prediction: FAKE — Confidence: {conf * 100:.1f}%")
        else:
            st.success(f"Prediction: REAL — Confidence: {conf * 100:.1f}%")

        st.caption(
            "Grad-CAM highlights the facial regions that most influenced the prediction. "
            "Warmer colors (red/yellow) = higher activation."
        )

    elif response.status_code == 400:
        st.error(f"Bad request: {response.json().get('detail', 'Unknown error')}")
    else:
        st.error(f"API error {response.status_code}.")