import base64
import binascii
import hashlib
import io
import os
import time

import requests
import streamlit as st
from PIL import Image

import components as ui
import metadata as md
import theme

API_URL = os.getenv("API_URL", "http://api:8000/predict")

st.set_page_config(page_title="Deepfake Detector", layout="wide")

ss = st.session_state
for key, default in [("entry", None), ("pending", None), ("last_upload_hash", None),
                     ("last_request", None), ("error", None)]:
    ss.setdefault(key, default)


def _detail(response):
    try:
        return response.json().get("detail", "Unknown error")
    except (ValueError, AttributeError):
        return "Unknown error"


def call_api(name, data, content_type):
    """POST the image to the API. Returns (result, error_message); exactly one is None."""
    try:
        started  = time.perf_counter()
        response = requests.post(
            API_URL, files={"file": (name, data, content_type)}, timeout=30
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. Make sure uvicorn is running on port 8000."
    except requests.exceptions.ReadTimeout:
        return None, "The API took too long to respond. Please try again."
    except requests.exceptions.RequestException:
        return None, "The request to the API failed. Please try again."

    if response.status_code == 200:
        try:
            body       = response.json()
            label      = body["prediction"]
            confidence = float(body["confidence"])
        except (ValueError, KeyError, TypeError):
            return None, "The API returned an unexpected response."
        if label not in ("real", "fake") or not 0.0 <= confidence <= 1.0:
            return None, "The API returned an unexpected response."
        return {"label": label, "confidence": confidence, "latency_ms": latency_ms,
                "gradcam_image": body.get("gradcam_image", "")}, None
    if response.status_code == 400:
        return None, f"Bad request: {_detail(response)}"
    if response.status_code == 413:
        return None, _detail(response)
    return None, f"API error {response.status_code}."


def build_entry(pending, result):
    data  = pending["data"]
    # No exif_transpose: the API sees the file as-is, so the preview must match its overlay.
    image = Image.open(io.BytesIO(data)).convert("RGB")
    preview = image.copy()
    preview.thumbnail((512, 512))
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=90)

    overlay = None
    if result["gradcam_image"]:
        try:
            overlay = base64.b64decode(result["gradcam_image"], validate=True)
            Image.open(io.BytesIO(overlay)).load()
        except (binascii.Error, ValueError, OSError):
            overlay = None

    attention = None
    if overlay:
        try:
            attention = md.attention_summary(preview, Image.open(io.BytesIO(overlay)))
        except Exception:
            # Supplementary readout; failing to decode the heatmap must never block the verdict.
            attention = None

    meta = md.image_metadata(data)
    return {
        "name": pending["name"],
        "label": result["label"], "confidence": result["confidence"],
        "latency_ms": result["latency_ms"], "preview_jpeg": buf.getvalue(),
        "overlay_png": overlay, "meta": meta,
        "attention_text": md.attention_status(attention, available=overlay is not None),
        "metadata_text": md.metadata_finding(meta),
    }


def retry():
    ss.pending = ss.last_request


st.markdown(theme.get_css(), unsafe_allow_html=True)
st.markdown(ui.brand(), unsafe_allow_html=True)
st.markdown(ui.page_header(), unsafe_allow_html=True)

with st.container():
    st.markdown(ui.eyebrow("Upload an image", card=True), unsafe_allow_html=True)
    uploaded = st.file_uploader("Face image", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    st.markdown(ui.trust_strip(), unsafe_allow_html=True)
    hint_slot = st.empty()

if uploaded is None:
    if ss.last_upload_hash is not None:      # the user removed the file: the result goes with it
        ss.entry = None
        ss.error = None
    ss.last_upload_hash = None
else:
    upload_bytes = uploaded.getvalue()
    upload_hash  = hashlib.sha256(upload_bytes).hexdigest()
    if upload_hash != ss.last_upload_hash:
        ss.last_upload_hash = upload_hash
        ss.pending = {"name": uploaded.name, "data": upload_bytes, "type": uploaded.type}

pending, ss.pending = ss.pending, None
if pending:
    ss.error        = None
    ss.entry        = None
    ss.last_request = pending
    with st.spinner("Analyzing image…"):
        result, error = call_api(pending["name"], pending["data"], pending["type"])
    if error:
        ss.error = error
    else:
        ss.entry = build_entry(pending, result)

if ss.entry is None and not ss.error:
    hint_slot.markdown(ui.hint("Results appear below once the image has been analyzed."),
                       unsafe_allow_html=True)

if ss.error:
    st.error(ss.error)
    st.button("Retry", on_click=retry)
    st.stop()

entry = ss.entry
if entry is not None:
    with st.container():
        st.markdown(ui.report_head(entry["name"], entry["meta"]), unsafe_allow_html=True)
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.markdown(ui.verdict_block(entry["label"], entry["confidence"]),
                        unsafe_allow_html=True)
            st.markdown(ui.signals_block(entry["attention_text"], entry["metadata_text"]),
                        unsafe_allow_html=True)
        with right:
            st.markdown(ui.eyebrow("Grad-CAM overlay"), unsafe_allow_html=True)
            image_slot = st.container()
            original   = Image.open(io.BytesIO(entry["preview_jpeg"]))
            if entry["overlay_png"]:
                strength = st.slider("Heatmap overlay strength (0% = original image)",
                                     0, 100, 100, format="%d%%", key="overlay_strength")
                shown = md.blend_overlay(original, Image.open(io.BytesIO(entry["overlay_png"])),
                                         strength / 100)
                note = ("Grad-CAM at the model's 224 × 224 input size. "
                        "Warmer colours mark regions that influenced the score most.")
            else:
                shown = original.convert("RGB").resize((224, 224))
                note  = "Grad-CAM not available for this analysis. Showing the model's 224 × 224 input."
            with image_slot:
                st.markdown(ui.image_html(ui.to_data_uri(shown, "JPEG", quality=92),
                                          "Analyzed image with Grad-CAM heatmap"),
                            unsafe_allow_html=True)
                st.markdown(ui.caption(note), unsafe_allow_html=True)
        st.markdown(ui.footer_strip(entry["latency_ms"]), unsafe_allow_html=True)
    st.markdown(ui.metadata_card(entry["meta"]), unsafe_allow_html=True)
