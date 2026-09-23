import io
import re

from PIL import Image

LOW_CONFIDENCE  = 0.70
HIGH_CONFIDENCE = 0.90
MAX_FIELD_LEN   = 80

EXIF_MAKE, EXIF_MODEL, EXIF_SOFTWARE, EXIF_DATETIME = 271, 272, 305, 306


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def _clean(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = "".join(ch for ch in str(value) if ch.isprintable()).strip()
    return text[:MAX_FIELD_LEN] or None


def _format_exif_date(text):
    match = re.fullmatch(r"(\d{4}):(\d{2}):(\d{2}) (\d{2}:\d{2}:\d{2})", text or "")
    return f"{match[1]}-{match[2]}-{match[3]} {match[4]}" if match else text


def image_metadata(data: bytes) -> dict:
    meta = {
        "readable": True, "size_bytes": len(data), "size_label": human_size(len(data)),
        "width": None, "height": None, "format": None,
        "has_exif": False, "camera": None, "software": None, "modified": None,
    }
    try:
        with Image.open(io.BytesIO(data)) as img:
            meta["width"], meta["height"] = img.size
            meta["format"] = img.format
            exif = img.getexif()
            meta["has_exif"] = len(exif) > 0
            make, model = _clean(exif.get(EXIF_MAKE)), _clean(exif.get(EXIF_MODEL))
            if make and model and not model.lower().startswith(make.lower()):
                meta["camera"] = f"{make} {model}"
            else:
                meta["camera"] = model or make
            meta["software"] = _clean(exif.get(EXIF_SOFTWARE))
            meta["modified"] = _format_exif_date(_clean(exif.get(EXIF_DATETIME)))
    except Exception:
        # Metadata is informational only; a parse failure must never block the verdict.
        meta["readable"] = False
    return meta


VERDICT_TEXT = {
    "synthetic": "LIKELY SYNTHETIC",
    "authentic": "LIKELY AUTHENTIC",
    "inconclusive": "INCONCLUSIVE",
}
LEAN_TEXT = {"fake": "synthetic", "real": "authentic"}

MODEL_SCOPE_STATUS = "StyleGAN + Stable Diffusion faces ·"
MODEL_SCOPE_NOTE = (
    "Trained on StyleGAN- and Stable Diffusion-generated faces. "
    "Does not reliably detect face-swap manipulation. "
)

def verdict_state(label: str, confidence: float) -> str:
    """Map the API's real/fake + confidence onto synthetic / authentic / inconclusive."""
    if label not in ("real", "fake"):
        raise ValueError(f"unexpected label: {label!r}")
    if confidence < LOW_CONFIDENCE:
        return "inconclusive"
    return "synthetic" if label == "fake" else "authentic"


def confidence_band(confidence: float) -> str:
    if confidence < LOW_CONFIDENCE:
        return "low"
    return "high" if confidence >= HIGH_CONFIDENCE else "moderate"


def verdict_reasoning(label: str, confidence: float):
    """Return (headline, detail) in plain language. Never claims proof either way."""
    band = confidence_band(confidence)
    if band == "low":
        return (
            "Inconclusive — manual review recommended.",
            f"Confidence of {confidence * 100:.1f}% is below the {LOW_CONFIDENCE * 100:.0f}% threshold.",
        )
    verb = "Flagged as synthetic" if label == "fake" else "Classified as authentic"
    proof = "manipulation" if label == "fake" else "authenticity"
    return (
        f"{verb} ({band} confidence).",
        "The highlighted regions influenced this score most. "
        f"Heatmaps show model attention, not proof of {proof}.",
    )


def blend_overlay(original: Image.Image, overlay: Image.Image, strength: float) -> Image.Image:
    """strength 0 -> original, 1 -> the API's Grad-CAM overlay (both at the model's input size)."""
    base = original.convert("RGB").resize(overlay.size)
    return Image.blend(base, overlay.convert("RGB"), max(0.0, min(1.0, strength)))


def metadata_finding(meta: dict) -> str:
    """One-line, neutral description of what the file's EXIF holds. Says nothing about authenticity."""
    if not meta.get("readable"):
        return "Could not be read"
    if not meta.get("has_exif"):
        return "No EXIF present"
    camera, software = meta.get("camera"), meta.get("software")
    if camera and software:
        return "EXIF present · camera + software tags recorded"
    if camera:
        return "EXIF present · camera tag recorded"
    if software:
        return "EXIF present · software tag recorded"
    return "EXIF present · no camera or software tags"


# ---------------------------------------------------------------------------------------------
# Attention region. The API returns only pytorch-grad-cam's blended overlay:
#   overlay = normaliser * (0.5 * image + 0.5 * JET(mask))        (image_weight=0.5)
# so the raw mask can be recovered from overlay + original without an API change.
# ---------------------------------------------------------------------------------------------
HEAT_THRESHOLD = 0.5
ATTENTION_STRIDE = 4            # decode every 4th pixel: ample for a 3x3 grid + a coverage percentage
ATTENTION_MAX_RESIDUAL = 0.05   # recoveries on real overlays measured <= 0.009; garbage is far above
_GRID_ROWS = ("upper", "centre", "lower")
_GRID_COLS = ("left", "centre", "right")


def _jet_lut():
    import cv2
    import numpy as np
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET)
    return lut[:, 0, ::-1].astype(np.float32) / 255.0


def attention_summary(original: Image.Image, overlay: Image.Image):
    """Return {"peak": "centre-left", "coverage_pct": 14.0} or None if the overlay can't be decoded."""
    try:
        import numpy as np
        lut = _jet_lut()
    except ImportError:
        return None

    ov = np.asarray(overlay.convert("RGB"), dtype=np.float32) / 255.0
    full_h, full_w = ov.shape[:2]
    img = np.asarray(original.convert("RGB").resize((full_w, full_h)), dtype=np.float32) / 255.0
    ov, img = ov[::ATTENTION_STRIDE, ::ATTENTION_STRIDE], img[::ATTENTION_STRIDE, ::ATTENTION_STRIDE]
    height, width = ov.shape[:2]
    flat_ov, flat_img = ov.reshape(-1, 3), img.reshape(-1, 3)

    lut_rows = np.ascontiguousarray(lut.T)

    def nearest(heat):
        d = np.zeros((len(heat), lut_rows.shape[1]), np.float32)
        for c in range(3):
            diff = heat[:, c:c + 1] - lut_rows[c][None, :]
            d += diff * diff
        idx = d.argmin(1)
        return idx, d[np.arange(len(heat)), idx]

    def heat_for(scale, rows):
        return np.clip((flat_ov[rows] * np.float32(scale) - 0.5 * flat_img[rows]) / 0.5, 0.0, 1.0)

    probe = np.arange(0, len(flat_ov), max(1, len(flat_ov) // 200))

    def cost(scale):
        return nearest(heat_for(scale, probe))[1].sum()

    coarse = min(np.linspace(0.5, 1.0, 11), key=cost)
    scale = min(np.linspace(max(0.5, coarse - 0.05), min(1.0, coarse + 0.05), 11), key=cost)
    idx, dist = nearest(heat_for(scale, slice(None)))
    if float(np.sqrt(dist.mean())) > ATTENTION_MAX_RESIDUAL:
        return None

    mask = (idx / 255.0).reshape(height, width)
    if mask.min() > 0.15 or mask.max() < 0.85:
        # Grad-CAM masks are min-max normalised, so a genuine one spans ~0..1 (real outputs: min 0.0, max >= 0.98).
        return None
    high = mask >= HEAT_THRESHOLD
    if not high.any():
        return None
    ys, xs = np.nonzero(high)
    weights = mask[high]
    cx, cy = float((xs * weights).sum() / weights.sum()), float((ys * weights).sum() / weights.sum())
    row = _GRID_ROWS[min(2, int(cy / height * 3))]
    col = _GRID_COLS[min(2, int(cx / width * 3))]
    return {"peak": "centre" if row == col == "centre" else f"{row}-{col}",
            "coverage_pct": round(float(high.mean()) * 100, 1)}


def attention_status(summary, available: bool = True) -> str:
    if not available:
        return "Not available"
    if not summary:
        return "Not estimated"
    return f"Peak: {summary['peak']} · {summary['coverage_pct']:.0f}% of image above 50% intensity"
