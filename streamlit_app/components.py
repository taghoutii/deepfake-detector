import base64
import io
from html import escape

from PIL import Image

import metadata as md

SEGMENTS = 20


def _html(s: str) -> str:
    # st.markdown treats indented / blank-line-separated HTML as code, so emit one compact line.
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def to_data_uri(img: Image.Image, fmt: str = "PNG", max_side=None, quality: int = 88) -> str:
    img = img.convert("RGB")
    if max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def brand() -> str:
    return _html("""
        <div class="dfd-brand"><span class="dfd-brand-mark"></span>
        <span class="dfd-brand-name">Deepfake Detector</span></div>""")


def page_header() -> str:
    return _html("""
        <p class="dfd-tag">Image analysis · EfficientNet-B0</p>
        <h1 class="dfd-title">Face image authenticity analysis</h1>
        <p class="dfd-sub">Upload a face image to check whether it is likely authentic or synthetic,
        with a visual explanation of what the model focused on.</p>""")


def trust_strip() -> str:
    items = ["Self-hosted model", "Not stored", "No third-party requests", "Images only · JPG PNG"]
    return '<div class="dfd-trust">' + "".join(
        f'<span class="dfd-trust-item">{escape(i)}</span>' for i in items) + "</div>"


def card(title: str, body: str, right: str = "") -> str:
    return _html(f"""
        <div class="dfd-card"><div class="dfd-card-title"><span>{escape(title)}</span>{right}</div>{body}</div>""")


def _check_label(label: str) -> str:
    if label not in ("real", "fake"):
        raise ValueError(f"unexpected label: {label!r}")
    return label


def report_head(name: str, meta: dict) -> str:
    """First element of the verdict card; the dfd-anchor class makes its container render as the card."""
    detail = ""
    if meta.get("readable") and meta.get("width"):
        detail = f" · {meta['width']}×{meta['height']} {meta.get('format') or ''}".rstrip()
    return _html(f"""
        <div class="dfd-report-head dfd-anchor"><span class="dfd-report-title">Analysis report</span>
        <span class="dfd-report-file">{escape(name)}{escape(detail)}</span></div>""")


def verdict_block(label: str, confidence: float) -> str:
    label = _check_label(label)
    state = md.verdict_state(label, confidence)
    pct = max(0.0, min(1.0, confidence)) * 100
    headline, detail = md.verdict_reasoning(label, confidence)
    lean = (f'<div class="dfd-verdict-lean">Leans {md.LEAN_TEXT[label]}</div>'
            if state == "inconclusive" else "")
    filled = min(SEGMENTS, int(pct / (100 / SEGMENTS) + 0.5))
    segments = "".join(
        f'<i class="dfd-seg-i{" dfd-seg-i--on" if i < filled else ""}"></i>' for i in range(SEGMENTS))
    mark = md.LOW_CONFIDENCE * 100
    return _html(f"""
        <div class="dfd-verdict dfd-verdict--{state}">
        <div class="dfd-verdict-head"><div class="dfd-verdict-label">{md.VERDICT_TEXT[state]}</div>{lean}</div>
        <div class="dfd-conf"><span class="dfd-conf-label">Confidence</span>
        <span class="dfd-conf-num">{pct:.1f}%</span></div>
        <div class="dfd-seg" role="progressbar" aria-label="Model confidence"
             aria-valuemin="0" aria-valuemax="100" aria-valuenow="{pct:.1f}">{segments}
        <i class="dfd-seg-mark" style="left:{mark:.0f}%"></i></div>
        <div class="dfd-seg-scale"><span>0%</span>
        <span style="left:{mark:.0f}%;transform:translateX(-50%)">{mark:.0f}% review threshold</span>
        <span>100%</span></div>
        <div class="dfd-reason"><p class="dfd-reason-head">{escape(headline)}</p>
        <p class="dfd-reason-body">{escape(detail)}</p></div></div>""")


def signals_block(attention_text: str, metadata_text: str) -> str:
    def row(name, value, note):
        return (f'<div class="dfd-signal"><span class="dfd-sig-k">{escape(name)}</span>'
                f'<span class="dfd-sig-v">{escape(value)}</span>'
                f'<span class="dfd-sig-note">{escape(note)}</span></div>')
    return (
        '<div class="dfd-signals"><p class="dfd-eyebrow">Signals</p>'
        + row("Attention region", attention_text,
              "Where the model's score was most sensitive. This is attention, not evidence of manipulation.")
        + row("File metadata", metadata_text,
              "Informational only. The model does not use it, and metadata can be stripped or forged.")
        + row("Model scope", md.MODEL_SCOPE_STATUS, md.MODEL_SCOPE_NOTE)
        + "</div>")


def footer_strip(latency_ms) -> str:
    latency = (f'<span class="dfd-latency">Analyzed in {latency_ms:,} ms</span>'
               if latency_ms is not None else "")
    return f'<div class="dfd-foot"><span>EfficientNet-B0 · 224×224 input</span>{latency}</div>'


def metadata_card(meta: dict) -> str:
    chip = '<span class="dfd-chip dfd-chip--info">Informational only</span>'
    if not meta.get("readable"):
        rows = '<p class="dfd-img-cap">File metadata could not be read.</p>'
    else:
        def row(name, value, absent="Not present"):
            cell = (f'<span class="dfd-v">{escape(str(value))}</span>' if value
                    else f'<span class="dfd-v dfd-na">{absent}</span>')
            return f'<div class="dfd-row"><span class="dfd-k">{escape(name)}</span>{cell}</div>'
        dims = f"{meta['width']} × {meta['height']} px" if meta.get("width") else None
        rows = '<div class="dfd-kv">' + "".join([
            row("Dimensions", dims, "Unknown"),
            row("Format", meta.get("format"), "Unknown"),
            row("File size", meta.get("size_label")),
            row("Camera", meta.get("camera")),
            row("Software", meta.get("software")),
            row("Modified (EXIF)", meta.get("modified")),
        ]) + "</div>"
        if not meta.get("has_exif"):
            rows += '<p class="dfd-img-cap">No EXIF metadata found in this file.</p>'
    note = ("Supplementary information only. The model's verdict does not use file metadata. "
            "Metadata can be stripped or forged, so its presence or absence is not evidence "
            "of authenticity.")
    return card("File metadata", f'{rows}<p class="dfd-note">{note}</p>', right=chip)


def image_html(uri: str, alt: str) -> str:
    return f'<img class="dfd-img" src="{uri}" alt="{escape(alt)}"/>'


def caption(text: str) -> str:
    return f'<p class="dfd-img-cap">{escape(text)}</p>'


def hint(text: str) -> str:
    return f'<p class="dfd-hint">{escape(text)}</p>'


def eyebrow(text: str, card: bool = False) -> str:
    # card=True marks this element's parent container to be styled as a card (see theme.py)
    return f'<p class="dfd-eyebrow{" dfd-anchor" if card else ""}">{escape(text)}</p>'
