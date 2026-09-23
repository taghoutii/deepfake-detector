import base64
import io
import re
import sys
from pathlib import Path

import pytest
import requests
from PIL import Image
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "streamlit_app"
sys.path.insert(0, str(APP_DIR))

import components as ui  # noqa: E402
import metadata as md    # noqa: E402


def jpeg_bytes(color=(120, 80, 60), size=(40, 30), exif=None):
    buf = io.BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    Image.new("RGB", size, color).save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


def overlay_b64():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (200, 40, 40)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def photo_like(size=(256, 256), seed=3):
    """A deterministic, smoothly varying colour image (stands in for a photo; no bundled samples)."""
    import numpy as np
    base = np.random.default_rng(seed).random((size[1] // 16 + 1, size[0] // 16 + 1, 3))
    return Image.fromarray((base * 255).astype("uint8")).resize(size, Image.BICUBIC)


def photo_jpeg_bytes(seed=3):
    buf = io.BytesIO()
    photo_like(seed=seed).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ---------- metadata ----------

def test_image_metadata_basic():
    data = jpeg_bytes(size=(40, 30))
    meta = md.image_metadata(data)
    assert meta["readable"] and (meta["width"], meta["height"]) == (40, 30)
    assert meta["format"] == "JPEG"
    assert meta["size_bytes"] == len(data)
    assert not meta["has_exif"] and meta["camera"] is None


def test_image_metadata_reads_exif_fields():
    exif = Image.Exif()
    exif[271], exif[272] = "Canon", "Canon EOS 5D"
    exif[305], exif[306] = "Adobe Photoshop 25.0", "2024:05:01 10:22:33"
    meta = md.image_metadata(jpeg_bytes(exif=exif))
    assert meta["has_exif"]
    assert meta["camera"] == "Canon EOS 5D"
    assert meta["software"] == "Adobe Photoshop 25.0"
    assert meta["modified"] == "2024-05-01 10:22:33"


def test_image_metadata_unreadable_never_raises():
    meta = md.image_metadata(b"definitely not an image")
    assert meta["readable"] is False


def test_exif_strings_are_sanitised_and_truncated():
    exif = Image.Exif()
    exif[305] = "A\x00B\x07" + "x" * 500
    software = md.image_metadata(jpeg_bytes(exif=exif))["software"]
    assert "\x00" not in software and "\x07" not in software
    assert len(software) <= md.MAX_FIELD_LEN


def test_metadata_card_escapes_html_from_exif():
    meta = md.image_metadata(jpeg_bytes())
    meta["software"] = "<script>alert(1)</script>"
    html = ui.metadata_card(meta)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_metadata_card_renders_six_labelled_rows():
    html = ui.metadata_card(md.image_metadata(jpeg_bytes(size=(40, 30))))
    assert html.count('class="dfd-row"') == 6
    for label in ("Dimensions", "Format", "File size", "Camera", "Software", "Modified (EXIF)"):
        assert f'<span class="dfd-k">{label}</span>' in html
    assert "40 × 30 px" in html


def test_component_css_is_scoped_under_stapp():
    import theme
    selectors = [s.strip() for group in re.findall(r"([^{}]+)\{", theme._SCOPED_COMPONENTS)
                 for s in group.split(",")]
    assert selectors and all(s.startswith(".stApp ") for s in selectors)
    assert ".stApp .dfd-title{" in theme.get_css()


def test_metadata_card_labels_signal_as_informational_only():
    html = ui.metadata_card(md.image_metadata(jpeg_bytes()))
    assert "Informational only" in html
    assert "does not use file metadata" in html
    assert "stripped or forged" in html
    assert "No EXIF metadata found" in html


# ---------- reasoning / confidence ----------

@pytest.mark.parametrize("conf,band", [(0.5, "low"), (0.699, "low"), (0.70, "moderate"),
                                       (0.899, "moderate"), (0.90, "high"), (1.0, "high")])
def test_confidence_bands(conf, band):
    assert md.confidence_band(conf) == band


def test_reasoning_fake_high():
    head, detail = md.verdict_reasoning("fake", 0.97)
    assert head == "Flagged as synthetic (high confidence)."
    assert detail == ("The highlighted regions influenced this score most. "
                      "Heatmaps show model attention, not proof of manipulation.")


def test_reasoning_real_moderate():
    head, detail = md.verdict_reasoning("real", 0.80)
    assert head == "Classified as authentic (moderate confidence)."
    assert "not proof of authenticity" in detail


@pytest.mark.parametrize("label", ["fake", "real"])
def test_reasoning_low_confidence_is_inconclusive(label):
    head, detail = md.verdict_reasoning(label, 0.61)
    assert head == "Inconclusive — manual review recommended."
    assert "61.0%" in detail and "70%" in detail


# ---------- blend ----------

def test_blend_overlay_endpoints_and_clamping():
    original = Image.new("RGB", (64, 64), (10, 10, 10))
    overlay  = Image.new("RGB", (224, 224), (250, 0, 0))
    assert md.blend_overlay(original, overlay, 0).getpixel((5, 5)) == (10, 10, 10)
    assert md.blend_overlay(original, overlay, 1).getpixel((5, 5)) == (250, 0, 0)
    assert md.blend_overlay(original, overlay, 5).getpixel((5, 5)) == (250, 0, 0)
    assert md.blend_overlay(original, overlay, -3).getpixel((5, 5)) == (10, 10, 10)
    mid = md.blend_overlay(original, overlay, 0.5).getpixel((5, 5))
    assert 120 <= mid[0] <= 130
    assert md.blend_overlay(original, overlay, 0.5).size == (224, 224)


# ---------- components ----------

def test_verdict_block_synthetic_high():
    html = ui.verdict_block("fake", 0.973)
    assert "dfd-verdict--synthetic" in html and ">LIKELY SYNTHETIC<" in html
    assert "97.3%" in html and 'aria-valuenow="97.3"' in html
    assert "INCONCLUSIVE" not in html and "Leans" not in html


def test_verdict_block_inconclusive_below_threshold():
    html = ui.verdict_block("real", 0.62)
    assert "dfd-verdict--inconclusive" in html and ">INCONCLUSIVE<" in html
    assert "Leans authentic" in html
    assert "Inconclusive — manual review recommended." in html


@pytest.mark.parametrize("label", ["maybe", "", "FAKE", None])
def test_components_refuse_unknown_labels(label):
    with pytest.raises(ValueError):
        ui.verdict_block(label, 0.9)


def test_report_head_escapes_filename():
    meta = md.image_metadata(jpeg_bytes())
    html = ui.report_head('"><img src=x onerror=alert(1)>.jpg', meta)
    assert "<img src=x" not in html and "&lt;img" in html


# ---------- the app script, with the API mocked ----------

class FakeResponse:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code, self._body, self._bad = status, body, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


def ok_body(label="fake", confidence=0.973, gradcam=None):
    return {"prediction": label, "confidence": confidence,
            "gradcam_image": overlay_b64() if gradcam is None else gradcam}


@pytest.fixture
def api(monkeypatch):
    state = {"calls": 0, "respond": lambda: FakeResponse(200, ok_body())}

    def fake_post(url, files=None, timeout=None):
        state["calls"] += 1
        result = state["respond"]()
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(requests, "post", fake_post)
    return state


def new_app():
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
    at.run()
    return at


def analyze(at, name="photo.jpg", data=None):
    """Drive the app like an upload does: the uploader hands the file to `pending`."""
    at.session_state["pending"] = {"name": name, "data": data or photo_jpeg_bytes(), "type": "image/jpeg"}
    at.run()
    return at


def page(at):
    return "".join(m.value for m in at.markdown)


def copy_only(text):
    return re.sub(r"<style>.*?</style>", "", text, flags=re.S)   # the stylesheet names every state


def test_initial_render_is_light_single_page_and_prompts_for_upload(api):
    at = new_app()
    assert not at.exception
    html = page(at)
    assert "--bg:#F6F8FB" in html and "--bg:#0A0B0D" not in html
    assert "Results appear below once the image has been analyzed." in html
    assert "Analysis report" not in html
    assert len(at.toggle) == 0 and len(at.button) == 0
    assert api["calls"] == 0


def test_theme_query_param_is_ignored(api):
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
    at.query_params["theme"] = "dark"
    at.run()
    assert not at.exception
    assert "--bg:#F6F8FB" in page(at) and "--bg:#0A0B0D" not in page(at)


def test_app_source_has_no_sidebar_toggle_samples_or_history():
    src = (APP_DIR / "app.py").read_text(encoding="utf-8")
    for gone in ("st.sidebar", "st.toggle", "dark_mode", "load_samples", "history", "SAMPLE_DIR", "sample_images"):
        assert gone not in src, gone
    assert not (APP_DIR / "sample_images").exists()
    for name in ("sample_thumb", "history_item", "empty_state"):
        assert not hasattr(ui, name), name


def test_analysis_shows_full_result_in_one_card(api):
    at = analyze(new_app())
    assert not at.exception
    html = page(at)
    assert "dfd-verdict--synthetic" in html and ">LIKELY SYNTHETIC<" in html and "97.3%" in html
    assert "Analyzed in" in html and "File metadata" in html and "Attention region" in html
    assert "Model scope" in html and "photo.jpg" in html
    assert "Results appear below" not in html
    assert api["calls"] == 1


def test_slider_does_not_recall_the_api(api):
    at = analyze(new_app())
    at.slider(key="overlay_strength").set_value(30).run()
    at.slider(key="overlay_strength").set_value(0).run()
    assert not at.exception
    assert api["calls"] == 1


def test_new_analysis_replaces_the_previous_result_and_nothing_persists(api):
    at = analyze(new_app(), name="first.jpg", data=photo_jpeg_bytes(seed=1))
    api["respond"] = lambda: FakeResponse(200, ok_body(label="real", confidence=0.95))
    analyze(at, name="second.jpg", data=photo_jpeg_bytes(seed=2))
    html = copy_only(page(at))
    assert "second.jpg" in html and "first.jpg" not in html
    assert "dfd-verdict--authentic" in html and "dfd-verdict--synthetic" not in html
    assert api["calls"] == 2
    assert not any("hist" in k.lower() for k in at.session_state.filtered_state)


def test_removing_the_uploaded_file_clears_the_result(api):
    at = analyze(new_app())
    assert at.session_state["entry"] is not None
    at.session_state["last_upload_hash"] = "hash-of-a-file-the-user-then-removed"
    at.run()
    assert at.session_state["entry"] is None
    assert "Analysis report" not in page(at) and "Results appear below" in page(at)


@pytest.mark.parametrize("respond,expected", [
    (lambda: requests.exceptions.ConnectionError(), "Cannot connect to the API"),
    (lambda: requests.exceptions.ConnectTimeout(), "Cannot connect to the API"),
    (lambda: requests.exceptions.ReadTimeout(), "took too long to respond"),
    (lambda: requests.exceptions.ChunkedEncodingError(), "request to the API failed"),
    (lambda: FakeResponse(400, {"detail": "Could not read image file."}), "Bad request: Could not read image file."),
    (lambda: FakeResponse(413, {"detail": "File too large. Max size is 10MB."}), "File too large"),
    (lambda: FakeResponse(500, {}), "API error 500."),
    (lambda: FakeResponse(200, bad_json=True), "unexpected response"),
    (lambda: FakeResponse(200, {"prediction": "maybe", "confidence": 0.9}), "unexpected response"),
    (lambda: FakeResponse(200, {"prediction": "fake", "confidence": 7}), "unexpected response"),
    (lambda: FakeResponse(200, {"confidence": 0.9}), "unexpected response"),
])
def test_api_failures_show_friendly_error_not_traceback(api, respond, expected):
    api["respond"] = respond
    at = analyze(new_app())
    assert not at.exception
    assert len(at.error) == 1 and expected in at.error[0].value
    assert at.session_state["entry"] is None


def test_retry_after_failure_recovers(api):
    api["respond"] = lambda: requests.exceptions.ConnectionError()
    at = analyze(new_app())
    assert at.error
    api["respond"] = lambda: FakeResponse(200, ok_body())
    next(b for b in at.button if b.label == "Retry").click().run()
    assert not at.exception and not at.error
    assert "dfd-verdict--synthetic" in page(at)


def test_missing_or_corrupt_gradcam_degrades_gracefully(api):
    for bad in ("", "%%%not-base64%%%"):
        api["respond"] = lambda bad=bad: FakeResponse(200, ok_body(gradcam=bad))
        at = analyze(new_app())
        assert not at.exception
        assert "Grad-CAM not available" in page(at)
        assert "dfd-verdict--synthetic" in page(at)
        assert "Not available" in page(at)


# =====================================================================================
# Report card: verdict states, signal rows, trust badges
# =====================================================================================

def gradcam_like_overlay(original, cx, cy, sigma=30):
    """A mask + the exact overlay pytorch-grad-cam would return for it (JET, image_weight=0.5)."""
    import numpy as np
    from pytorch_grad_cam.utils.image import show_cam_on_image
    yy, xx = np.mgrid[0:224, 0:224]
    mask = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)
    mask /= mask.max()
    base = np.asarray(original.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    return mask, Image.fromarray(show_cam_on_image(base, mask, use_rgb=True))


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("label,conf,state", [
    ("fake", 0.699, "inconclusive"), ("real", 0.699, "inconclusive"), ("fake", 0.5, "inconclusive"),
    ("fake", 0.70, "synthetic"), ("fake", 1.0, "synthetic"),
    ("real", 0.70, "authentic"), ("real", 0.95, "authentic"),
])
def test_verdict_state_uses_the_70_percent_threshold(label, conf, state):
    assert md.verdict_state(label, conf) == state


def test_verdict_labels_are_hedged_and_symmetric():
    assert md.VERDICT_TEXT == {"synthetic": "LIKELY SYNTHETIC", "authentic": "LIKELY AUTHENTIC",
                               "inconclusive": "INCONCLUSIVE"}


def test_verdict_state_rejects_unknown_label():
    with pytest.raises(ValueError):
        md.verdict_state("maybe", 0.9)


def test_footer_strip_shows_model_and_latency():
    html = ui.footer_strip(340)
    assert "Analyzed in 340 ms" in html and "EfficientNet-B0" in html and "224×224" in html
    assert "Analyzed in" not in ui.footer_strip(None)


@pytest.mark.parametrize("meta,expected", [
    ({"readable": False}, "Could not be read"),
    ({"readable": True, "has_exif": False}, "No EXIF present"),
    ({"readable": True, "has_exif": True, "camera": "Canon EOS R5", "software": "Adobe Photoshop"},
     "EXIF present · camera + software tags recorded"),
    ({"readable": True, "has_exif": True, "camera": "Canon EOS R5", "software": None},
     "EXIF present · camera tag recorded"),
    ({"readable": True, "has_exif": True, "camera": None, "software": "GIMP"},
     "EXIF present · software tag recorded"),
    ({"readable": True, "has_exif": True, "camera": None, "software": None},
     "EXIF present · no camera or software tags"),
])
def test_metadata_finding_is_neutral_and_precise(meta, expected):
    assert md.metadata_finding(meta) == expected


def test_signals_block_has_three_named_neutral_rows():
    html = ui.signals_block("Peak: centre · 12% of image above 50% intensity", "No EXIF present")
    assert html.count('class="dfd-signal"') == 3
    for name in ("Attention region", "File metadata", "Model scope"):
        assert f'<span class="dfd-sig-k">{name}</span>' in html
    assert "not evidence of manipulation" in html
    assert "can be stripped or forged" in html
    assert md.MODEL_SCOPE_NOTE in html and md.MODEL_SCOPE_STATUS in html
    # rows are plain readouts: no verdict-state colour classes anywhere inside them
    assert "dfd-verdict--" not in html and "dfd-chip--" not in html


def test_signals_block_escapes_values():
    html = ui.signals_block("<b>x</b>", "<script>alert(1)</script>")
    assert "<script>" not in html and "&lt;script&gt;" in html and "<b>x</b>" not in html


def test_trust_strip_has_exactly_the_four_approved_badges():
    html = ui.trust_strip()
    items = re.findall(r'class="dfd-trust-item">([^<]+)<', html)
    assert items == ["Self-hosted model", "Not stored", "No third-party requests", "Images only · JPG PNG"]


# ---------- attention region recovered from the overlay ----------

def test_attention_summary_recovers_peak_and_coverage_from_real_gradcam_blend():
    original = photo_like()
    for (cx, cy), peak in [((180, 180), "lower-right"), ((40, 40), "upper-left"), ((112, 112), "centre"),
                           ((40, 112), "centre-left"), ((112, 190), "lower-centre")]:
        mask, overlay = gradcam_like_overlay(original, cx, cy)
        got = md.attention_summary(original, overlay)
        assert got is not None and got["peak"] == peak, (cx, cy, got)
        assert abs(got["coverage_pct"] - float((mask >= 0.5).mean() * 100)) < 1.5


def test_attention_summary_works_when_the_preview_is_not_square_or_a_different_size():
    original = Image.new("RGB", (900, 500), (90, 120, 150))
    _, overlay = gradcam_like_overlay(original, 180, 60)
    preview = original.copy()
    preview.thumbnail((512, 512))            # what app.py keeps
    assert md.attention_summary(preview, overlay)["peak"] == "upper-right"


def _noise_overlay():
    import numpy as np
    return Image.fromarray((np.random.default_rng(0).random((224, 224, 3)) * 255).astype("uint8"))


@pytest.mark.parametrize("make_overlay", [lambda: Image.new("RGB", (224, 224), (200, 40, 40)), _noise_overlay])
def test_attention_summary_returns_none_when_overlay_is_not_a_gradcam_blend(make_overlay):
    assert md.attention_summary(Image.new("RGB", (224, 224), (100, 100, 100)), make_overlay()) is None


def test_attention_status_texts():
    assert md.attention_status({"peak": "lower-left", "coverage_pct": 14.0}) == \
        "Peak: lower-left · 14% of image above 50% intensity"
    assert md.attention_status(None) == "Not estimated"
    assert md.attention_status(None, available=False) == "Not available"


# ---------- palette, type and control styling ----------

SPEC_PALETTE = {
    "bg": "#F6F8FB", "surface": "#FFFFFF", "surface2": "#EEF3F9",
    "border": "#DCE4EF", "border_strong": "#B9C6DA",
    "text": "#16202E", "muted": "#5B6B82",
    "primary": "#2563EB", "primary_soft": "rgba(37,99,235,0.08)",
    "synth": "#D9534F", "synth_fill": "#D9534F", "synth_soft": "rgba(217,83,79,0.08)",
    "auth": "#1E9E6B", "auth_fill": "#1E9E6B", "auth_soft": "rgba(30,158,107,0.08)",
    "inc": "#C58B12", "inc_fill": "#C58B12", "inc_soft": "rgba(197,139,18,0.08)",
    "shadow": "0 1px 3px rgba(22,32,46,0.08)",
}


def css_rule(css, cls):
    match = re.search(r"\.stApp \." + re.escape(cls) + r"\{([^}]*)\}", css)
    assert match, f"no scoped rule for .{cls}"
    return match.group(1)


def test_single_light_palette_matches_the_spec_and_has_no_mode_switch():
    import theme
    assert theme.PALETTE == SPEC_PALETTE
    assert not hasattr(theme, "PALETTES")
    with pytest.raises(TypeError):
        theme.get_css("dark")
    css = theme.get_css()
    assert css.count("<style>") == 1
    for key, value in SPEC_PALETTE.items():
        assert f"--{key.replace('_', '-')}:{value};" in css, key


def test_primary_is_distinct_from_the_verdict_colours_and_neutrals_stay_neutral():
    import theme
    verdict = {theme.PALETTE[k] for k in ("synth", "synth_fill", "auth", "auth_fill", "inc", "inc_fill")}
    assert theme.PALETTE["primary"] not in verdict
    for key in ("bg", "surface", "surface2", "border", "border_strong", "text", "muted"):
        r, g, b = (int(theme.PALETTE[key][i:i + 2], 16) for i in (1, 3, 5))
        assert max(r, g, b) - min(r, g, b) <= 40, f"{key} is not a (cool) neutral"


def test_no_monospace_anywhere_and_sans_is_used():
    import theme
    css = theme.get_css().lower()
    for banned in ("--mono", "monospace", "consolas", "menlo", "courier", "cascadia", "jetbrains"):
        assert banned not in css, banned
    assert "--sans:" in css
    # Streamlit 1.39 draws the slider's value and tick labels in a monospace font unless overridden
    assert re.search(r'\[data-testid="stSlider"\] \*[^{]*\{[^}]*font-family:var\(--sans\) !important', theme.get_css())
    for source in ("app.py", "components.py", "theme.py", "metadata.py"):
        text = (APP_DIR / source).read_text(encoding="utf-8").lower()
        assert "monospace" not in text and "--mono" not in text, source
    assert "font-family:var(--sans)" in css


LABEL_CLASSES = ["dfd-eyebrow", "dfd-tag", "dfd-card-title", "dfd-sig-k", "dfd-conf-label", "dfd-report-title"]


def test_label_classes_are_sans_sentence_case_and_visually_distinct():
    import theme
    css = theme.get_css()
    treatments = {}
    for cls in LABEL_CLASSES:
        body = css_rule(css, cls)
        assert "uppercase" not in body and "capitalize" not in body, cls
        assert "font-family" not in body, cls                      # inherits --sans
        size = float(re.search(r"font-size:([\d.]+)rem", body).group(1))
        assert 0.72 <= size <= 0.8, (cls, size)
        spacing = re.search(r"letter-spacing:(-?[\d.]+)em", body)
        assert spacing is None or float(spacing.group(1)) <= 0.02, cls
        treatments[cls] = (re.search(r"color:(var\(--[a-z-]+\))", body).group(1),
                           re.search(r"font-weight:(\d+)", body).group(1))
    # section headers are primary; row/card names use the text colour; only secondary text is muted
    assert treatments["dfd-eyebrow"][0] == treatments["dfd-card-title"][0] == "var(--primary)"
    assert treatments["dfd-sig-k"][0] == treatments["dfd-report-title"][0] == "var(--text)"
    assert treatments["dfd-tag"][0] == treatments["dfd-conf-label"][0] == "var(--muted)"
    assert len(set(treatments.values())) >= 4      # no longer one shared treatment


def test_dropzone_has_a_solid_border_and_turns_primary_on_hover_and_focus():
    import theme
    css = theme.get_css()
    assert "dashed" not in css
    base = re.search(r'\[data-testid="stFileUploaderDropzone"\] \{([^}]*)\}', css).group(1)
    assert "solid var(--border-strong)" in base
    active = re.search(r'\[data-testid="stFileUploaderDropzone"\]:hover[^{]*\{([^}]*)\}', css)
    assert active and "border-color:var(--primary)" in active.group(1)
    assert ':focus-within' in css


def test_controls_and_brand_use_the_primary_blue():
    import theme
    css = theme.get_css()
    assert '[role="slider"] { background:var(--primary)' in css
    assert re.search(r"\.stButton button \{[^}]*color:var\(--primary\)", css)
    assert re.search(r"\[data-testid=\"stFileUploaderDropzone\"\] button \{[^}]*background:var\(--primary\)", css)
    assert "outline:2px solid var(--primary)" in css
    assert "linear-gradient(135deg, var(--primary), var(--auth))" in css_rule(css, "dfd-brand-mark")
    assert "var(--accent" not in css


def test_streamlit_config_matches_the_light_palette_and_keeps_telemetry_off():
    import tomllib
    import theme
    config = tomllib.loads((APP_DIR.parent / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["browser"]["gatherUsageStats"] is False       # the "no third-party requests" badge depends on this
    assert config["theme"]["base"] == "light"
    for key, palette_key in [("primaryColor", "primary"), ("backgroundColor", "bg"),
                             ("secondaryBackgroundColor", "surface2"), ("textColor", "text")]:
        assert config["theme"][key].lower() == theme.PALETTE[palette_key].lower(), key


# ---------- the app: computed rows and scope guard ----------

def test_result_shows_computed_attention_row_and_metadata_finding(api):
    data = photo_jpeg_bytes()
    _, overlay = gradcam_like_overlay(Image.open(io.BytesIO(data)), 180, 180)
    api["respond"] = lambda: FakeResponse(200, ok_body(gradcam=png_b64(overlay)))
    at = analyze(new_app(), data=data)
    assert not at.exception
    html = page(at)
    assert "Peak: lower-right ·" in html and "No EXIF present" in html


def test_inconclusive_result(api):
    api["respond"] = lambda: FakeResponse(200, ok_body(label="real", confidence=0.62))
    at = analyze(new_app())
    html = copy_only(page(at))
    assert not at.exception
    assert "dfd-verdict--inconclusive" in html and "Leans authentic" in html
    assert "dfd-verdict--authentic" not in html


def test_page_never_implies_out_of_scope_features(api):
    forbidden = re.compile(r"\b(pricing|price|login|log in|sign in|sign up|signup|account|blog|subscribe|"
                           r"video|audio|text detection|multi-?modal|retention|free trial)\b", re.I)
    at = new_app()
    empty_text = page(at)
    analyze(at)
    full_text = page(at) + " ".join(str(c.value) for c in at.caption)
    for text in (empty_text, full_text):
        assert not forbidden.search(copy_only(text)), forbidden.search(copy_only(text)).group(0)
    assert "Images only" in empty_text
