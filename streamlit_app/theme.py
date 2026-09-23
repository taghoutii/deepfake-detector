import re

# Single light palette. `primary` (blue) marks interactive and section-header elements; synth / auth /
# inc are reserved for the three verdict states.
PALETTE = {
    "bg": "#F6F8FB", "surface": "#FFFFFF", "surface2": "#EEF3F9",
    "border": "#DCE4EF", "border_strong": "#B9C6DA",
    "text": "#16202E", "muted": "#5B6B82",
    "primary": "#2563EB", "primary_soft": "rgba(37,99,235,0.08)",
    "synth": "#D9534F", "synth_fill": "#D9534F", "synth_soft": "rgba(217,83,79,0.08)",
    "auth": "#1E9E6B", "auth_fill": "#1E9E6B", "auth_soft": "rgba(30,158,107,0.08)",
    "inc": "#C58B12", "inc_fill": "#C58B12", "inc_soft": "rgba(197,139,18,0.08)",
    "shadow": "0 1px 3px rgba(22,32,46,0.08)",
}

_SANS = '-apple-system,"Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif'
_VARS = ":root { " + " ".join(f"--{k.replace('_', '-')}:{v};" for k, v in PALETTE.items()) + f" --sans:{_SANS}; }}"

_CARD_SEL = '[data-testid="stVerticalBlock"]:has(> :is(.element-container, [data-testid="stElementContainer"]) .dfd-anchor)'

# Overrides for Streamlit's own elements. Selectors already target Streamlit test ids / classes.
_CHROME = """
/* hide the element container that only carries this stylesheet */
.element-container:has(.dfd-style), [data-testid="stElementContainer"]:has(.dfd-style) { display:none; }

.stApp, [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--text); font-family:var(--sans); }
[data-testid="stDecoration"] { display:none; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stHeader"] * { color:var(--muted) !important; }
.block-container { max-width:1100px; padding-top:2rem; padding-bottom:3rem; }

.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"] { color:var(--text); }
.stApp [data-testid="stCaptionContainer"], .stApp small { color:var(--muted) !important; }
[data-testid="stWidgetLabel"] *, .stToggle label *, [data-testid="stToggle"] * { color:var(--text) !important; }

/* uploader: solid border; primary on hover / focus. Drag-active is Streamlit's own inset ring, which uses primaryColor from .streamlit/config.toml */
[data-testid="stFileUploaderDropzone"] { background:var(--surface); border:1.5px solid var(--border-strong); border-radius:10px;
  transition:border-color .15s ease, background-color .15s ease; }
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploaderDropzone"]:focus-within { border-color:var(--primary); background:var(--primary-soft); }
[data-testid="stFileUploaderDropzone"] * { color:var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] svg { color:var(--primary) !important; }
[data-testid="stFileUploaderDropzone"] button { background:var(--primary); border:1px solid var(--primary); border-radius:8px; }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color:#fff !important; }
[data-testid="stFileUploaderFile"] * , [data-testid="stFileUploaderFileName"] { color:var(--text) !important; }
[data-testid="stFileChip"] { background:var(--surface2) !important; border:1px solid var(--border-strong); border-radius:8px; }
[data-testid="stFileChip"] * { color:var(--muted) !important; }
[data-testid="stFileChipName"] { color:var(--text) !important; }
[data-testid="stFileChip"] > div:first-child { background:var(--primary-soft) !important; }
[data-testid="stFileChip"] svg { color:var(--primary) !important; }

/* buttons + focus */
.stButton button { background:var(--surface); color:var(--primary); border:1px solid var(--primary);
  border-radius:8px; font-weight:600; box-shadow:none; }
.stButton button:hover:not(:disabled) { background:var(--primary-soft); border-color:var(--primary); color:var(--primary); }
.stButton button p { color:inherit !important; }
.stApp button:focus-visible, .stApp [role="slider"]:focus-visible, .stApp input:focus-visible { outline:2px solid var(--primary) !important; outline-offset:2px; }

/* slider */
[data-testid="stSlider"] { margin-top:.8rem; width:100% !important; max-width:100% !important; }
[data-testid="stSlider"] [role="slider"] { background:var(--primary); box-shadow:none; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { color:var(--muted) !important; }
[data-testid="stSliderThumbValue"] { color:var(--primary) !important; font-weight:600; }
/* Streamlit 1.39 renders the slider value and tick labels in "Source Code Pro"; keep everything sans */
[data-testid="stSlider"] *, [data-testid="stSliderThumbValue"] { font-family:var(--sans) !important; }

/* native containers styled as cards: the block whose direct child holds a .dfd-anchor */
@CARD@ {
  box-sizing:border-box; background:var(--surface); border:1px solid var(--border); border-radius:12px;
  box-shadow:var(--shadow); padding:1.15rem 1.25rem; }

/* Streamlit 1.39 gives these children (and their markdown wrappers) the block's full outer width,
   ignoring the card padding, so full-width rules and right-aligned text overflow the card */
@CARD@ > :is(.element-container, [data-testid="stElementContainer"]) { width:100% !important; max-width:100% !important; }
@CARD@ > :is(.element-container, [data-testid="stElementContainer"]) [data-testid="stMarkdown"],
@CARD@ > :is(.element-container, [data-testid="stElementContainer"]) [data-testid="stMarkdownContainer"] { width:100% !important; max-width:100% !important; }
""".replace("@CARD@", _CARD_SEL)

# Component rules. _scope() prefixes each selector with `.stApp` so these outrank Streamlit's own
# heading/paragraph rules identically across Streamlit versions.
#
# Label hierarchy (all sans, sentence case):
#   section headers      -> primary, semibold           (.dfd-eyebrow, .dfd-card-title)
#   row / card names     -> text colour, semibold       (.dfd-sig-k, .dfd-report-title)
#   secondary text       -> muted                       (.dfd-tag, .dfd-conf-label, .dfd-report-file, notes)
_COMPONENTS = """
.dfd-brand { display:flex; align-items:center; gap:.65rem; margin:0 0 1.1rem 0; }
.dfd-brand-mark { width:28px; height:28px; border-radius:8px; background:linear-gradient(135deg, var(--primary), var(--auth)); display:inline-block; flex:none; }
.dfd-brand-name { font-weight:700; font-size:1.05rem; letter-spacing:-.005em; color:var(--text); }

.dfd-eyebrow { font-size:.76rem; font-weight:600; letter-spacing:.01em; color:var(--primary); margin:0 0 .6rem 0; line-height:1.3; }
.dfd-tag { font-size:.78rem; font-weight:500; color:var(--muted); margin:0 0 .45rem 0; }
.dfd-title { font-size:1.75rem; font-weight:700; letter-spacing:-.015em; line-height:1.25; color:var(--text); margin:0; padding:0; }
.dfd-sub { color:var(--muted); font-size:.95rem; line-height:1.55; margin:.4rem 0 1.4rem 0; max-width:62ch; }

.dfd-trust { display:flex; flex-wrap:wrap; gap:.5rem .6rem; margin:1rem 0 1.1rem 0; }
.dfd-trust-item { display:inline-flex; align-items:center; gap:.45rem; font-size:.76rem; font-weight:500; color:var(--text);
  background:var(--surface2); border:1px solid var(--border); border-radius:999px; padding:.28rem .7rem; }
.dfd-trust-item::before { content:""; width:6px; height:6px; border-radius:50%; background:var(--primary); display:inline-block; }

.dfd-card { background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:1.15rem 1.25rem; box-shadow:var(--shadow); margin-bottom:1rem; }
.dfd-card-title { font-size:.8rem; font-weight:700; letter-spacing:.01em; color:var(--primary);
  margin:0 0 .8rem 0; display:flex; align-items:center; justify-content:space-between; gap:.5rem; }

.dfd-report-head { display:flex; justify-content:space-between; align-items:baseline; gap:1rem; margin:0 0 1rem 0; padding-bottom:.75rem; border-bottom:1px solid var(--border); }
.dfd-report-title { font-size:.8rem; font-weight:700; color:var(--text); }
.dfd-report-file { font-size:.8rem; font-weight:500; color:var(--muted); text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }

.dfd-verdict--synthetic { --v:var(--synth); --v-fill:var(--synth-fill); --v-soft:var(--synth-soft); }
.dfd-verdict--authentic { --v:var(--auth); --v-fill:var(--auth-fill); --v-soft:var(--auth-soft); }
.dfd-verdict--inconclusive { --v:var(--inc); --v-fill:var(--inc-fill); --v-soft:var(--inc-soft); }
.dfd-verdict-head { border-left:4px solid var(--v); background:var(--v-soft); border-radius:0 8px 8px 0; padding:.85rem 1rem; }
.dfd-verdict-label { font-size:1.9rem; font-weight:760; letter-spacing:.02em; line-height:1.15; color:var(--v); }
.dfd-verdict-lean { font-size:.8rem; font-weight:500; color:var(--muted); margin-top:.35rem; }

.dfd-chip { display:inline-block; font-size:.72rem; font-weight:600; line-height:1.4; padding:.2rem .55rem;
  border-radius:999px; border:1px solid var(--border-strong); color:var(--muted); background:var(--surface2); vertical-align:middle; }
.dfd-chip--info { color:var(--primary); background:var(--primary-soft); border-color:var(--border-strong); }

.dfd-conf { display:flex; align-items:baseline; gap:.7rem; margin:1.15rem 0 .55rem 0; }
.dfd-conf-label { font-size:.8rem; font-weight:500; color:var(--muted); }
.dfd-conf-num { font-size:2rem; font-weight:700; line-height:1.15; font-variant-numeric:tabular-nums; color:var(--text); }
.dfd-seg { position:relative; display:flex; gap:3px; }
.dfd-seg-i { flex:1; height:10px; border-radius:2px; background:var(--surface2); box-shadow:inset 0 0 0 1px var(--border); }
.dfd-seg-i--on { background:var(--v-fill); box-shadow:none; }
.dfd-seg-mark { position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--text); opacity:.75; }
.dfd-seg-scale { position:relative; height:1.1rem; font-size:.72rem; color:var(--muted); margin-top:.45rem; font-variant-numeric:tabular-nums; }
.dfd-seg-scale span { position:absolute; top:0; white-space:nowrap; }
.dfd-seg-scale span:first-child { left:0; }
.dfd-seg-scale span:last-child { right:0; }

.dfd-reason { margin:.9rem 0 0 0; padding-top:.85rem; border-top:1px solid var(--border); }
.dfd-reason-head { font-size:1rem; font-weight:650; line-height:1.4; color:var(--text); margin:0 0 .25rem 0; }
.dfd-reason-body { font-size:.88rem; line-height:1.55; color:var(--muted); margin:0; }

.dfd-signals { margin:1.3rem 0 0 0; padding-top:.9rem; border-top:1px solid var(--border); }
.dfd-signal { display:grid; grid-template-columns:9.2rem 1fr; gap:.15rem 1rem; padding:.6rem 0; border-bottom:1px solid var(--border); align-items:baseline; }
.dfd-signal:last-child { border-bottom:none; }
.dfd-sig-k { font-size:.8rem; font-weight:600; color:var(--text); }
.dfd-sig-v { font-size:.88rem; font-weight:500; color:var(--text); line-height:1.45; }
.dfd-sig-note { grid-column:2; font-size:.78rem; color:var(--muted); line-height:1.5; }

.dfd-foot { display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-top:.9rem; padding:.85rem 0 .15rem 0; border-top:1px solid var(--border);
  font-size:.78rem; color:var(--muted); }
.dfd-latency { font-variant-numeric:tabular-nums; }

.dfd-kv { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); column-gap:2.5rem; margin:0; }
.dfd-row { display:flex; justify-content:space-between; align-items:baseline; gap:1rem; padding:.35rem 0; font-size:.86rem; line-height:1.4; border-bottom:1px solid var(--border); }
.dfd-k { color:var(--muted); }
.dfd-v { font-size:.86rem; font-weight:500; color:var(--text); text-align:right; word-break:break-word; font-variant-numeric:tabular-nums; }
.dfd-v.dfd-na { color:var(--muted); font-weight:400; }
.dfd-kv + .dfd-img-cap { margin-top:.9rem; }
.dfd-note { font-size:.78rem; color:var(--muted); line-height:1.5; margin:.9rem 0 0 0; padding-top:.8rem; border-top:1px solid var(--border); }

.dfd-img { display:block; width:100%; height:auto; object-fit:fill !important; border-radius:8px; border:1px solid var(--border); background:var(--surface2); }
.dfd-img-cap { font-size:.78rem; color:var(--muted); margin:.6rem 0 0 0; line-height:1.5; }
.dfd-hint { font-size:.85rem; color:var(--muted); margin:.3rem 0 .35rem 0; line-height:1.5; }
"""


def _scope(css: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    def prefix(match):
        selectors = ",".join(f".stApp {s.strip()}" for s in match.group(1).split(","))
        return f"{selectors}{{{match.group(2)}}}"

    return re.sub(r"([^{}]+)\{([^{}]*)\}", prefix, css)


_SCOPED_COMPONENTS = _scope(_COMPONENTS)


def get_css() -> str:
    return f'<style>{_VARS}{_CHROME}{_SCOPED_COMPONENTS}</style><span class="dfd-style"></span>'
