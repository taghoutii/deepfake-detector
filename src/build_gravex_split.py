"""Build the GRAVEX-augmented dataset (data/processed_v2) without touching data/processed.

Two stages, so the plan can be inspected/evaluated before any files are copied:

    python -m src.build_gravex_split plan          # writes manifest.csv + runs integrity checks
    python -m src.build_gravex_split materialize   # copies files into data/processed_v2 and data/probes

Design (see the sampling proposal):
  * data/processed (4000+4000 / 1000+1000 / 1000+1000, StyleGAN vs FFHQ) is carried over unchanged.
  * All 9,000 Stable Diffusion fakes are used. Split is by (gender, index) so a given index goes to
    the same split at every resolution: 800/200/500 indices per gender -> 1600/400/1000 per version.
  * Reals for SD come from GRAVEX's own re-encoded FFHQ copies (same JPEG/resize pipeline as the SD
    fakes), so file encoding is not predictive of the label. Each is drawn from the same 140k split it
    belongs to and never from an image already in data/processed.
  * FF++ / DFDC / Celeb-DF / ai_detect are never trained on: 300 fake + 300 real per source (ai_detect: all)
    go to data/probes as eval-only sets. GRAVEX's augmented (aug_ai_*, Aumented real) files are excluded.
"""
import argparse
import csv
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
GRAVEX_DIR    = Path(os.getenv("GRAVEX_DIR", ROOT / "data/raw_gravex200k/my_real_vs_ai_dataset/my_real_vs_ai_dataset"))
RAW_DIR       = Path(os.getenv("DATA_RAW_DIR", ROOT / "data/raw/real_vs_fake/real_vs_fake"))
PROCESSED_DIR = ROOT / "data/processed"
OUT_DIR       = ROOT / "data/processed_v2"
PROBE_DIR     = ROOT / "data/probes"
MANIFEST      = OUT_DIR / "manifest.csv"

SEED = 42
SD_INDICES_PER_GENDER = {"train": 800, "val": 200, "test": 500}   # of 1500 per gender
FFHQ_REAL_COUNTS      = {"train": 4800, "val": 1200, "test": 3000}
PROBE_N               = 300
FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "DeepFakeDetection"]
RAW_SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}   # our split name -> 140k folder name

FIELDS = ["split", "label", "source", "version", "gender", "group_id", "src_path", "dest_rel", "orig_path"]


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def classify(sub, name):
    """Map a GRAVEX filename to a source family (None = excluded)."""
    s = Path(name).stem
    if s.startswith("aug_ai_") or s.startswith("Aumented"):
        return "augmented"
    if s.startswith("ai_detect"):
        return "ai_detect"
    if s.startswith("dfdc"):
        return "dfdc"
    if s.startswith("celebdf"):
        return "celebdf"
    if s.startswith("fake_") or re.fullmatch(r"real_\d+_\d+", s):
        return "ffpp"
    if re.fullmatch(r"\d+_(woman|man)_(SD|men)_\d+", s):
        return "sd"
    if sub == "real" and s.isdigit():
        return "ffhq_reencoded"
    if sub == "ai_images" and re.fullmatch(r"[A-Z0-9]{10}", s):
        return "stylegan_reencoded"     # same images as the 140k fakes; not used
    return "other"


def row(split, label, source, src_path, dest_rel, version="", gender="", group_id="", orig_path=""):
    return dict(split=split, label=label, source=source, version=version, gender=gender,
                group_id=group_id, src_path=rel(src_path), dest_rel=dest_rel,
                orig_path=rel(orig_path) if orig_path else "")


def plan():
    rng = random.Random(SEED)
    rows = []

    # 1) existing processed set, unchanged
    processed_real_names = set()
    for split in ("train", "val", "test"):
        for label in ("fake", "real"):
            for f in sorted((PROCESSED_DIR / split / label).iterdir()):
                if label == "real":
                    processed_real_names.add(f.name)
                src = "stylegan_140k" if label == "fake" else "ffhq_140k"
                rows.append(row(split, label, src, f, f"{split}/{label}/{f.name}"))

    # 2) Stable Diffusion, split by (gender, index)
    sd = defaultdict(dict)   # (gender, idx) -> {res: path}
    for f in (GRAVEX_DIR / "ai_images").iterdir():
        m = re.fullmatch(r"(\d+)_(woman|man)_(?:SD|men)_(\d+)\.jpg", f.name)
        if m:
            sd[(m.group(2), int(m.group(3)))][m.group(1)] = f
    for gender in ("woman", "man"):
        idxs = sorted(i for g, i in sd if g == gender)
        rng.shuffle(idxs)
        n_tr, n_va = SD_INDICES_PER_GENDER["train"], SD_INDICES_PER_GENDER["val"]
        assign = {"train": idxs[:n_tr], "val": idxs[n_tr:n_tr + n_va], "test": idxs[n_tr + n_va:]}
        for split, ids in assign.items():
            for i in sorted(ids):
                for res, path in sorted(sd[(gender, i)].items()):
                    rows.append(row(split, "fake", "sd", path, f"{split}/fake/{path.name}",
                                    version=f"{res}px", gender=gender, group_id=f"{gender}_{i:04d}"))

    # 3) FFHQ reals (GRAVEX re-encoded copies), matched to SD counts, leak-guarded
    raw_split_of = {}
    for split, d in RAW_SPLIT_DIRS.items():
        for f in (RAW_DIR / d / "real").iterdir():
            raw_split_of[f.name] = split
    gravex_ffhq = sorted(f for f in (GRAVEX_DIR / "real").iterdir() if classify("real", f.name) == "ffhq_reencoded")
    for split, n in FFHQ_REAL_COUNTS.items():
        cands = [f for f in gravex_ffhq
                 if raw_split_of.get(f.name) == split and f.name not in processed_real_names]
        assert len(cands) >= n, f"only {len(cands)} FFHQ candidates for {split}, need {n}"
        for f in rng.sample(cands, n):
            orig = RAW_DIR / RAW_SPLIT_DIRS[split] / "real" / f.name
            rows.append(row(split, "real", "ffhq_reencoded", f, f"{split}/real/gravex_ffhq_{f.name}",
                            orig_path=orig))

    # 4) eval-only probes
    by_source = defaultdict(lambda: {"fake": [], "real": []})
    for sub, label in (("ai_images", "fake"), ("real", "real")):
        for f in sorted((GRAVEX_DIR / sub).iterdir()):
            fam = classify(sub, f.name)
            if fam in ("ffpp", "dfdc", "celebdf", "ai_detect"):
                by_source[fam][label].append(f)
    for fam in ("ffpp", "dfdc", "celebdf", "ai_detect"):
        for label in ("fake", "real"):
            pool = by_source[fam][label]
            if fam == "ai_detect":
                chosen = pool                                   # small: use all
            elif fam == "ffpp" and label == "fake":             # stratify across manipulation methods
                per = PROBE_N // len(FFPP_METHODS)
                chosen = []
                for m in FFPP_METHODS:
                    sub_pool = [f for f in pool if f.name.startswith(f"fake_{m}_")]
                    chosen += rng.sample(sub_pool, per)
            else:
                chosen = rng.sample(pool, PROBE_N)
            for f in chosen:
                rows.append(row(f"probe_{fam}", label, fam, f, f"{fam}/{label}/{f.name}"))

    verify(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\nManifest written: {rel(MANIFEST)} ({len(rows)} rows)")


def verify(rows):
    """Hard integrity checks; raises AssertionError on any violation."""
    dests = Counter((r["split"].startswith("probe_"), r["dest_rel"]) for r in rows)
    assert max(dests.values()) == 1, "duplicate destination paths"

    # SD: every (gender, index) lives in exactly one split, all 3 resolutions together
    grp = defaultdict(set)
    for r in rows:
        if r["source"] == "sd":
            grp[r["group_id"]].add((r["split"]))
    assert all(len(s) == 1 for s in grp.values()), "SD index split across train/val/test"
    res_per_grp = Counter(r["group_id"] for r in rows if r["source"] == "sd")
    assert set(res_per_grp.values()) == {3}, "SD index missing a resolution"

    # FFHQ leak guard: no filename appears in more than one split, nor in data/processed
    ffhq_new = {Path(r["src_path"]).name: r["split"] for r in rows if r["source"] == "ffhq_reencoded"}
    ffhq_old = {Path(r["src_path"]).name for r in rows if r["source"] == "ffhq_140k"}
    assert len(ffhq_new) == sum(FFHQ_REAL_COUNTS.values()), "duplicate FFHQ picks"
    assert not (set(ffhq_new) & ffhq_old), "FFHQ real overlaps data/processed"
    for r in rows:
        if r["source"] == "ffhq_reencoded":
            assert (ROOT / r["orig_path"]).exists(), f"missing original-encoding copy {r['orig_path']}"

    # no augmented files anywhere; all source files exist
    for r in rows:
        assert not Path(r["src_path"]).name.startswith(("aug_ai_", "Aumented")), r["src_path"]
        assert (ROOT / r["src_path"]).exists(), r["src_path"]

    # class balance per train/val/test split, and expected SD counts
    for split in ("train", "val", "test"):
        c = Counter(r["label"] for r in rows if r["split"] == split)
        assert c["fake"] == c["real"], f"{split} not balanced: {c}"
    for split, n in SD_INDICES_PER_GENDER.items():
        assert sum(1 for r in rows if r["split"] == split and r["source"] == "sd") == n * 2 * 3


def summarize(rows):
    print(f"{'split':<18}{'source':<18}{'label':<6}{'count':>7}")
    c = Counter((r["split"], r["source"], r["label"]) for r in rows)
    for (split, source, label), n in sorted(c.items(), key=lambda x: (x[0][0].startswith("probe"), x[0])):
        print(f"{split:<18}{source:<18}{label:<6}{n:>7}")
    print()
    for split in ("train", "val", "test"):
        c = Counter(r["label"] for r in rows if r["split"] == split)
        print(f"{split}: {sum(c.values())} images  ({c['real']} real / {c['fake']} fake)")
    print("integrity checks: OK")


def materialize(force=False):
    assert MANIFEST.exists(), "run `plan` first"
    if not force:
        for split in ("train", "val", "test"):
            assert not (OUT_DIR / split).exists(), f"{OUT_DIR / split} exists; pass --force to overwrite"
    with open(MANIFEST, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        dest_root = PROBE_DIR if r["split"].startswith("probe_") else OUT_DIR
        dest = dest_root / r["dest_rel"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r["src_path"], dest)
    print(f"Copied {len(rows)} files -> {rel(OUT_DIR)} and {rel(PROBE_DIR)}")
    for split in ("train", "val", "test"):
        for label in ("fake", "real"):
            print(f"  {split}/{label}: {len(list((OUT_DIR / split / label).iterdir()))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "materialize"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    plan() if a.stage == "plan" else materialize(a.force)
