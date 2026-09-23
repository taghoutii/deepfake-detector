"""Leave-one-SD-version-out split, derived from data/processed_v2/manifest.csv.

Holds one Stable Diffusion version out of training entirely (all 3,000 of its images,
regardless of which train/val/test split the main v2 run happened to assign them to).
Train/val is otherwise a literal subset of the v2 run's own selections: same StyleGAN
140k portion, same SD fakes and paired GRAVEX-reencoded FFHQ reals for the two kept
versions (just fewer reencoded reals, resampled to keep classes balanced after the
held-out version's fakes are dropped). Nothing here needs re-verifying beyond confirming
the held-out version has zero footprint in train/val, since it's built entirely from
already-verified v2 manifest rows rather than a fresh sample of the raw data.

    python -m src.build_loo_split plan --held-out 1024px
    python -m src.build_loo_split materialize --held-out 1024px
"""
import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2_MANIFEST = ROOT / "data/processed_v2/manifest.csv"
OUT_DIR = ROOT / "data/processed_loo"
SEED = 42
VERSIONS = ["512px", "768px", "1024px"]
FIELDS = ["role", "split", "label", "source", "version", "dest_rel", "src_path"]


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def load_v2():
    with open(V2_MANIFEST, newline="") as fh:
        return list(csv.DictReader(fh))


def manifest_path(held_out):
    return OUT_DIR / f"manifest_{held_out}.csv"


def plan(held_out):
    rng = random.Random(SEED)
    v2 = load_v2()
    rows = []

    def add(role, split, label, source, version, dest_rel, src_path):
        rows.append(dict(role=role, split=split, label=label, source=source, version=version,
                          dest_rel=dest_rel, src_path=src_path))

    for split in ("train", "val"):
        # StyleGAN / FFHQ 140k portion: identical to v2, reused as-is
        for r in v2:
            if r["split"] == split and r["source"] in ("stylegan_140k", "ffhq_140k"):
                add("train_val", split, r["label"], r["source"], "", r["dest_rel"], r["src_path"])

        # SD fakes: v2's own selection, minus the held-out version
        sd_keep = [r for r in v2 if r["split"] == split and r["source"] == "sd" and r["version"] != held_out]
        for r in sd_keep:
            add("train_val", split, "fake", "sd", r["version"], r["dest_rel"], r["src_path"])

        # matching reencoded reals: resample v2's pool down to the new (smaller) fake count
        n_needed = len(sd_keep)
        pool = [r for r in v2 if r["split"] == split and r["source"] == "ffhq_reencoded"]
        assert len(pool) >= n_needed, f"{split}: only {len(pool)} reencoded reals, need {n_needed}"
        for r in rng.sample(pool, n_needed):
            add("train_val", split, "real", "ffhq_reencoded", "", r["dest_rel"], r["src_path"])

    # held-out SD version: every one of its images, regardless of which v2 split it landed in
    for r in v2:
        if r["source"] == "sd" and r["version"] == held_out:
            add("heldout_fake", "heldout", "fake", "sd", held_out, "", r["src_path"])

    # paired reals: v2's TEST reencoded reals — untouched by v2 training AND by this train/val
    for r in v2:
        if r["split"] == "test" and r["source"] == "ffhq_reencoded":
            add("heldout_real", "heldout", "real", "ffhq_reencoded", "", "", r["src_path"])

    # sanity-check set: v2's original StyleGAN/FFHQ test (same one v2's own eval reports on)
    for r in v2:
        if r["split"] == "test" and r["source"] in ("stylegan_140k", "ffhq_140k"):
            add("orig_test", "test", r["label"], r["source"], "", "", r["src_path"])

    verify(rows, held_out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpath = manifest_path(held_out)
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summarize(rows, held_out)
    print(f"\nManifest written: {rel(mpath)} ({len(rows)} rows)")


def verify(rows, held_out):
    tv = [r for r in rows if r["role"] == "train_val"]
    assert not any(r["source"] == "sd" and r["version"] == held_out for r in tv), \
        "held-out version leaked into train/val"
    kept_versions = {r["version"] for r in tv if r["source"] == "sd"}
    assert kept_versions == set(VERSIONS) - {held_out}, f"expected both other versions, got {kept_versions}"

    for split in ("train", "val"):
        c = Counter(r["label"] for r in tv if r["split"] == split)
        assert c["fake"] == c["real"], f"{split} not balanced: {c}"

    dests = Counter((r["split"], r["dest_rel"]) for r in tv)
    assert max(dests.values()) == 1, "duplicate destination in train/val"
    for r in tv:
        assert (ROOT / r["src_path"]).exists(), r["src_path"]

    hf = [r for r in rows if r["role"] == "heldout_fake"]
    hr = [r for r in rows if r["role"] == "heldout_real"]
    assert len(hf) == 3000 and all(r["version"] == held_out for r in hf), "heldout fake set wrong"
    assert len(hr) == 3000, "heldout real set wrong"
    assert not (set(r["src_path"] for r in hf) & set(r["src_path"] for r in tv)), "heldout fake overlaps train/val"
    assert not (set(r["src_path"] for r in hr) & set(r["src_path"] for r in tv)), "heldout real overlaps train/val"

    ot = [r for r in rows if r["role"] == "orig_test"]
    assert len(ot) == 2000


def summarize(rows, held_out):
    print(f"held-out SD version: {held_out}\n")
    c = Counter((r["split"], r["source"], r["label"]) for r in rows if r["role"] == "train_val")
    for k, n in sorted(c.items()):
        print(f"{k[0]:<8}{k[1]:<18}{k[2]:<6}{n:>6}")
    for split in ("train", "val"):
        cc = Counter(r["label"] for r in rows if r["role"] == "train_val" and r["split"] == split)
        print(f"{split}: {sum(cc.values())} images ({cc['real']} real / {cc['fake']} fake)")
    print(f"heldout fake ({held_out}, never trained on): {sum(1 for r in rows if r['role']=='heldout_fake')}")
    print(f"heldout real (paired, GRAVEX test reencoded): {sum(1 for r in rows if r['role']=='heldout_real')}")
    print(f"orig_test sanity set (StyleGAN/FFHQ):         {sum(1 for r in rows if r['role']=='orig_test')}")
    print("integrity checks: OK")


def materialize(held_out, force=False):
    mpath = manifest_path(held_out)
    assert mpath.exists(), "run `plan` first"
    with open(mpath, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["role"] == "train_val"]
    if not force:
        for split in ("train", "val"):
            assert not (OUT_DIR / split).exists(), f"{OUT_DIR / split} exists; pass --force to overwrite"
    for r in rows:
        dest = OUT_DIR / r["dest_rel"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r["src_path"], dest)
    print(f"Copied {len(rows)} files -> {rel(OUT_DIR)}")
    for split in ("train", "val"):
        for label in ("fake", "real"):
            print(f"  {split}/{label}: {len(list((OUT_DIR / split / label).iterdir()))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "materialize"])
    ap.add_argument("--held-out", required=True, choices=VERSIONS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    plan(a.held_out) if a.stage == "plan" else materialize(a.held_out, a.force)
