"""Evaluate a checkpoint on the GRAVEX split manifest, with the re-encoding control.

    python -m src.eval_gravex --weights model_v1_140k.pt --tag baseline_v1_140k

Convention (same as train.py / ImageFolder): fake=0, real=1, model outputs P(real); P(real) > 0.5 -> "real".
FPR below = fraction of REAL images classified fake.

Groups reported:
  orig_test         data/processed test set (StyleGAN vs FFHQ), sanity check
  sd_test           Stable Diffusion fakes from the held-out test split
  ffhq_reenc_test   GRAVEX re-encoded FFHQ reals (same encoding pipeline as the SD fakes)
  ffhq_orig_control the SAME 3,000 real images in their original 140k encoding (paired control)
  probe_*           FF++/DFDC/Celeb-DF/ai_detect, never trained on
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.dataset import get_transforms
from src.model import build_model

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/processed_v2/manifest.csv"


class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths, self.tf = paths, get_transforms("val")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.array(Image.open(ROOT / self.paths[i]).convert("RGB"))
        return self.tf(image=img)["image"]


def score(model, paths, bs):
    """Return raw logits (P(real) = sigmoid(logit)) for each unique path."""
    uniq = sorted(set(paths))
    out = {}
    with torch.no_grad():
        for k, batch in enumerate(DataLoader(PathDataset(uniq), batch_size=bs, num_workers=0)):
            logits = model(batch).squeeze(1).numpy()
            for p, l in zip(uniq[k * bs:(k + 1) * bs], logits):
                out[p] = float(l)
            if k % 20 == 0:
                print(f"  scored {min((k + 1) * bs, len(uniq))}/{len(uniq)}", flush=True)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate(flags):
    k, n = int(sum(flags)), len(flags)
    lo, hi = wilson(k, n)
    return dict(k=k, n=n, rate=k / n if n else float("nan"), ci95=[lo, hi])


def auc(fake_logits, real_logits):
    y = [0] * len(fake_logits) + [1] * len(real_logits)
    return float(roc_auc_score(y, list(fake_logits) + list(real_logits)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(MANIFEST, newline="")))
    sel = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]
    orig_test = [r for r in rows if r["split"] == "test" and r["source"] in ("stylegan_140k", "ffhq_140k")]
    sd_test = sel(split="test", source="sd")
    reenc = sel(split="test", source="ffhq_reencoded")
    probes = [r for r in rows if r["split"].startswith("probe_")]

    paths = ([r["src_path"] for r in orig_test + sd_test + reenc + probes] + [r["orig_path"] for r in reenc])
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(ROOT / a.weights, map_location="cpu"))
    model.eval()
    print(f"Scoring {len(set(paths))} images with {a.weights} ...", flush=True)
    L = score(model, paths, a.batch_size)

    res = {"weights": a.weights, "tag": a.tag}
    fake_flag = lambda l: l <= 0            # logit <= 0  <=>  P(real) <= 0.5  -> classified fake

    # sanity: original test set
    of = [L[r["src_path"]] for r in orig_test if r["label"] == "fake"]
    orl = [L[r["src_path"]] for r in orig_test if r["label"] == "real"]
    res["orig_test"] = dict(fake_recall=rate([fake_flag(l) for l in of]),
                            real_fpr=rate([fake_flag(l) for l in orl]), auc=auc(of, orl))

    # SD fakes
    sd_l = [L[r["src_path"]] for r in sd_test]
    res["sd_fake_recall"] = {"all": rate([fake_flag(l) for l in sd_l])}
    for v in sorted({r["version"] for r in sd_test}):
        res["sd_fake_recall"][v] = rate([fake_flag(L[r["src_path"]]) for r in sd_test if r["version"] == v])
    for g in ("woman", "man"):
        res["sd_fake_recall"][g] = rate([fake_flag(L[r["src_path"]]) for r in sd_test if r["gender"] == g])

    # the re-encoding control: same reals, two encodings
    re_l = np.array([L[r["src_path"]] for r in reenc])
    or_l = np.array([L[r["orig_path"]] for r in reenc])
    res["real_fpr"] = {"reencoded": rate(list(re_l <= 0)), "original_encoding": rate(list(or_l <= 0))}
    res["paired_reals"] = dict(
        n=len(reenc),
        fake_only_when_reencoded=int(((re_l <= 0) & (or_l > 0)).sum()),
        fake_only_when_original=int(((re_l > 0) & (or_l <= 0)).sum()),
        mean_logit_reencoded=float(re_l.mean()), mean_logit_original=float(or_l.mean()),
        mean_logit_shift_reenc_minus_orig=float((re_l - or_l).mean()),
        median_logit_shift=float(np.median(re_l - or_l)),
    )

    # separability, threshold-free
    res["auc_sd_vs_reals"] = {
        "same_encoding(sd vs reencoded_reals)": auc(sd_l, re_l),
        "cross_encoding(sd vs original_reals)": auc(sd_l, or_l),
    }
    for v in sorted({r["version"] for r in sd_test}):
        vl = [L[r["src_path"]] for r in sd_test if r["version"] == v]
        res["auc_sd_vs_reals"][f"{v}_same_encoding"] = auc(vl, re_l)

    # probes
    res["probes"] = {}
    for src in ("ffpp", "dfdc", "celebdf", "ai_detect"):
        f = [L[r["src_path"]] for r in probes if r["source"] == src and r["label"] == "fake"]
        rl = [L[r["src_path"]] for r in probes if r["source"] == src and r["label"] == "real"]
        res["probes"][src] = dict(fake_recall=rate([fake_flag(l) for l in f]),
                                  real_fpr=rate([fake_flag(l) for l in rl]), auc=auc(f, rl))

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"eval_{a.tag}.json").write_text(json.dumps(res, indent=2))
    with open(out_dir / f"eval_{a.tag}_logits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "logit"])
        w.writerows(sorted(L.items()))

    pct = lambda d: f"{100 * d['rate']:5.1f}% [{100 * d['ci95'][0]:.1f}-{100 * d['ci95'][1]:.1f}]  (n={d['n']})"
    print(f"\n=== {a.tag} ({a.weights}) ===")
    print(f"orig test      fake recall {pct(res['orig_test']['fake_recall'])} | real FPR {pct(res['orig_test']['real_fpr'])} | AUC {res['orig_test']['auc']:.4f}")
    print("SD fake recall (fakes classified fake):")
    for k, v in res["sd_fake_recall"].items():
        print(f"  {k:<7} {pct(v)}")
    print("Real FPR (reals classified fake):")
    print(f"  re-encoded FFHQ     {pct(res['real_fpr']['reencoded'])}")
    print(f"  original encoding   {pct(res['real_fpr']['original_encoding'])}   <- same 3,000 images")
    print("Paired:", json.dumps(res["paired_reals"]))
    print("AUC SD vs reals:", json.dumps(res["auc_sd_vs_reals"]))
    print("Probes:")
    for k, v in res["probes"].items():
        print(f"  {k:<10} fake recall {pct(v['fake_recall'])} | real FPR {pct(v['real_fpr'])} | AUC {v['auc']:.3f}")


if __name__ == "__main__":
    main()
