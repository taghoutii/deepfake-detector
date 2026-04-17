import os
import shutil
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw/real_vs_fake/real_vs_fake")
PROCESSED_DIR = Path("data/processed")

SAMPLE_SIZES = {
    "train": 4000,   # per class → 8000 total
    "valid": 1000,   # per class → 2000 total
    "test":  1000,   # per class → 2000 total
}

# raw/ uses "valid", processed/ uses "val" — we rename for clarity
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
CLASSES   = ["real", "fake"]
SEED      = 42
# ────────────────────────────────────────────────────────────────────────────

def sample_and_copy():
    random.seed(SEED)

    for raw_split, proc_split in SPLIT_MAP.items():
        n = SAMPLE_SIZES[raw_split]

        for cls in CLASSES:
            src_dir  = RAW_DIR / raw_split / cls
            dest_dir = PROCESSED_DIR / proc_split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)

            all_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))

            if len(all_files) < n:
                print(f"WARNING: {src_dir} only has {len(all_files)} images, requested {n}. Using all.")
                selected = all_files
            else:
                selected = random.sample(all_files, n)

            for i, filepath in enumerate(selected):
                shutil.copy(filepath, dest_dir / filepath.name)

            print(f"Copied {len(selected)} images → {dest_dir}")

    print("\nDone. Processed dataset ready.")

if __name__ == "__main__":
    sample_and_copy()