from pathlib import Path

import pytest

from src import preprocessing


def _make_raw_images(raw_dir, split, cls, count):
    d = raw_dir / split / cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (d / f"{cls}_{i}.jpg").write_bytes(b"fake image bytes")
    return d


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw" / "real_vs_fake" / "real_vs_fake"
    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(preprocessing, "RAW_DIR", raw_dir)
    monkeypatch.setattr(preprocessing, "PROCESSED_DIR", processed_dir)
    return raw_dir, processed_dir


def test_split_mapping_and_folder_creation(patched_dirs, monkeypatch):
    raw_dir, processed_dir = patched_dirs
    monkeypatch.setattr(preprocessing, "SAMPLE_SIZES", {"train": 3, "valid": 2, "test": 2})

    for raw_split in ["train", "valid", "test"]:
        for cls in ["real", "fake"]:
            _make_raw_images(raw_dir, raw_split, cls, 5)

    preprocessing.sample_and_copy()

    # raw/train -> processed/train, raw/valid -> processed/val, raw/test -> processed/test
    for proc_split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            assert (processed_dir / proc_split / cls).is_dir()

    # the raw split name "valid" must never leak through as a processed folder name
    assert not (processed_dir / "valid").exists()


def test_sample_counts_match_requested_sizes(patched_dirs, monkeypatch):
    raw_dir, processed_dir = patched_dirs
    monkeypatch.setattr(preprocessing, "SAMPLE_SIZES", {"train": 3, "valid": 2, "test": 1})

    for raw_split in ["train", "valid", "test"]:
        for cls in ["real", "fake"]:
            _make_raw_images(raw_dir, raw_split, cls, 10)

    preprocessing.sample_and_copy()

    assert len(list((processed_dir / "train" / "real").glob("*.jpg"))) == 3
    assert len(list((processed_dir / "train" / "fake").glob("*.jpg"))) == 3
    assert len(list((processed_dir / "val" / "real").glob("*.jpg"))) == 2
    assert len(list((processed_dir / "val" / "fake").glob("*.jpg"))) == 2
    assert len(list((processed_dir / "test" / "real").glob("*.jpg"))) == 1
    assert len(list((processed_dir / "test" / "fake").glob("*.jpg"))) == 1


def test_fallback_when_not_enough_source_files(patched_dirs, monkeypatch, capsys):
    raw_dir, processed_dir = patched_dirs
    # Request far more images than actually exist in the source folders
    monkeypatch.setattr(preprocessing, "SAMPLE_SIZES", {"train": 100, "valid": 100, "test": 100})

    for raw_split in ["train", "valid", "test"]:
        for cls in ["real", "fake"]:
            _make_raw_images(raw_dir, raw_split, cls, 4)

    preprocessing.sample_and_copy()

    # Should copy everything available instead of crashing or sampling with replacement
    assert len(list((processed_dir / "train" / "real").glob("*.jpg"))) == 4
    assert len(list((processed_dir / "train" / "fake").glob("*.jpg"))) == 4
    assert len(list((processed_dir / "val" / "real").glob("*.jpg"))) == 4
    assert len(list((processed_dir / "test" / "fake").glob("*.jpg"))) == 4

    out = capsys.readouterr().out
    assert "WARNING" in out
