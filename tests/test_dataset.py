from PIL import Image
from torchvision import datasets


def test_imagefolder_label_ordering(tmp_path):
    # The whole fake=0 / real=1 convention (used in train.py's pos_label
    # choices, gradcam.py's is_fake check, and api/main.py's sigmoid
    # interpretation) depends on torchvision.datasets.ImageFolder assigning
    # class indices alphabetically. Guard that assumption directly.
    for cls in ["fake", "real"]:
        d = tmp_path / cls
        d.mkdir()
        Image.new("RGB", (4, 4), color=(0, 0, 0)).save(d / "sample.jpg")

    dataset = datasets.ImageFolder(root=str(tmp_path))

    assert dataset.classes == ["fake", "real"]
    assert dataset.class_to_idx == {"fake": 0, "real": 1}
