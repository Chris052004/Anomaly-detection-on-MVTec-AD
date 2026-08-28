from pathlib import Path

import torch
from PIL import Image

from data_classes.mvtec_dataset import MVTecTestDataset, gather_test_samples, load_mask


def _write_dummy_image(path: Path, size=(64, 64), color=(120, 60, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_dummy_mask(path: Path, size=(64, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 255).save(path)


def _build_category(tmp_path: Path) -> Path:
    category_dir = tmp_path / "bottle"
    _write_dummy_image(category_dir / "test" / "good" / "000.png")
    _write_dummy_image(category_dir / "test" / "good" / "001.png")
    _write_dummy_image(category_dir / "test" / "broken_large" / "000.png")
    _write_dummy_mask(category_dir / "ground_truth" / "broken_large" / "000_mask.png")
    return category_dir


def test_gather_test_samples_labels_and_masks(tmp_path):
    category_dir = _build_category(tmp_path)
    samples = gather_test_samples(category_dir)
    assert len(samples) == 3

    good_samples = [s for s in samples if s.label == 0]
    defect_samples = [s for s in samples if s.label == 1]
    assert len(good_samples) == 2
    assert len(defect_samples) == 1
    assert all(s.mask_path is None for s in good_samples)
    assert defect_samples[0].mask_path is not None
    assert defect_samples[0].mask_path.name == "000_mask.png"


def test_load_mask_none_returns_zeros():
    mask = load_mask(None, image_size=256)
    assert mask.shape == (1, 256, 256)
    assert torch.equal(mask, torch.zeros(1, 256, 256))


def test_load_mask_thresholds_to_binary(tmp_path):
    mask_path = tmp_path / "000_mask.png"
    _write_dummy_mask(mask_path)
    mask = load_mask(mask_path, image_size=256)
    assert mask.shape == (1, 256, 256)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
    assert mask.max() == 1.0


def test_test_dataset_returns_image_label_mask(tmp_path):
    category_dir = _build_category(tmp_path)
    samples = gather_test_samples(category_dir)
    ds = MVTecTestDataset(samples, image_size=256)
    assert len(ds) == 3
    image, label, mask = ds[0]
    assert image.shape == (3, 256, 256)
    assert label in (0, 1)
    assert mask.shape == (1, 256, 256)
