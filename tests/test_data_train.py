from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_classes.mvtec_dataset import (
    MVTecTrainDataset,
    list_train_good_paths,
    load_image,
    split_train_val,
)


def _write_dummy_image(path: Path, size=(64, 64), color=(120, 60, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_load_image_shape_and_range(tmp_path):
    img_path = tmp_path / "img.png"
    _write_dummy_image(img_path)
    tensor = load_image(img_path, image_size=256)
    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_list_train_good_paths(tmp_path):
    category_dir = tmp_path / "bottle"
    for name in ["000.png", "001.png", "002.png"]:
        _write_dummy_image(category_dir / "train" / "good" / name)
    paths = list_train_good_paths(category_dir)
    assert len(paths) == 3
    assert all(p.suffix == ".png" for p in paths)


def test_split_train_val_is_deterministic_and_covers_all_paths(tmp_path):
    paths = [tmp_path / f"{i}.png" for i in range(10)]
    train_a, val_a = split_train_val(paths, val_split=0.2, seed=42)
    train_b, val_b = split_train_val(paths, val_split=0.2, seed=42)
    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 2
    assert len(train_a) == 8
    assert set(train_a) | set(val_a) == set(paths)
    assert set(train_a) & set(val_a) == set()


def test_train_dataset_with_noise_differs_from_clean(tmp_path):
    img_path = tmp_path / "img.png"
    _write_dummy_image(img_path)
    ds = MVTecTrainDataset([img_path], image_size=256, add_noise=True, noise_std=0.5)
    noisy, clean = ds[0]
    assert noisy.shape == (3, 256, 256)
    assert clean.shape == (3, 256, 256)
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert not torch.allclose(noisy, clean)


def test_train_dataset_without_noise_matches_clean(tmp_path):
    img_path = tmp_path / "img.png"
    _write_dummy_image(img_path)
    ds = MVTecTrainDataset([img_path], image_size=256, add_noise=False, noise_std=0.5)
    noisy, clean = ds[0]
    assert torch.equal(noisy, clean)


def test_train_dataset_len(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"{i}.png"
        _write_dummy_image(p)
        paths.append(p)
    ds = MVTecTrainDataset(paths, image_size=256, add_noise=False, noise_std=0.1)
    assert len(ds) == 4
