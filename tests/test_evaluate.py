import json
from pathlib import Path

import torch
from PIL import Image

from anomaly_ae.config import Config
from anomaly_ae.evaluation import evaluate_category
from anomaly_ae.training import train_one_category


def _write_dummy_image(path: Path, size=(256, 256), color=(120, 60, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_dummy_mask(path: Path, size=(256, 256), value=255):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, value).save(path)


def _build_category(tmp_path: Path) -> Path:
    category_dir = tmp_path / "bottle"
    for i in range(6):
        color = (i * 10 % 255, i * 20 % 255, i * 30 % 255)
        _write_dummy_image(category_dir / "train" / "good" / f"{i:03d}.png", color=color)
    _write_dummy_image(category_dir / "test" / "good" / "000.png", color=(50, 50, 50))
    _write_dummy_image(category_dir / "test" / "good" / "001.png", color=(60, 60, 60))
    _write_dummy_image(category_dir / "test" / "broken_large" / "000.png", color=(200, 0, 0))
    _write_dummy_mask(category_dir / "ground_truth" / "broken_large" / "000_mask.png")
    return category_dir


def test_evaluate_category_writes_metrics_and_samples(tmp_path):
    _build_category(tmp_path)
    output_dir = tmp_path / "outputs"
    config = Config(batch_size=2, max_epochs=1, val_split=0.34)
    device = torch.device("cpu")

    train_one_category("bottle", tmp_path, output_dir, config, device)
    metrics = evaluate_category("bottle", tmp_path, output_dir, config, device, n_sample_grids=2)

    assert metrics["category"] == "bottle"
    assert 0.0 <= metrics["image_level_auroc"] <= 1.0
    assert 0.0 <= metrics["pixel_level_auroc"] <= 1.0
    assert isinstance(metrics["threshold"], float)

    # score/etichette per-immagine sono salvati così la soglia può guidare una decisione concreta
    assert isinstance(metrics["image_scores"], list)
    assert isinstance(metrics["image_labels"], list)
    assert len(metrics["image_scores"]) == 3  # 2 good + 1 defective test images
    assert len(metrics["image_labels"]) == 3
    assert sorted(metrics["image_labels"]) == [0, 0, 1]

    metrics_path = output_dir / "bottle" / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        saved = json.load(f)
    assert saved == metrics

    sample_pngs = list((output_dir / "bottle").glob("sample_*.png"))
    assert len(sample_pngs) == 2
