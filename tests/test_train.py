import csv
from pathlib import Path

import torch
from PIL import Image

from anomaly_ae.config import Config
from anomaly_ae.training import build_dataloaders, train_one_category


def _write_dummy_image(path: Path, size=(256, 256), color=(120, 60, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _build_category(tmp_path: Path, n_images: int = 6) -> Path:
    category_dir = tmp_path / "bottle"
    for i in range(n_images):
        color = (i * 10 % 255, i * 20 % 255, i * 30 % 255)
        _write_dummy_image(category_dir / "train" / "good" / f"{i:03d}.png", color=color)
    return category_dir


def test_train_one_category_produces_checkpoint_and_history(tmp_path):
    category_dir = _build_category(tmp_path)
    output_dir = tmp_path / "outputs"
    config = Config(batch_size=2, max_epochs=2, early_stopping_patience=10, val_split=0.34)

    model_path = train_one_category(
        "bottle", tmp_path, output_dir, config, device=torch.device("cpu")
    )

    assert model_path == output_dir / "bottle" / "model.pt"
    assert model_path.exists()

    history_path = output_dir / "bottle" / "history.csv"
    assert history_path.exists()
    with open(history_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert set(rows[0].keys()) == {"epoch", "train_loss", "val_loss"}


def test_build_dataloaders_never_adds_noise_to_validation(tmp_path):
    category_dir = _build_category(tmp_path, n_images=6)
    config = Config(batch_size=2, use_denoising=True, noise_std=0.1, val_split=0.34)

    train_loader, val_loader = build_dataloaders(category_dir, config)

    for noisy, clean in val_loader:
        assert torch.equal(noisy, clean)

    train_noisy, train_clean = next(iter(train_loader))
    assert not torch.equal(train_noisy, train_clean)


def test_train_one_category_checkpoint_is_loadable(tmp_path):
    from model_classes.autoencoder_model import ConvAutoencoder

    category_dir = _build_category(tmp_path)
    output_dir = tmp_path / "outputs"
    config = Config(batch_size=2, max_epochs=1, val_split=0.34)

    model_path = train_one_category(
        "bottle", tmp_path, output_dir, config, device=torch.device("cpu")
    )

    model = ConvAutoencoder(config.latent_dim)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)  # non deve sollevare eccezioni
