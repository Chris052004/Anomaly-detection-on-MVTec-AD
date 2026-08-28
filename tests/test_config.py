from pathlib import Path
import tempfile

import pytest

from anomaly_ae.config import Config, load_config


def test_config_defaults_match_spec():
    cfg = Config()
    assert cfg.image_size == 256
    assert cfg.latent_dim == 100
    assert cfg.noise_std == 0.1
    assert cfg.use_denoising is True
    assert cfg.loss_mode == "ssim_mse"
    assert cfg.ssim_weight == 0.85
    assert cfg.mse_weight == 0.15
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 1e-3
    assert cfg.max_epochs == 200
    assert cfg.early_stopping_patience == 20
    assert cfg.lr_scheduler_patience == 10
    assert cfg.lr_scheduler_factor == 0.5
    assert cfg.val_split == 0.1
    assert cfg.seed == 42
    assert cfg.threshold_percentile == 95.0


def test_load_config_returns_defaults_when_path_is_none():
    assert load_config(None) == Config()


def test_load_config_overrides_only_specified_fields():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "custom.yaml"
        yaml_path.write_text("batch_size: 8\nmax_epochs: 5\n")
        cfg = load_config(yaml_path)
        assert cfg.batch_size == 8
        assert cfg.max_epochs == 5
        assert cfg.image_size == 256  # untouched default


def test_config_rejects_non_256_image_size():
    with pytest.raises(ValueError, match="image_size deve essere 256"):
        Config(image_size=512)


def test_load_config_rejects_non_256_image_size():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "bad.yaml"
        yaml_path.write_text("image_size: 320\n")
        with pytest.raises(ValueError, match="image_size deve essere 256"):
            load_config(yaml_path)
