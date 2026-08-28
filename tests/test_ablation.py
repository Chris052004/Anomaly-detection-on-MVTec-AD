from pathlib import Path

import pytest

from anomaly_ae.ablation import (
    ABLATION_CATEGORIES,
    apply_ablation,
    output_dir_for_ablation,
    resolve_category,
)
from anomaly_ae.config import Config
from data_classes.mvtec_dataset import CATEGORIES


def test_ablation_categories_is_subset_of_all_categories():
    assert set(ABLATION_CATEGORIES).issubset(set(CATEGORIES))
    assert len(ABLATION_CATEGORIES) == 7


def test_apply_ablation_main_leaves_config_unchanged():
    cfg = apply_ablation(Config(), "main")
    assert cfg == Config()


def test_apply_ablation_mse_only_sets_loss_mode():
    cfg = apply_ablation(Config(), "mse_only")
    assert cfg.loss_mode == "mse_only"
    assert cfg.use_denoising is True  # campo non correlato, invariato


def test_apply_ablation_no_denoising_disables_noise():
    cfg = apply_ablation(Config(), "no_denoising")
    assert cfg.use_denoising is False
    assert cfg.loss_mode == "ssim_mse"  # campo non correlato, invariato


def test_apply_ablation_unknown_raises():
    with pytest.raises(ValueError):
        apply_ablation(Config(), "not_a_real_ablation")


def test_output_dir_for_ablation():
    root = Path("outputs")
    assert output_dir_for_ablation(root, "main") == root
    assert output_dir_for_ablation(root, "mse_only") == root / "ablation_mse_only"
    assert output_dir_for_ablation(root, "no_denoising") == root / "ablation_no_denoising"


def test_resolve_category_allowed():
    assert resolve_category("bottle", "mse_only") == "bottle"


def test_resolve_category_not_allowed_raises():
    with pytest.raises(ValueError):
        resolve_category("zipper", "mse_only")  # zipper non è in ABLATION_CATEGORIES


def test_resolve_category_missing_raises():
    with pytest.raises(ValueError):
        resolve_category(None, "main")
