from pathlib import Path

import pytest

import train as train_script


def test_parse_args_defaults():
    args = train_script.parse_args([])
    assert args.category is None
    assert args.ablation == "main"
    assert args.config is None
    assert args.data_root == Path("data/mvtec_ad")
    assert args.output_root == Path("outputs")


def test_main_requires_category(monkeypatch):
    monkeypatch.setattr(train_script, "train_one_category", lambda *a, **k: None)

    with pytest.raises(ValueError):
        train_script.main([])


def test_main_trains_single_category(monkeypatch):
    calls = []

    def fake_train_one_category(category, data_root, output_dir, config, device):
        calls.append((category, data_root, output_dir, config))
        return output_dir / category / "model.pt"

    monkeypatch.setattr(train_script, "train_one_category", fake_train_one_category)

    train_script.main(["--category", "bottle"])

    assert len(calls) == 1
    assert calls[0][0] == "bottle"


def test_main_train_ablation_applies_override_and_output_dir(monkeypatch):
    calls = []

    def fake_train_one_category(category, data_root, output_dir, config, device):
        calls.append((category, config.loss_mode, output_dir))
        return output_dir / category / "model.pt"

    monkeypatch.setattr(train_script, "train_one_category", fake_train_one_category)

    train_script.main(["--category", "screw", "--ablation", "mse_only"])

    category, loss_mode, output_dir = calls[0]
    assert category == "screw"
    assert loss_mode == "mse_only"
    assert str(output_dir).endswith("ablation_mse_only")


def test_main_train_category_not_in_ablation_raises(monkeypatch):
    monkeypatch.setattr(train_script, "train_one_category", lambda *a, **k: None)

    with pytest.raises(ValueError):
        train_script.main(["--category", "zipper", "--ablation", "mse_only"])
