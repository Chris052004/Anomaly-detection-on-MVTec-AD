from pathlib import Path

import pytest

import test as test_script


def test_parse_args_defaults():
    args = test_script.parse_args([])
    assert args.category is None
    assert args.ablation == "main"
    assert args.config is None
    assert args.data_root == Path("data/mvtec_ad")
    assert args.output_root == Path("outputs")


def test_main_requires_category(monkeypatch):
    monkeypatch.setattr(test_script, "evaluate_category", lambda *a, **k: {})

    with pytest.raises(ValueError):
        test_script.main([])


def test_main_evaluates_single_category(monkeypatch):
    calls = []

    def fake_evaluate_category(category, data_root, output_dir, config, device):
        calls.append(category)
        return {"category": category, "image_level_auroc": 0.9, "pixel_level_auroc": 0.8, "threshold": 0.5}

    monkeypatch.setattr(test_script, "evaluate_category", fake_evaluate_category)

    test_script.main(["--category", "bottle"])

    assert calls == ["bottle"]


def test_main_prints_results(monkeypatch, capsys):
    def fake_evaluate_category(category, data_root, output_dir, config, device):
        return {"category": category, "image_level_auroc": 0.9, "pixel_level_auroc": 0.8, "threshold": 0.5}

    monkeypatch.setattr(test_script, "evaluate_category", fake_evaluate_category)

    test_script.main(["--category", "bottle"])

    captured = capsys.readouterr()
    assert "bottle" in captured.out
    assert "image_level_auroc" in captured.out
    assert "0.9000" in captured.out


def test_print_results_shows_all_metrics(capsys):
    metrics = {"image_level_auroc": 0.9, "pixel_level_auroc": 0.8, "threshold": 0.5}
    test_script.print_results("bottle", metrics)
    captured = capsys.readouterr()
    assert "bottle" in captured.out
    assert "0.9000" in captured.out
    assert "0.8000" in captured.out
