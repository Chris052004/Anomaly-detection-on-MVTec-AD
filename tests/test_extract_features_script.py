from pathlib import Path

import pytest

import extract_features as extract_features_script


class _FakeExtractor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cpu"


def test_parse_args_defaults():
    args = extract_features_script.parse_args([])
    assert args.category is None
    assert args.model_name == "google/vit-base-patch16-224"
    assert args.data_root == Path("data/mvtec_ad")
    assert args.output_root == Path("outputs")


def test_parse_args_rejects_unknown_category():
    with pytest.raises(SystemExit):
        extract_features_script.parse_args(["--category", "not_a_real_category"])


def test_main_requires_category(monkeypatch):
    monkeypatch.setattr(extract_features_script, "VisionEmbeddings", _FakeExtractor)
    monkeypatch.setattr(extract_features_script, "extract_category_features", lambda *a, **k: None)

    with pytest.raises(ValueError):
        extract_features_script.main([])


def test_main_extracts_single_category(monkeypatch, tmp_path):
    calls = []

    def fake_extract_category_features(category, data_root, output_dir, extractor):
        calls.append((category, data_root, output_dir, extractor.model_name))
        return output_dir / category / "vit_features"

    monkeypatch.setattr(extract_features_script, "VisionEmbeddings", _FakeExtractor)
    monkeypatch.setattr(extract_features_script, "extract_category_features", fake_extract_category_features)

    extract_features_script.main(["--category", "bottle", "--output-root", str(tmp_path)])

    assert len(calls) == 1
    category, data_root, output_dir, model_name = calls[0]
    assert category == "bottle"
    assert output_dir == tmp_path
    assert model_name == "google/vit-base-patch16-224"
