import json
from pathlib import Path

from anomaly_ae.report import ablation_comparison_dataframe, load_metrics, metrics_to_dataframe


def _write_metrics(output_root: Path, category: str, image_auroc: float, pixel_auroc: float):
    category_dir = output_root / category
    category_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "category": category,
        "image_level_auroc": image_auroc,
        "pixel_level_auroc": pixel_auroc,
        "threshold": 0.5,
    }
    with open(category_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)


def test_load_metrics_reads_all_categories(tmp_path):
    _write_metrics(tmp_path, "bottle", 0.9, 0.8)
    _write_metrics(tmp_path, "screw", 0.7, 0.6)
    metrics = load_metrics(tmp_path, ["bottle", "screw"])
    assert metrics["bottle"]["image_level_auroc"] == 0.9
    assert metrics["screw"]["pixel_level_auroc"] == 0.6


def test_metrics_to_dataframe_indexed_by_category(tmp_path):
    _write_metrics(tmp_path, "bottle", 0.9, 0.8)
    metrics = load_metrics(tmp_path, ["bottle"])
    df = metrics_to_dataframe(metrics)
    assert df.index.name == "category"
    assert df.loc["bottle", "image_level_auroc"] == 0.9


def test_ablation_comparison_dataframe(tmp_path):
    main_root = tmp_path / "main"
    ablation_root = tmp_path / "ablation"
    _write_metrics(main_root, "bottle", 0.9, 0.8)
    _write_metrics(ablation_root, "bottle", 0.7, 0.6)

    main_metrics = load_metrics(main_root, ["bottle"])
    ablation_metrics = load_metrics(ablation_root, ["bottle"])

    df = ablation_comparison_dataframe(main_metrics, ablation_metrics, "mse_only")
    assert df.loc["bottle", "image_auroc_main"] == 0.9
    assert df.loc["bottle", "image_auroc_mse_only"] == 0.7
    assert df.loc["bottle", "pixel_auroc_main"] == 0.8
    assert df.loc["bottle", "pixel_auroc_mse_only"] == 0.6
