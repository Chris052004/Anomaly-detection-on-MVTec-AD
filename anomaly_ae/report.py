from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_metrics(output_root: Path, categories: list[str]) -> dict[str, dict]:
    """Legge i file metrics.json di più categorie in un unico dizionario
    {nome_categoria: contenuto_del_json}, usato dal notebook di report."""
    metrics = {}
    for category in categories:
        with open(output_root / category / "metrics.json") as f:
            metrics[category] = json.load(f)
    return metrics


def metrics_to_dataframe(metrics: dict[str, dict]) -> pd.DataFrame:
    """Trasforma il dizionario di metriche in una tabella pandas, una riga per categoria."""
    df = pd.DataFrame(list(metrics.values()))
    return df.set_index("category")  # così si può scrivere df.loc["bottle"]


def ablation_comparison_dataframe(
    main_metrics: dict[str, dict], ablation_metrics: dict[str, dict], ablation_label: str
) -> pd.DataFrame:
    """Tabella di confronto tra il modello principale e una ablation, categoria per categoria."""
    rows = []
    for category, main_row in main_metrics.items():
        ablation_row = ablation_metrics[category]
        rows.append(
            {
                "category": category,
                "image_auroc_main": main_row["image_level_auroc"],
                f"image_auroc_{ablation_label}": ablation_row["image_level_auroc"],  # nome dinamico, es. "..._mse_only"
                "pixel_auroc_main": main_row["pixel_level_auroc"],
                f"pixel_auroc_{ablation_label}": ablation_row["pixel_level_auroc"],
            }
        )
    return pd.DataFrame(rows).set_index("category")
