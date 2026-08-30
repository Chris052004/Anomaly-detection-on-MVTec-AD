from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anomaly_ae.ablation import apply_ablation, output_dir_for_ablation, resolve_category
from anomaly_ae.config import Config, load_config
from anomaly_ae.evaluation import evaluate_category


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Valuta l'autoencoder su UNA categoria (per tutte insieme vedi run_all.sh).

    Es. `python test.py --category bottle --ablation mse_only`
    """
    parser = argparse.ArgumentParser(description="Valuta l'autoencoder di anomaly detection su MVTec AD.")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--ablation", choices=["main", "mse_only", "no_denoising"], default="main")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser.parse_args(argv)


def print_results(category: str, metrics: dict) -> None:
    """Stampa le metriche della categoria appena valutata (scritte anche in metrics.json)."""
    print(f"\nRisultati test - {category}")
    print("-" * 35)
    print(f"  {'image_level_auroc':<20}: {metrics['image_level_auroc']:.4f}")
    print(f"  {'pixel_level_auroc':<20}: {metrics['pixel_level_auroc']:.4f}")
    print(f"  {'threshold':<20}: {metrics['threshold']:.4f}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    category = resolve_category(args.category, args.ablation)
    base_config = load_config(args.config) if args.config else Config()
    config = apply_ablation(base_config, args.ablation)
    output_dir = output_dir_for_ablation(args.output_root, args.ablation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Valutazione di {category} (ablation={args.ablation}) su {device}...")
    metrics = evaluate_category(category, args.data_root, output_dir, config, device)
    print_results(category, metrics)


if __name__ == "__main__":  # esegue main() solo se lanciato direttamente, non se importato
    main()
