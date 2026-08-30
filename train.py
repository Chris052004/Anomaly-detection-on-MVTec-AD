from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anomaly_ae.ablation import apply_ablation, output_dir_for_ablation, resolve_category
from anomaly_ae.config import Config, load_config
from anomaly_ae.training import train_one_category


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Allena l'autoencoder su UNA categoria (per tutte insieme vedi run_all.sh).

    Es. `python train.py --category bottle --ablation mse_only`
    """
    parser = argparse.ArgumentParser(description="Allena l'autoencoder di anomaly detection su MVTec AD.")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--ablation", choices=["main", "mse_only", "no_denoising"], default="main")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    category = resolve_category(args.category, args.ablation)
    base_config = load_config(args.config) if args.config else Config()
    config = apply_ablation(base_config, args.ablation)
    output_dir = output_dir_for_ablation(args.output_root, args.ablation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU se disponibile, altrimenti CPU

    print(f"Allenamento di {category} (ablation={args.ablation}) su {device}...")
    train_one_category(category, args.data_root, output_dir, config, device)  # la logica vera è in anomaly_ae/training.py


if __name__ == "__main__":  # esegue main() solo se lanciato direttamente, non se importato
    main()
