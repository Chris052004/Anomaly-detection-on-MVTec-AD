from __future__ import annotations

import argparse
from pathlib import Path

from anomaly_ae.features import extract_category_features
from data_classes.mvtec_dataset import CATEGORIES
from extract_representations.vision_embeddings import VisionEmbeddings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Estrae le feature ViT di UNA categoria (train "good" + test) e le mette in cache.

    Es. `python extract_features.py --category bottle`
    """
    parser = argparse.ArgumentParser(description="Estrae le feature ViT di una categoria MVTec AD.")
    parser.add_argument("--category", type=str, default=None, choices=CATEGORIES)
    parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.category is None:
        raise ValueError("--category e' obbligatorio")

    extractor = VisionEmbeddings(model_name=args.model_name)
    print(f"Estrazione feature ViT ({args.model_name}) per {args.category} su {extractor.device}...")
    features_dir = extract_category_features(args.category, args.data_root, args.output_root, extractor)
    print(f"Salvate in {features_dir}")


if __name__ == "__main__":  # esegue main() solo se lanciato direttamente, non se importato
    main()
