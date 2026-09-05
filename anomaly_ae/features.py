from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data_classes.mvtec_dataset import gather_test_samples, list_train_good_paths
from extract_representations.vision_embeddings import VisionEmbeddings


def _extract_paths(paths: list[Path], extractor: VisionEmbeddings) -> np.ndarray:
    """Estrae l'embedding ViT di ogni immagine in paths, una alla volta (l'estrazione
    non richiede batching: è solo un forward pass, senza training)."""
    embeddings = []
    for path in paths:
        with Image.open(path) as img:
            embeddings.append(extractor.extract(img.convert("RGB"))[0])  # [0]: toglie la dimensione di batch (1)
    return np.stack(embeddings)  # (n_immagini, 768)


def _feature_columns(n_features: int) -> list[str]:
    return [f"f{i}" for i in range(n_features)]


def extract_category_features(
    category: str,
    data_root: Path,
    output_dir: Path,
    extractor: VisionEmbeddings | None = None,
) -> Path:
    """Estrae e mette in cache su disco (CSV) le feature ViT delle immagini "good" di
    training e di tutte le immagini di test di una categoria — per riutilizzarle senza
    dover rifare l'estrazione (lenta, un forward pass del ViT per ogni immagine) ogni
    volta che si vuole provare un modello classico diverso sopra le stesse feature."""
    extractor = extractor or VisionEmbeddings()
    category_dir = data_root / category
    features_dir = output_dir / category / "vit_features"
    features_dir.mkdir(parents=True, exist_ok=True)

    train_paths = list_train_good_paths(category_dir)
    train_features = _extract_paths(train_paths, extractor)
    pd.DataFrame(train_features, columns=_feature_columns(train_features.shape[1])).to_csv(
        features_dir / "train.csv", index=False
    )

    test_samples = gather_test_samples(category_dir)
    test_features = _extract_paths([s.image_path for s in test_samples], extractor)
    test_labels = np.array([s.label for s in test_samples])
    test_df = pd.DataFrame(test_features, columns=_feature_columns(test_features.shape[1]))
    test_df.insert(0, "label", test_labels)  # prima colonna, per leggibilità (0=good, 1=difettoso)
    test_df.to_csv(features_dir / "test.csv", index=False)

    return features_dir


def load_category_features(output_dir: Path, category: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ricarica le feature già estratte e messe in cache da extract_category_features."""
    features_dir = output_dir / category / "vit_features"
    train_features = pd.read_csv(features_dir / "train.csv").to_numpy()
    test_df = pd.read_csv(features_dir / "test.csv")
    test_labels = test_df["label"].to_numpy()
    test_features = test_df.drop(columns=["label"]).to_numpy()
    return train_features, test_features, test_labels
