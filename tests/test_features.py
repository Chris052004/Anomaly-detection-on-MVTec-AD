from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from anomaly_ae.features import extract_category_features, load_category_features


def _write_dummy_image(path: Path, size=(32, 32), color=(120, 60, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _build_category(tmp_path: Path) -> Path:
    category_dir = tmp_path / "bottle"
    _write_dummy_image(category_dir / "train" / "good" / "000.png", color=(10, 10, 10))
    _write_dummy_image(category_dir / "train" / "good" / "001.png", color=(20, 20, 20))
    _write_dummy_image(category_dir / "test" / "good" / "000.png", color=(30, 30, 30))
    _write_dummy_image(category_dir / "test" / "broken_large" / "000.png", color=(40, 40, 40))
    return category_dir


class _FakeExtractor:
    """Sostituisce VisionEmbeddings nei test: nessun ViT reale, nessuna rete/GPU
    necessaria — restituisce un embedding deterministico basato sul colore medio
    dell'immagine, sufficiente per verificare la pipeline di estrazione/cache."""

    def extract(self, image):
        array = np.asarray(image, dtype=np.float32)
        return array.mean(axis=(0, 1)).reshape(1, -1)  # (1, 3): finto embedding a 3 numeri, come VisionEmbeddings.extract


def test_extract_category_features_writes_expected_shapes(tmp_path):
    data_root = tmp_path / "data"
    _build_category(data_root)
    output_dir = tmp_path / "outputs"

    features_dir = extract_category_features("bottle", data_root, output_dir, extractor=_FakeExtractor())

    assert features_dir == output_dir / "bottle" / "vit_features"
    assert (features_dir / "train.csv").exists()
    assert (features_dir / "test.csv").exists()

    train_features = pd.read_csv(features_dir / "train.csv")
    test_df = pd.read_csv(features_dir / "test.csv")

    assert train_features.shape == (2, 3)  # 2 immagini di training "good"
    assert list(train_features.columns) == ["f0", "f1", "f2"]
    assert test_df.shape == (2, 4)  # 2 immagini di test (1 good + 1 difettosa) + colonna label
    assert list(test_df.columns) == ["label", "f0", "f1", "f2"]
    assert sorted(test_df["label"].tolist()) == [0, 1]  # una "good" (0) e una "broken_large" (1)


def test_load_category_features_matches_extraction(tmp_path):
    data_root = tmp_path / "data"
    _build_category(data_root)
    output_dir = tmp_path / "outputs"
    extract_category_features("bottle", data_root, output_dir, extractor=_FakeExtractor())

    X_train, X_test, y_test = load_category_features(output_dir, "bottle")

    assert X_train.shape == (2, 3)
    assert X_test.shape == (2, 3)
    assert y_test.shape == (2,)
