import numpy as np

from anomaly_ae.metrics import (
    compute_image_level_auroc,
    compute_pixel_level_auroc,
    compute_threshold,
)


def test_image_level_auroc_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert compute_image_level_auroc(scores, labels) == 1.0


def test_image_level_auroc_worst_case():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [0, 0, 1, 1]
    assert compute_image_level_auroc(scores, labels) == 0.0


def test_pixel_level_auroc_perfect_separation():
    anomaly_maps = np.array([[[0.1, 0.9], [0.1, 0.9]]])  # shape (1, 2, 2)
    masks = np.array([[[0, 1], [0, 1]]])
    assert compute_pixel_level_auroc(anomaly_maps, masks) == 1.0


def test_compute_threshold_is_percentile():
    scores = list(range(1, 101))  # 1..100
    threshold = compute_threshold(scores, percentile=95.0)
    assert 94.0 < threshold < 96.0
