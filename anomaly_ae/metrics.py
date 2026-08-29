from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_image_level_auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC a livello di immagine: 1.0 se lo score separa perfettamente le
    immagini normali (0) da quelle difettose (1), 0.5 se equivale al caso."""
    return float(roc_auc_score(labels, scores))  # float(...) per poter salvare il risultato in JSON


def compute_pixel_level_auroc(anomaly_maps: np.ndarray, masks: np.ndarray) -> float:
    """AUROC a livello di pixel: quanto bene la mappa di anomalia localizza
    esattamente DOVE si trova il difetto, confrontata con la maschera reale."""
    return float(roc_auc_score(masks.reshape(-1), anomaly_maps.reshape(-1)))  # appiattisce tutti i pixel in un vettore


def compute_threshold(train_scores: list[float], percentile: float) -> float:
    """Soglia normale/anomalo: percentile degli score sulle immagini di
    training (tutte "good" per definizione), es. 95 = sotto questa soglia
    cade il 95% degli score "normali"."""
    return float(np.percentile(train_scores, percentile))
