from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend senza schermo/finestra: va impostato PRIMA di importare pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch

from anomaly_ae.config import Config
from anomaly_ae.losses import anomaly_map
from anomaly_ae.metrics import compute_image_level_auroc, compute_pixel_level_auroc, compute_threshold
from data_classes.mvtec_dataset import MVTecTestDataset, gather_test_samples, list_train_good_paths, load_image
from model_classes.autoencoder_model import ConvAutoencoder


def _save_sample_grids(
    model: ConvAutoencoder,
    test_ds: MVTecTestDataset,
    output_dir: Path,
    device: torch.device,
    n_samples: int,
) -> list[Path]:
    """Salva alcune immagini di esempio per l'ispezione visiva: confronto
    affiancato tra originale, ricostruzione, mappa di anomalia e maschera reale."""
    n = min(n_samples, len(test_ds))  # non si può chiedere più esempi di quanti test ce ne siano
    if n == 0:
        return []
    step = max(len(test_ds) // n, 1)  # indici distribuiti uniformemente, non solo i primi n
    indices = list(range(0, len(test_ds), step))[:n]

    saved_paths = []
    for idx in indices:
        image, _label, mask = test_ds[idx]
        image_batch = image.unsqueeze(0).to(device)  # aggiunge la dimensione di "batch" (1 immagine)
        with torch.no_grad():
            reconstruction = model(image_batch)
            amap = anomaly_map(reconstruction, image_batch)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))  # 4 pannelli affiancati
        axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())  # (canali,H,W) -> (H,W,canali) per matplotlib
        axes[0].set_title("original")
        axes[1].imshow(reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axes[1].set_title("reconstruction")
        axes[2].imshow(amap.squeeze(0).squeeze(0).cpu().numpy(), cmap="hot")  # più bianco = più anomalo
        axes[2].set_title("anomaly map")
        axes[3].imshow(mask.squeeze(0).cpu().numpy(), cmap="gray")
        axes[3].set_title("ground truth")
        for ax in axes:
            ax.axis("off")  # nasconde gli assi numerici, non servono per un'ispezione visiva

        path = output_dir / f"sample_{idx}.png"
        fig.savefig(path)
        plt.close(fig)  # libera la memoria della figura
        saved_paths.append(path)
    return saved_paths


def evaluate_category(
    category: str,
    data_root: Path,
    output_dir: Path,
    config: Config,
    device: torch.device,
    n_sample_grids: int = 8,
) -> dict:
    """Valuta il modello già allenato per una categoria: calcola le metriche
    (AUROC image/pixel-level, soglia) e salva sia metrics.json sia alcune immagini di esempio."""
    category_dir = data_root / category
    category_output = output_dir / category

    model = ConvAutoencoder(config.latent_dim).to(device)
    state_dict = torch.load(category_output / "model.pt", map_location=device, weights_only=True)  # weights_only=True: sicurezza
    model.load_state_dict(state_dict)
    model.eval()  # modalità valutazione: nessun aggiornamento dei pesi

    # Passo 1: score di anomalia sulle immagini di TRAINING (tutte "good"), per definire "normale".
    train_scores = []
    with torch.no_grad():
        for path in list_train_good_paths(category_dir):
            image = load_image(path, config.image_size).unsqueeze(0).to(device)
            reconstruction = model(image)
            train_scores.append(anomaly_map(reconstruction, image).max().item())  # pixel più anomalo = score immagine
    threshold = compute_threshold(train_scores, config.threshold_percentile)  # soglia calcolata SOLO sul training

    # Passo 2: valutazione su TUTTE le immagini di test (good + difettose).
    test_samples = gather_test_samples(category_dir)
    test_ds = MVTecTestDataset(test_samples, config.image_size)

    scores, labels, maps, masks = [], [], [], []
    with torch.no_grad():
        for image, label, mask in test_ds:
            image_batch = image.unsqueeze(0).to(device)
            reconstruction = model(image_batch)
            amap = anomaly_map(reconstruction, image_batch).squeeze(0).cpu().numpy()
            scores.append(float(amap.max()))  # score complessivo dell'immagine
            labels.append(label)              # etichetta vera: 0=good, 1=difettosa
            maps.append(amap)                 # mappa completa (per l'AUROC pixel-level)
            masks.append(mask.numpy())        # maschera reale del difetto

    metrics = {
        "category": category,
        "image_level_auroc": compute_image_level_auroc(scores, labels),  # distingue "con difetto" da "normale"
        "pixel_level_auroc": compute_pixel_level_auroc(np.stack(maps), np.stack(masks)),  # localizza il difetto
        "threshold": threshold,
        "image_scores": scores,   # score per-immagine: permettono una decisione concreta (score > soglia)
        "image_labels": labels,   # etichette per-immagine, per confrontarle con la decisione
    }

    category_output.mkdir(parents=True, exist_ok=True)
    with open(category_output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _save_sample_grids(model, test_ds, category_output, device, n_sample_grids)
    return metrics
