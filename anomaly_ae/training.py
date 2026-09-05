from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_ae.config import Config
from anomaly_ae.losses import combined_loss
from data_classes.mvtec_dataset import MVTecTrainDataset, list_train_good_paths, split_train_val
from model_classes.autoencoder_model import ConvAutoencoder


def set_seed(seed: int) -> None:
    """Fissa il seed di tutti i generatori casuali usati dal progetto, per la riproducibilità."""
    random.seed(seed)          # modulo random di Python (usato altrove nel progetto)
    np.random.seed(seed)       # NumPy
    torch.manual_seed(seed)    # PyTorch su CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch su GPU (se disponibile)


def build_dataloaders(category_dir: Path, config: Config) -> tuple[DataLoader, DataLoader]:
    """Prepara i DataLoader di training e validazione per una categoria."""
    paths = list_train_good_paths(category_dir)
    train_paths, val_paths = split_train_val(paths, config.val_split, config.seed)
    train_ds = MVTecTrainDataset(train_paths, config.image_size, config.use_denoising, config.noise_std)  # rumore secondo config
    # Il rumore è un'augmentation SOLO di training (spec, "Loss e rumore"): applicarla anche in
    # validazione renderebbe val_loss stocastica, alterando scheduler/early stopping/checkpoint e
    # rendendo le due varianti di ablation non confrontabili. Per questo qui è fisso a False.
    val_ds = MVTecTrainDataset(val_paths, config.image_size, False, config.noise_std)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)  # shuffle: mescola ad ogni epoca
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def _run_epoch(
    model: ConvAutoencoder,
    loader: DataLoader,
    config: Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Esegue un'epoca su tutti i batch del loader e restituisce la loss media.
    Se optimizer è None è una semplice valutazione (nessun aggiornamento dei pesi)."""
    total_loss = 0.0
    n_batches = 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)  # sposta i tensori su GPU/CPU
        reconstruction = model(noisy)  # forward pass: il modello prova a ricostruire l'immagine pulita
        loss = combined_loss(reconstruction, clean, config.loss_mode, config.ssim_weight, config.mse_weight)
        if optimizer is not None:
            optimizer.zero_grad()  # azzera i gradienti del passo precedente
            loss.backward()        # calcola i gradienti della loss rispetto ai pesi
            optimizer.step()       # aggiorna i pesi usando i gradienti appena calcolati
        total_loss += loss.item()  # .item(): dal tensore-scalare al numero Python
        n_batches += 1
    return total_loss / max(n_batches, 1)  # max(...,1) evita una divisione per zero


def _is_improvement(val_loss: float, best_val_loss: float, min_delta: float) -> bool:
    """Un calo di val_loss conta come miglioramento solo se supera min_delta: altrimenti
    oscillazioni numeriche minuscole resetterebbero il contatore dell'early stopping
    all'infinito, impedendogli di fermarsi anche quando la loss si e' di fatto stabilizzata."""
    return val_loss < best_val_loss - min_delta


def train_one_category(
    category: str,
    data_root: Path,
    output_dir: Path,
    config: Config,
    device: torch.device,
) -> Path:
    """Allena un autoencoder per UNA categoria e restituisce il percorso del checkpoint salvato."""
    set_seed(config.seed)  # riproducibilità: sempre PRIMA di creare dataset/modello
    category_dir = data_root / category
    train_loader, val_loader = build_dataloaders(category_dir, config)

    model = ConvAutoencoder(config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # decide come aggiornare i pesi
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # riduce il learning rate se val_loss ristagna
        optimizer, factor=config.lr_scheduler_factor, patience=config.lr_scheduler_patience
    )

    category_output = output_dir / category
    category_output.mkdir(parents=True, exist_ok=True)  # crea la cartella se non esiste già
    model_path = category_output / "model.pt"
    history_path = category_output / "history.csv"

    best_val_loss = float("inf")       # nessun modello ancora salvato
    epochs_without_improvement = 0     # contatore per l'early stopping
    history_rows = []                  # train_loss/val_loss di ogni epoca

    for epoch in range(config.max_epochs):
        model.train()  # modalità training (es. BatchNorm si comporta diversamente)
        train_loss = _run_epoch(model, train_loader, config, device, optimizer)
        model.eval()   # modalità valutazione (nessun aggiornamento dei pesi)
        with torch.no_grad():  # disabilita il calcolo dei gradienti: più veloce, meno memoria
            val_loss = _run_epoch(model, val_loader, config, device, optimizer=None)
        scheduler.step(val_loss)  # informa lo scheduler della loss di validazione di questa epoca
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        pd.DataFrame(history_rows).to_csv(history_path, index=False)  # riscritto ogni epoca: niente perso se si interrompe
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if _is_improvement(val_loss, best_val_loss, config.early_stopping_min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)  # salva il modello migliore visto finora
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break  # troppe epoche senza miglioramenti: ci fermiamo prima di max_epochs

    if not model_path.exists():  # caso limite (es. max_epochs=0): salva comunque lo stato attuale
        torch.save(model.state_dict(), model_path)
    if not history_path.exists():
        pd.DataFrame(history_rows, columns=["epoch", "train_loss", "val_loss"]).to_csv(
            history_path, index=False
        )

    return model_path
