from __future__ import annotations  # abilita annotazioni di tipo come "Path | None"

import dataclasses  # per creare la classe Config senza scrivere __init__ a mano
from pathlib import Path  # per gestire i percorsi dei file in modo portabile

import yaml  # per leggere l'eventuale file .yaml passato con --config


@dataclasses.dataclass  # genera automaticamente __init__/__repr__/__eq__ dai campi sotto
class Config:
    """Tutti gli iperparametri della pipeline in un unico posto.

    Ogni funzione del progetto (dataset, training, valutazione) riceve
    un'istanza di questa classe invece di leggere valori sparsi: così i
    parametri usati per allenare un modello sono sempre gli stessi usati
    per valutarlo.
    """

    image_size: int = 256          # risoluzione fissa: vedi il controllo in __post_init__
    latent_dim: int = 100          # dimensione del "collo di bottiglia" dell'autoencoder
    noise_std: float = 0.1         # deviazione standard del rumore gaussiano aggiunto in training
    use_denoising: bool = True     # se False (ablation), il training non aggiunge rumore
    loss_mode: str = "ssim_mse"    # "ssim_mse" (loss combinata) oppure "mse_only" (ablation)
    ssim_weight: float = 0.85      # peso della componente SSIM nella loss combinata
    mse_weight: float = 0.15       # peso della componente MSE nella loss combinata
    batch_size: int = 32           # quante immagini vengono processate insieme ad ogni passo
    learning_rate: float = 1e-3    # passo di aggiornamento dei pesi (ottimizzatore Adam)
    max_epochs: int = 200          # numero massimo di epoche di training
    early_stopping_patience: int = 20   # dopo quante epoche senza miglioramento ci si ferma
    lr_scheduler_patience: int = 10     # dopo quante epoche senza miglioramento si riduce il learning rate
    lr_scheduler_factor: float = 0.5    # di quanto si riduce il learning rate (es. dimezzato)
    val_split: float = 0.1         # frazione delle immagini "good" tenuta da parte per la validazione
    seed: int = 42                 # seed globale per rendere gli esperimenti riproducibili
    threshold_percentile: float = 95.0  # percentile usato per calcolare la soglia normale/anomalo

    def __post_init__(self) -> None:
        # Chiamato automaticamente dopo l'assegnazione dei campi (anche da dataclasses.replace).
        if self.image_size != 256:  # il modello (model.py) è costruito solo per 256x256
            raise ValueError(
                f"image_size deve essere 256, ricevuto {self.image_size}: l'architettura "
                "ConvAutoencoder è cablata per input 256x256 (il collo di bottiglia usa una "
                "convoluzione k8 s1 p0 sulla mappa di feature 8x8 prodotta da esattamente cinque "
                "downsampling stride-2 a partire da 256). Qualunque altra dimensione produrrebbe "
                "silenziosamente un collo di bottiglia più largo e privo di senso, invece di fallire."
            )


def load_config(path: Path | None) -> Config:
    """Costruisce una Config, opzionalmente sovrascrivendo alcuni campi da un file YAML."""
    if path is None:
        return Config()  # nessun file passato: tutti i valori restano quelli di default
    with open(path) as f:
        overrides = yaml.safe_load(f) or {}  # dict letto dal file (o {} se il file è vuoto)
    return dataclasses.replace(Config(), **overrides)  # copia Config() sostituendo solo questi campi
