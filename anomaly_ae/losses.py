from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(window_size: int, sigma: float) -> torch.Tensor:
    """Crea un vettore di pesi a forma di campana gaussiana, usato per costruire la finestra SSIM."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2  # es. [-5,-4,...,0,...,4,5]
    g = torch.exp(-(coords**2) / (2 * sigma**2))  # formula della gaussiana: più alto al centro
    return g / g.sum()  # normalizza: la somma dei pesi deve fare 1


def _create_window(window_size: int, channels: int) -> torch.Tensor:
    """Costruisce la finestra 2D (gaussiana) usata per calcolare medie/varianze locali della SSIM."""
    g1d = _gaussian_kernel_1d(window_size, sigma=1.5)
    window2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)  # prodotto esterno 1D x 1D -> gaussiana 2D
    return window2d.expand(channels, 1, window_size, window_size).contiguous()  # stessa finestra per ogni canale


def ssim_map(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, data_range: float = 1.0
) -> torch.Tensor:
    """Calcola la mappa di SSIM (Structural Similarity) pixel per pixel tra due immagini.

    Restituisce una mappa (B, 1, H, W) — un valore di similarità per ogni
    posizione, non un unico numero — perché questa stessa mappa serve sia
    come base della loss di training sia come mappa di anomalia in
    valutazione (vedi anomaly_map più sotto).
    """
    channels = img1.size(1)  # es. 3 per immagini RGB
    window = _create_window(window_size, channels).to(device=img1.device, dtype=img1.dtype)
    padding = window_size // 2  # mantiene la mappa risultante della stessa dimensione dell'input

    # groups=channels: ogni canale colore viene processato separatamente, senza mescolarsi.
    mu1 = F.conv2d(img1, window, padding=padding, groups=channels)  # media locale (pesata) di img1
    mu2 = F.conv2d(img2, window, padding=padding, groups=channels)  # media locale di img2
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channels) - mu1_sq  # varianza locale di img1
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channels) - mu2_sq  # varianza locale di img2
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu1_mu2   # covarianza locale tra le due

    c1 = (0.01 * data_range) ** 2  # costanti di stabilizzazione (standard nella formula SSIM):
    c2 = (0.03 * data_range) ** 2  # evitano divisioni per zero quando media/varianza sono piccole

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)  # confronta luminosità e struttura
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)  # confronta contrasto
    per_channel = numerator / denominator  # vicino a 1 dove simili, vicino a 0 dove diverse
    return per_channel.mean(dim=1, keepdim=True)  # media sui canali: (B,3,H,W) -> (B,1,H,W)


def anomaly_map(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Vicino a 0 dove la ricostruzione è fedele (nessuna anomalia), vicino a 1
    dove fallisce (probabile difetto mai visto in training)."""
    return 1.0 - ssim_map(reconstruction, target)


def combined_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mode: str,
    ssim_weight: float,
    mse_weight: float,
) -> torch.Tensor:
    """Loss di training: combina errore SSIM e MSE, oppure solo MSE (modalità ablation)."""
    if mode not in ("ssim_mse", "mse_only"):
        raise ValueError(f"Unknown loss mode: {mode!r}")
    mse = F.mse_loss(reconstruction, target)  # errore quadratico medio pixel per pixel
    if mode == "mse_only":
        return mse  # ablation: la componente SSIM non viene nemmeno calcolata
    ssim_loss = 1.0 - ssim_map(reconstruction, target).mean()  # 0 se simili, 1 se diverse
    return ssim_weight * ssim_loss + mse_weight * mse  # combinazione pesata (pesi da Config)
