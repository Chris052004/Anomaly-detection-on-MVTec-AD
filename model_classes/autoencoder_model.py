from __future__ import annotations

import torch
from torch import nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    use_bn: bool = True,
    activation: str | None = "leaky_relu",
) -> nn.Sequential:
    """Un blocco base dell'encoder: convoluzione (+ eventuale normalizzazione e attivazione).

    Evita di ripetere lo stesso schema sei volte identico nell'encoder.
    """
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]  # riduce risoluzione, aumenta canali
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))  # stabilizza il training normalizzando per batch
    if activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))  # non linearità (altrimenti sarebbe tutto lineare)
    return nn.Sequential(*layers)  # incatena i layer: output dell'uno = input del successivo


def _deconv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    use_bn: bool = True,
    activation: str | None = "relu",
) -> nn.Sequential:
    """Un blocco base del decoder: l'operazione inversa della convoluzione (ri-espande la risoluzione)."""
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)  # opposto di Conv2d: più grande, meno canali
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "sigmoid":
        layers.append(nn.Sigmoid())  # solo sull'ultimo layer: riporta l'output in [0,1] come l'input
    return nn.Sequential(*layers)


class ConvAutoencoder(nn.Module):
    """Conv-autoencoder per denoising. Input/output sono fissi a 3x256x256 —
    il kernel_size=8 del collo di bottiglia riduce a 1x1 solo perché i cinque
    layer stride-2 dell'encoder portano prima 256 esattamente a 8."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()  # inizializza la parte "nn.Module" di PyTorch (obbligatorio)

        # ENCODER: comprime 256x256x3 fino a un vettore di latent_dim numeri.
        self.encoder = nn.Sequential(
            _conv_block(3, 32, 4, 2, 1),      # 256x256x3   -> 128x128x32
            _conv_block(32, 64, 4, 2, 1),     # 128x128x32  -> 64x64x64
            _conv_block(64, 128, 4, 2, 1),    # 64x64x64    -> 32x32x128
            _conv_block(128, 256, 4, 2, 1),   # 32x32x128   -> 16x16x256
            _conv_block(256, 512, 4, 2, 1),   # 16x16x256   -> 8x8x512
            _conv_block(512, latent_dim, 8, 1, 0, use_bn=False, activation=None),  # 8x8x512 -> 1x1xlatent_dim (bottleneck)
        )

        # DECODER: copia speculare dell'encoder, riespande fino a 256x256x3.
        self.decoder = nn.Sequential(
            _deconv_block(latent_dim, 512, 8, 1, 0),      # 1x1xlatent_dim -> 8x8x512
            _deconv_block(512, 256, 4, 2, 1),             # 8x8x512    -> 16x16x256
            _deconv_block(256, 128, 4, 2, 1),             # 16x16x256  -> 32x32x128
            _deconv_block(128, 64, 4, 2, 1),               # 32x32x128  -> 64x64x64
            _deconv_block(64, 32, 4, 2, 1),                # 64x64x64   -> 128x128x32
            _deconv_block(32, 3, 4, 2, 1, use_bn=False, activation="sigmoid"),  # 128x128x32 -> 256x256x3, valori in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))  # comprime, poi ricostruisce
