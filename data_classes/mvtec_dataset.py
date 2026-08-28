from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image  # libreria per aprire/ridimensionare immagini
from torch.utils.data import Dataset  # classe base di PyTorch per i dataset

# Le 15 categorie di prodotti del dataset MVTec AD (ognuna in data/mvtec_ad/<categoria>/).
CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def load_image(path: Path, image_size: int) -> torch.Tensor:
    """Carica un'immagine da disco e la trasforma in un tensore (3, image_size, image_size)
    con valori float32 in [0, 1], pronto per il modello."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)  # forza 3 canali + risoluzione fissa
        array = np.asarray(img, dtype=np.float32) / 255.0  # da interi 0-255 a float in [0, 1]
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()  # (H,W,3) -> (3,H,W), come vuole PyTorch


def list_train_good_paths(category_dir: Path) -> list[Path]:
    """Elenca tutte le immagini di training (solo quelle SENZA difetti) di una categoria."""
    good_dir = category_dir / "train" / "good"
    return sorted(good_dir.glob("*.png"))  # sorted = ordine stabile tra esecuzioni diverse


def split_train_val(
    paths: list[Path], val_split: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Divide una lista di percorsi in training e validazione, in modo riproducibile."""
    shuffled = list(paths)  # copia, per non modificare la lista originale
    random.Random(seed).shuffle(shuffled)  # generatore isolato: a parità di seed, stesso mescolamento
    n_val = int(len(shuffled) * val_split)  # es. 10% delle immagini -> validazione
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]
    return train_paths, val_paths


class MVTecTrainDataset(Dataset):
    """Dataset di training: per ogni immagine "good" restituisce (versione rumorosa, versione pulita).

    L'autoencoder impara a ricostruire la versione pulita a partire da
    quella rumorosa (o da quella pulita stessa, se add_noise=False, come
    nell'ablation "no_denoising").
    """

    def __init__(
        self,
        image_paths: list[Path],
        image_size: int,
        add_noise: bool,
        noise_std: float,
    ):
        self.image_paths = image_paths
        self.image_size = image_size
        self.add_noise = add_noise
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.image_paths)  # richiesto da PyTorch: quante immagini contiene il dataset

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = load_image(self.image_paths[idx], self.image_size)  # richiesto da PyTorch: elemento idx-esimo
        if self.add_noise:
            noisy = (clean + torch.randn_like(clean) * self.noise_std).clamp(0.0, 1.0)  # rumore gaussiano, riportato in [0,1]
        else:
            noisy = clean.clone()  # copia indipendente, stesso valore di clean
        return noisy, clean


@dataclasses.dataclass  # raggruppa 3 informazioni collegate senza scrivere una classe a mano
class TestSample:
    image_path: Path
    label: int  # 0 = immagine normale ("good"), 1 = immagine con difetto
    mask_path: Path | None  # percorso della maschera del difetto, None se l'immagine è "good"


def gather_test_samples(category_dir: Path) -> list[TestSample]:
    """Costruisce l'elenco di tutte le immagini di test di una categoria, con etichetta e maschera.

    test/ contiene una sottocartella "good" e una per ogni tipo di difetto;
    la maschera di un'immagine difettosa sta in ground_truth/<tipo>/ con lo
    stesso nome file più il suffisso "_mask".
    """
    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"
    samples: list[TestSample] = []
    for subdir in sorted(p for p in test_dir.iterdir() if p.is_dir()):  # ordine alfabetico, solo cartelle
        is_good = subdir.name == "good"
        for image_path in sorted(subdir.glob("*.png")):
            if is_good:
                samples.append(TestSample(image_path=image_path, label=0, mask_path=None))  # nessuna maschera
            else:
                # es. test/broken_large/000.png -> ground_truth/broken_large/000_mask.png
                mask_path = ground_truth_dir / subdir.name / f"{image_path.stem}_mask.png"
                samples.append(TestSample(image_path=image_path, label=1, mask_path=mask_path))
    return samples


def load_mask(path: Path | None, image_size: int) -> torch.Tensor:
    """Carica una maschera di segmentazione del difetto come tensore binario (0/1)."""
    if path is None:
        return torch.zeros(1, image_size, image_size)  # "good": nessun pixel anomalo
    with Image.open(path) as img:
        img = img.convert("L").resize((image_size, image_size), Image.NEAREST)  # 1 canale, NEAREST = niente sfumature
        array = np.asarray(img, dtype=np.float32)
    binary = (array > 127).astype(np.float32)  # soglia a metà scala: ogni pixel diventa 0.0 o 1.0
    return torch.from_numpy(binary).unsqueeze(0).contiguous()  # (H,W) -> (1,H,W)


class MVTecTestDataset(Dataset):
    """Dataset di valutazione: per ogni immagine di test restituisce (immagine, etichetta, maschera)."""

    def __init__(self, samples: list[TestSample], image_size: int):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        sample = self.samples[idx]
        image = load_image(sample.image_path, self.image_size)
        mask = load_mask(sample.mask_path, self.image_size)
        return image, sample.label, mask
