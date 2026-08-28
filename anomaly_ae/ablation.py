from __future__ import annotations

import dataclasses
from pathlib import Path

from anomaly_ae.config import Config
from data_classes.mvtec_dataset import CATEGORIES

# Le due ablation girano su queste 7 categorie: le 5 originali, scelte come mix
# rappresentativo di texture e oggetti rigidi, piu' grid e tile, aggiunte in seguito perche'
# collassano sotto ssim_mse (vedi Limiti noti in design.md) - un caso limite interessante da
# confrontare, tenuto separato dalle 5 originali per non falsare il campione rappresentativo.
ABLATION_CATEGORIES = ["carpet", "leather", "bottle", "screw", "transistor", "grid", "tile"]

_ABLATION_OVERRIDES = {
    "main": {},                                    # modello principale: nessun override
    "mse_only": {"loss_mode": "mse_only"},         # disattiva la componente SSIM della loss
    "no_denoising": {"use_denoising": False},      # disattiva il rumore in training
}

_OUTPUT_SUBDIRS = {
    "main": "",                            # scrive direttamente in outputs/<categoria>/
    "mse_only": "ablation_mse_only",       # scrive in outputs/ablation_mse_only/<categoria>/
    "no_denoising": "ablation_no_denoising",
}


def apply_ablation(config: Config, ablation: str) -> Config:
    """Restituisce una NUOVA Config con gli override della ablation richiesta."""
    if ablation not in _ABLATION_OVERRIDES:
        raise ValueError(f"Unknown ablation: {ablation!r}. Valid: {list(_ABLATION_OVERRIDES)}")
    return dataclasses.replace(config, **_ABLATION_OVERRIDES[ablation])  # copia cambiando solo questi campi


def output_dir_for_ablation(output_root: Path, ablation: str) -> Path:
    """Calcola la cartella di output corretta per una data ablation."""
    if ablation not in _OUTPUT_SUBDIRS:
        raise ValueError(f"Unknown ablation: {ablation!r}. Valid: {list(_OUTPUT_SUBDIRS)}")
    subdir = _OUTPUT_SUBDIRS[ablation]
    return output_root / subdir if subdir else output_root  # "" -> output_root invariato


def resolve_category(category: str | None, ablation: str) -> str:
    """Valida --category per la ablation scelta (train.py/test.py girano su una
    categoria alla volta; per tutte insieme vedi run_all.sh)."""
    allowed = CATEGORIES if ablation == "main" else ABLATION_CATEGORIES  # main: tutte e 15; ablation: solo 7
    if category is None:
        raise ValueError("--category e' obbligatorio (per tutte le categorie insieme usa run_all.sh)")
    if category not in allowed:
        raise ValueError(  # es. "zipper" non è tra le 7 categorie ammesse per le ablation
            f"Category {category!r} not valid for ablation {ablation!r}. Allowed: {allowed}"
        )
    return category
