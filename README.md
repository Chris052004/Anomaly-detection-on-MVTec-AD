# MVTec AD Anomaly Detection

Autoencoder convoluzionale denoising (loss SSIM + MSE) allenato per ciascuna
categoria di prodotto del dataset MVTec AD, per il rilevamento di anomalie a
livello di immagine e di pixel. Vedi `docs/design.md` per il design completo.

## Setup

La wheel di `torch` di default su PyPI per Windows è solo CPU. Installa
prima la build CUDA (corrisponde a `torch==2.13.0`, versione fissata), poi
il resto delle dipendenze — pip vedrà `torch` già soddisfatto e lo salterà:

    .venv/Scripts/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
    .venv/Scripts/python -m pip install -r requirements.txt

(oppure esegui direttamente `bash prepare.sh`, che fa entrambi i passi).

Verifica che la GPU sia rilevata:

    .venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"

## 1. Estrai il dataset

    .venv/Scripts/python scripts/extract_dataset.py

Estrae `mvtec_anomaly_detection.tar.xz` in `data/mvtec_ad/`.

## 2. Allena

`train.py` allena una categoria alla volta:

    .venv/Scripts/python train.py --category bottle

Ablation (ristrette a carpet, leather, bottle, screw, transistor):

    .venv/Scripts/python train.py --category bottle --ablation mse_only
    .venv/Scripts/python train.py --category bottle --ablation no_denoising

Per allenare tutte le categorie in sequenza, vedi `run_all.sh`.

## 3. Valuta

    .venv/Scripts/python test.py --category bottle

Scrive `outputs/<categoria>/metrics.json` (e le varianti delle ablation in
`outputs/ablation_mse_only/`, `outputs/ablation_no_denoising/`), oltre alle
immagini di esempio per l'ispezione visiva.

## 4. Report

    .venv/Scripts/python -m jupyter notebook notebook/report.ipynb

Legge i file `metrics.json` prodotti sopra — esegui prima i passi 2–3 per il
modello principale ed entrambe le ablation.

## Test

    .venv/Scripts/python -m pytest
