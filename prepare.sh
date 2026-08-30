#!/usr/bin/env bash
set -e

# La wheel di torch di default su PyPI per Windows e' solo CPU: installa prima la
# build CUDA (versione fissata), poi il resto delle dipendenze - pip vedra' torch
# gia' soddisfatto e lo salta.
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Estrae il dataset se l'archivio e' presente e non ancora estratto (idempotente).
if [ -f "mvtec_anomaly_detection.tar.xz" ]; then
    python scripts/extract_dataset.py
else
    echo "mvtec_anomaly_detection.tar.xz non trovato: scaricalo da https://www.mvtec.com/company/research/datasets/mvtec-ad prima di allenare."
fi
