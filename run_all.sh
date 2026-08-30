#!/usr/bin/env bash
# run_all.sh
# Allena e valuta il modello principale su tutte e 15 le categorie, poi le due
# ablation (loss mse_only, denoising disattivato) sulle 7 categorie di ablation.
# Ogni combinazione categoria/ablation scrive il proprio checkpoint separato
# (outputs/<categoria>/ oppure outputs/ablation_<nome>/<categoria>/), quindi i
# risultati non si sovrascrivono mai a vicenda.
#
# Uso:
#   bash run_all.sh                 # esegue tutto
#   bash run_all.sh --data-root X   # eventuali argomenti extra sono inoltrati
#                                      a ogni invocazione di train.py/test.py
# ----------------------------------------------------------------------------

EXTRA_ARGS="$@"
CFG="--config config/default.yaml"

MAIN_CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
ABLATION_CATEGORIES=(carpet leather bottle screw transistor grid tile)

echo "============================================================"
echo "  Modello principale (15 categorie)"
echo "============================================================"
for category in "${MAIN_CATEGORIES[@]}"; do
    echo "--- $category ---"
    python train.py $CFG --category "$category" $EXTRA_ARGS
    python test.py  $CFG --category "$category" $EXTRA_ARGS
done

for ablation in mse_only no_denoising; do
    echo ""
    echo "============================================================"
    echo "  Ablation: $ablation (7 categorie)"
    echo "============================================================"
    for category in "${ABLATION_CATEGORIES[@]}"; do
        echo "--- $category ---"
        python train.py $CFG --category "$category" --ablation "$ablation" $EXTRA_ARGS
        python test.py  $CFG --category "$category" --ablation "$ablation" $EXTRA_ARGS
    done
done

echo ""
echo "Tutti i run sono completati. Risultati in ./outputs/"
