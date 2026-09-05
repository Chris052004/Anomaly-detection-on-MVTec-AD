#!/usr/bin/env bash
# extract_all_features.sh
# Estrae le feature ViT (train "good" + test) di tutte e 15 le categorie MVTec AD.
# Ogni categoria scrive la propria cache separata (outputs/<categoria>/vit_features/),
# quindi i risultati non si sovrascrivono mai a vicenda.
#
# Uso:
#   bash extract_all_features.sh                    # esegue tutto
#   bash extract_all_features.sh --model-name X      # eventuali argomenti extra sono
#                                                        inoltrati ad ogni invocazione
# ----------------------------------------------------------------------------

EXTRA_ARGS="$@"

CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)

echo "============================================================"
echo "  Estrazione feature ViT (15 categorie)"
echo "============================================================"
for category in "${CATEGORIES[@]}"; do
    echo "--- $category ---"
    python extract_features.py --category "$category" $EXTRA_ARGS
done

echo ""
echo "Tutte le estrazioni sono completate. Risultati in ./outputs/<categoria>/vit_features/"
