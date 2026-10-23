#!/bin/bash
cd ../src

# 1. Chemins absolus vers tes configurations et données
CONFIG_FILE="/mnt/data/pbys/training/config/AgeGender/test/test_config_age_gender_onnx_probayes.yaml"

# Chemins pour le dataset rapide (FORVIA)
INPUT_DIR="/mnt/data/OoP/data_to_infer/forvia_30_01"
CPP_INF_DIR="/mnt/data/OoP/data_to_infer/forvia_30_01/output_AgeGender"
BASE_OUTPUT_DIR="/mnt/data/pbys/training/saved_evals/AgeGender/private/ablation_study_forvia"

# Les 4 combinaisons de l'étude d'ablation
# Les combinaisons de l'étude d'ablation finale
declare -a tests=(
    # "r_channel bbox forced"
    # "grayscale bbox forced"
    # "r_channel keypoints forced"
    "grayscale keypoints forced"      # Le test avec le problème d'inférence
    "grayscale keypoints letterbox"   # 🎯 LE TEST ULTIME : Clone C++ parfait
)

echo "=========================================================="
echo "Démarrage de l'étude d'ablation sur le dataset Forvia"
echo "=========================================================="

for test in "${tests[@]}"; do
    read -r color crop resize <<< "$test"
    
    echo -e "\n---> 🚀 LANCEMENT DU TEST : Color = $color | Crop = $crop | Resize = $resize <---"
    
    EVAL_DIR="$BASE_OUTPUT_DIR/eval_${color}_${crop}_${resize}"
    KPI_DIR="$BASE_OUTPUT_DIR/kpi_${color}_${crop}_${resize}"
    mkdir -p "$EVAL_DIR" "$KPI_DIR"

    echo "1/2 : Génération des prédictions Python ONNX..."
    python run_ablation_eval.py -c "$CONFIG_FILE" \
        --color_mode "$color" \
        --crop_mode "$crop" \
        --resize_mode "$resize" \
        --output_dir "$EVAL_DIR" \
        --input_dir "$INPUT_DIR"
        
    echo "✅ Terminé pour [$color + $crop]. Dashboard sauvegardé dans : $KPI_DIR"
done

echo -e "\n🎉 Tous les tests sont terminés !"