#!/bin/bash
cd ../src

# 1. Chemins absolus vers tes configurations et données
CONFIG_FILE="/mnt/data/pbys/training/config/AgeGender/test/test_config_age_gender_onnx_probayes.yaml"

# Chemins pour le dataset rapide (FORVIA)
INPUT_DIR="/mnt/data/OoP/data_to_infer/forvia_30_01"
CPP_INF_DIR="/mnt/data/OoP/data_to_infer/forvia_30_01/output_AgeGender"
BASE_OUTPUT_DIR="/mnt/data/pbys/training/saved_evals/AgeGender/private/ablation_study_forvia"

# Les combinaisons de l'étude d'ablation finale
declare -a tests=(
    # "r_channel bbox forced"
    # "grayscale bbox forced"
    # "r_channel keypoints forced"
    # "grayscale keypoints forced"      
    "grayscale keypoints letterbox rolling"     
    "grayscale keypoints letterbox cumulative"      # 🎯 LE TEST ULTIME : Clone C++ parfait
)

echo "=========================================================="
echo "Démarrage de l'étude d'ablation sur le dataset Forvia"
echo "=========================================================="

for test in "${tests[@]}"; do
    read -r color crop resize smoothing <<< "$test"
    
    echo -e "\n---> 🚀 LANCEMENT DU TEST : Color=$color | Crop=$crop | Resize=$resize | Smooth=$smoothing <---"
    
    EVAL_DIR="$BASE_OUTPUT_DIR/eval_${color}_${crop}_${resize}_${smoothing}"
    KPI_DIR="$BASE_OUTPUT_DIR/kpi_${color}_${crop}_${resize}_${smoothing}"
    mkdir -p "$EVAL_DIR" "$KPI_DIR"

    echo "1/2 : Génération des prédictions Python ONNX..."
    python run_ablation_eval.py -c "$CONFIG_FILE" \
        --color_mode "$color" \
        --crop_mode "$crop" \
        --resize_mode "$resize" \
        --smoothing_mode "$smoothing" \
        --output_dir "$EVAL_DIR" \
        --input_dir "$INPUT_DIR"
        
    echo "2/2 : Calcul des KPI et de l'alignement..."
    # Appel de ton script de comparaison (ajuste les noms d'arguments si besoin selon ton fichier comparison_eval_c.py)
    python comparison_eval_c.py \
        --eval_dir "$EVAL_DIR" \
        --inf_dir "$CPP_INF_DIR" \
        --output_dir "$KPI_DIR"
        
    echo "✅ Terminé. Dashboard sauvegardé dans : $KPI_DIR"
done

echo -e "\n🎉 Tous les tests sont terminés !"