#!/bin/bash
set -e

# Vérification rapide du dossier d'exécution
current_folder=$(basename "$PWD")
if [ "$current_folder" != "scripts" ]; then
    echo "⚠️ Merci de lancer ce script depuis le dossier 'scripts'"
    exit 1
fi

echo '============================================================'
echo 'LANCEMENT DES INFÉRENCES PYTHON EN LOT (KPIs SMARTEYE)'
echo '============================================================'

# Chemins statiques fournis
videos_dir="/mnt/data/pbys/scratch/OoP/python_inference/data/forvia4testset"
config_path="/mnt/data/pbys/scratch/OoP/python_inference/data/forvia4testset/eval_onnx_july_4_forvia_sequences.yaml"
gt_csv_dir="/mnt/data/pbys/scratch/OoP/python_inference/output/results/golden_reference/me/eval_inference/forvia4test/onnx"
python_script="../src/py_inference_v2_same_result_C_kpi_golden_ref_multiple_video.py"

# Dossiers dynamiques racines
base_model_dir="/mnt/data/pbys/training/saved_models/OoP/data_aug_smartEye"
base_out_dir="/mnt/data/pbys/scratch/OoP/python_inference/output/results/data_aug_bboxes/inference_mimic_c/forvia4test"


for i in {1..5}; do
    echo ""
    echo ">>> Analyse du dossier avec bruit : 0.${i} <<<"
    
    model_dir="${base_model_dir}/noise0point${i}/ckpt"
    
    if [ ! -d "$model_dir" ]; then
        echo "  ⚠️ Dossier de poids introuvable : $model_dir"
        continue
    fi

    # Boucle sur TOUS les modèles best ou 2nd_best trouvés
    for onnx_model in $(find "$model_dir" -maxdepth 1 -type f \( -name "best*.onnx" -o -name "2nd_best*.onnx" \)); do
        
        model_name=$(basename "$onnx_model" .onnx)
        echo "  -> ONNX chargé : $model_name.onnx"
        
        # Le dossier de sortie inclut maintenant le nom du modèle pour éviter les écrasements
        output_dir="${base_out_dir}/augpoint${i}_${model_name}"
        mkdir -p "$output_dir"
        
        echo "  -> Dossier de sortie : $output_dir"
        echo "  -> Lancement de l'inférence..."
        
        python3 "$python_script" \
            --model_path "$onnx_model" \
            --videos_dir "$videos_dir" \
            --with_tatoo_keypoints False \
            --output_dir "$output_dir" \
            --config_path "$config_path" \
            --gt_csv_dir "$gt_csv_dir"
            
    done
done

echo ""
echo "✅ Toutes les inférences et calculs de KPI sont terminés !"