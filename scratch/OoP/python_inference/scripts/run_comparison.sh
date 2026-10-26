#!/bin/bash
set -e

# Vérification rapide du dossier d'exécution
current_folder=$(basename "$PWD")
if [ "$current_folder" != "scripts" ]; then
    echo "⚠️ Merci de lancer ce script depuis le dossier 'scripts'"
    exit 1
fi

echo '============================================================'
echo 'LANCEMENT DE LA COMPARAISON FINALE EN LOT'
echo '============================================================'

# Script Python ciblé
python_script="../../golden_reference_creation/src/OoP/comparison_3_inference_table_output_multiple_video_frame_align.py"

# Dossiers dynamiques racines basés sur tes scripts précédents
base_model_dir="/mnt/data/pbys/training/saved_models/OoP/data_aug_smartEye"
base_py_inf_dir="/mnt/data/pbys/scratch/OoP/python_inference/output/results/data_aug_bboxes/inference_mimic_c/forvia4test"
base_eval_dir="/mnt/data/pbys/scratch/OoP/python_inference/output/results/data_aug_bboxes/eval/forvia4test"

# Dossier de sortie pour les CSV de comparaison
base_out_dir="/mnt/data/pbys/scratch/OoP/python_inference/output/results/data_aug_bboxes/comparison/forvia4test"

# Dossier pour l'inférence C++ (chemin fourni)
c_inf_dir="/mnt/data/pbys/scratch/OoP/python_inference/output/results/golden_reference/sunan/inference_c/forvia4test"

# Note personnelle : Assure-toi que ton environnement Conda est bien activé avant de lancer !

for i in {1..5}; do
    echo ""
    echo ">>> Génération de la comparaison pour bruit : 0.${i} <<<"
    
    model_dir="${base_model_dir}/noise0point${i}/ckpt"
    
    if [ ! -d "$model_dir" ]; then
        echo "  ⚠️ Dossier de poids introuvable : $model_dir"
        continue
    fi

    # Boucle sur TOUS les modèles best ou 2nd_best trouvés
    for onnx_model in $(find "$model_dir" -maxdepth 1 -type f \( -name "best*.onnx" -o -name "2nd_best*.onnx" \)); do
        
        model_name=$(basename "$onnx_model" .onnx)
        echo "  -> Traitement du modèle : $model_name"
        
        # Reconstitution des chemins exacts générés lors des deux étapes précédentes
        eval_csv_dir="${base_eval_dir}/augpoint${i}_${model_name}"
        py_inf_csv_dir="${base_py_inf_dir}/augpoint${i}_${model_name}"
        
        # Fichier CSV de sortie spécifique à ce modèle
        mkdir -p "$base_out_dir"
        output_csv="${base_out_dir}/comparison_augpoint${i}_${model_name}.csv"
        
        echo "  -> Fichier de sortie : $output_csv"
        
        python3 "$python_script" \
            --eval_csv_dir "$eval_csv_dir" \
            --py_inf_csv_dir "$py_inf_csv_dir" \
            --c_inf_csv_dir "$c_inf_dir" \
            --output_csv "$output_csv"
            
    done
done

echo ""
echo "✅ Toutes les comparaisons sont terminées !"