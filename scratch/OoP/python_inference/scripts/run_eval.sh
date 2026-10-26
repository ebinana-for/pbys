#!/bin/bash
set -e

# Vérification rapide du dossier d'exécution
current_folder=$(basename "$PWD")
if [ "$current_folder" != "scripts" ]; then
    echo "⚠️ Merci de lancer ce script depuis le dossier 'scripts'"
    exit 1
fi

echo '============================================================'
echo 'LANCEMENT DE LA SECONDE ÉVALUATION PYTHON EN LOT'
echo '============================================================'

# Chemins statiques fournis
config_path="/mnt/data/pbys/scratch/OoP/python_inference/data/forvia4testset/eval_onnx_july_4_forvia_sequences.yaml"
python_script="../src/evaluation_inference_20_08_kpi_golden_ref_multiple_vid.py"
# Dossiers dynamiques racines
base_model_dir="/mnt/data/pbys/training/saved_models/OoP/data_aug_smartEye"
# J'ai adapté le dossier de sortie pour le différencier du précédent
base_out_dir="/mnt/data/pbys/scratch/OoP/python_inference/output/results/data_aug_bboxes/eval/forvia4test"

# Note personnelle : Assure-toi que ton environnement Conda est bien activé avant de lancer !

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
        
        # Le dossier de sortie inclut le nom du modèle pour éviter les écrasements
        output_dir="${base_out_dir}/augpoint${i}_${model_name}"
        mkdir -p "$output_dir"
        
        echo "  -> Dossier de sortie : $output_dir"
        echo "  -> Lancement de l'évaluation python..."
        
        python3 "$python_script" \
            --model_path "$onnx_model" \
            --config_path "$config_path" \
            --output_dir "$output_dir"
            
    done
done

echo ""
echo "✅ Toutes les évaluations sont terminées !"