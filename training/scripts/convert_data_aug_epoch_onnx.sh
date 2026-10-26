#!/bin/bash
set -e

current_path=$(pwd)
current_folder=$(basename "$current_path")
mandatory_folder="scripts"
if [ "$current_folder" != "$mandatory_folder" ]; then
    echo "You have to be in folder $mandatory_folder to launch the scripts"
    exit 1
fi

echo '============================================================'
echo 'LANCEMENT DE LA CONVERSION ONNX EN LOT (SWEEP SMARTEYE)'
echo '============================================================'

device="0"
config_file="../config/OoP/train/train_config_posture_smarteye.yaml"
base_dir="/mnt/data/pbys/training/saved_models/OoP/data_aug_smartEye"

for i in {1..5}
do
    # Ajout du sous-dossier /ckpt/ exactement là où se trouvent tes modèles
    folder_path="${base_dir}/noise0point${i}/ckpt"
    
    if [ ! -d "$folder_path" ]; then
        echo "⚠️  Dossier introuvable, ignoré : $folder_path"
        continue
    fi

    echo ""
    echo ">>> Analyse du dossier : noise0point${i}/ckpt <<<"

    # On cherche les fichiers best_*.pth ou 2nd_best_*.pth
    find "$folder_path" -maxdepth 1 -type f \( -name "best*.pth" -o -name "2nd_best*.pth" \) | while read pth_file; do
        
        filename=$(basename "$pth_file")
        echo "  -> Démarrage de la conversion pour : $filename"
        
        bash convert_posture.sh -d "$device" -b 1 -c "$config_file" -m "$pth_file"
        
    done
done

echo ""
echo "✅ Toutes les conversions ONNX sont terminées !"