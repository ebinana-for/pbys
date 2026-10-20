#!/bin/bash
set -e
current_path=$(pwd)
current_folder=$(basename $current_path)
mandatory_folder="scripts"
if [ "$current_folder" != "$mandatory_folder" ]; then
        echo "You have to be in folder $mandatory_folder to launch the scripts"
        exit 1
fi

# default params
device="0"
config_file="../config/OoP/train/train_config_posture_smarteye.yaml"

while getopts d:c: flag
do
        case "${flag}" in
                d) device=${OPTARG};;
                c) config_file=${OPTARG};;
        esac
done
echo "device: $device"
echo "config_file: $config_file"

echo '============================================================'
echo 'LAUNCH TRAINING POSTURE SMART EYE SWEEP'
echo '============================================================'

# --- AJOUT ICI POUR LA REPRODUCTIBILITÉ ABSOLUE ---
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# --------------------------------------------------

# Liste des valeurs de bruit (edge_noise) à tester pour simuler SmartEye
noises=(0.1 0.2 0.3 0.4 0.5)

for noise in "${noises[@]}"
do
    echo ""
    echo ">>> Démarrage de l'itération avec edge_noise = $noise <<<"
    
    # 1. Création d'un fichier de configuration temporaire pour cette itération
    temp_config="${config_file%.yaml}_temp_${noise}.yaml"
    
    # 2. Remplacement à la volée de la valeur edge_noise en conservant l'indentation YAML
    sed "s/edge_noise: .*/edge_noise: ${noise}/g" "$config_file" > "$temp_config"

    # 3. Lancement de l'entraînement sur le modèle de posture
    CUDA_VISIBLE_DEVICES=$device python ../src/posture/train_posture.py --config_file "$temp_config"

    # 4. Nettoyage du fichier temporaire pour garder le dossier propre
    rm "$temp_config"
done

echo "Toutes les itérations sont terminées avec succès !"