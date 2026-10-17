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
config_file="../config/train_config_age_gender.yaml"

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
echo 'LAUNCH TRAINING AGE GENDER'
echo '============================================================'

# --- AJOUT ICI POUR LA REPRODUCTIBILITÉ ABSOLUE ---
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# --------------------------------------------------

CUDA_VISIBLE_DEVICES=$device python ../src/age_gender/train_multi_heads.py --config_file $config_file
