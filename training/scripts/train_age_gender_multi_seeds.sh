#!/bin/bash

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

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
config_file="../config/AgeGender/train/train_config_age_gender.yaml"
seeds="42 42 55 55 66 66 77 77 1000 1000"
# seeds=$(shuf -i 1-10000 -n 10) # 10 random seeds selection

while getopts d:c: flag
do
        case "${flag}" in
                d) device=${OPTARG};;
                c) config_file=${OPTARG};;
                s) seeds=${OPTARG};;
        esac
done
echo "device: $device"
echo "config_file: $config_file"
echo "seeds to run: $seeds"

echo '============================================================'
echo 'LAUNCH TRAINING AGE GENDER'
echo '============================================================'


for seed in $seeds; do
    echo "------------------------------------------------------------"
    echo "Starting training with SEED: $seed"
    echo "------------------------------------------------------------"
    
    export PYTHONHASHSEED=$seed
    
    CUDA_VISIBLE_DEVICES=$device python ../src/age_gender/train_multi_heads.py \
        --config_file $config_file \
        --seed $seed
        
    echo "Finished training for SEED: $seed"
    sleep 2
done