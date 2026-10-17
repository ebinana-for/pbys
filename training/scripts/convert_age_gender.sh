#!/bin/bash
set -e
current_path=$(pwd)
current_folder=$(basename $current_path)
mandatory_folder="scripts"
if [ "$current_folder" != "$mandatory_folder" ]; then
        echo "You have to be in folder $mandatory_folder to launch the scripts"
        exit 1
fi

FUSE=false
DYNAMIC=false

while getopts d:b:c:m:fy flag
do
        case "${flag}" in
                d) DEVICE=${OPTARG};;
                b) BATCH_SIZE=${OPTARG};;
                c) CONFIG_PATH=${OPTARG};;
                m) MODEL_PATH=${OPTARG};;
                f) FUSE=true ;;
                y) DYNAMIC=true;;
        esac
done
echo "DEVICE: $DEVICE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "DYNAMIC: $DYNAMIC"
echo "FUSE: $FUSE"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "CONFIG_PATH: $CONFIG_PATH"


CMD="python ../src/age_gender/convertONNX.py --config_file $CONFIG_PATH --model_path $MODEL_PATH --batch_size $BATCH_SIZE"
if $DYNAMIC
then
        MODEL_SUFFIX="dynamic"
        CMD="${CMD} --dynamic"
else
        MODEL_SUFFIX=$BATCH_SIZE
fi

if $FUSE
then
        MODEL_SUFFIX="${MODEL_SUFFIX}_fuse"
        CMD="${CMD} --fuse"
else
        MODEL_SUFFIX=$BATCH_SIZE
fi

echo $CMD
eval $CMD

mv ${MODEL_PATH/.pth/.onnx} ${MODEL_PATH/.pth/_batch_${MODEL_SUFFIX}.onnx}
onnxsim ${MODEL_PATH/.pth/_batch_${MODEL_SUFFIX}.onnx}   ${MODEL_PATH/.pth/_batch_${MODEL_SUFFIX}.onnx}

if ! $FUSE
then
echo '============================================================'
echo 'TEST INFERENCE ONNX'
echo '============================================================'
CUDA_VISIBLE_DEVICES=$DEVICE python ../src/age_gender/inference.py --model_path ${MODEL_PATH/.pth/_batch_${MODEL_SUFFIX}.onnx} --batch_size ${BATCH_SIZE}
fi
