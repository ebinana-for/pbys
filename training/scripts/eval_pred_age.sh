CONFIG_FILE=/mnt/data/jdefretin/Forvia/saved_models/age_gender/2025-04-02T16h50m01s/config.yaml
CKPT_FILE=/mnt/data/jdefretin/Forvia/saved_models/age_gender/2025-04-02T16h50m01s/ckpt/epoch_95.pth
CUDA_VISIBLE_DEVICES=3 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/eval_pred_age.py" --config_file $CONFIG_FILE --ckpt_file $CKPT_FILE