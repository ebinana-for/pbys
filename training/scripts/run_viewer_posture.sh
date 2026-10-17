CONFIG_FILE="/media/jdefretin/DATA/PSA_domain_adaptation/sshfs_mountpoint_kirby_mnt_data/jdefretin/Forvia/saved_models/posture1/2025-05-10T13h16m06s/config.yaml"
CKPT_FILE="/media/jdefretin/DATA/PSA_domain_adaptation/sshfs_mountpoint_kirby_mnt_data/jdefretin/Forvia/saved_models/posture1/2025-05-10T13h16m06s/ckpt/epoch_43.pth"
OUT_PATH="/media/jdefretin/DATA/PSA_domain_adaptation/sshfs_mountpoint_kirby_mnt_data/jdefretin/Forvia/Debug"

CUDA_VISIBLE_DEVICES=2 python "$(dirname ${BASH_SOURCE[0]})/../src/posture/viewer.py" --model_config "$CONFIG_FILE" --model_path "$CKPT_FILE" --out_path "$OUT_PATH"
