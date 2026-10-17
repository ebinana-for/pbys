CKPT_FILE="/media/jdefretin/DATA/PSA_domain_adaptation/sshfs_mountpoint_kirby_mnt_data/jdefretin/Forvia/saved_models/2025-01-20T10h20m02s/ckpt/epoch_95.onnx"
OUT_PATH="/media/jdefretin/DATA/PSA_domain_adaptation/sshfs_mountpoint_kirby_mnt_data/forvia/private_datasets/recordings_forvia_30_01/mpeg_with_keypoints/"

VIDEO_PATH="/media/jdefretin/DATA/PSA_domain_adaptation/sshfs_mountpoint_kirby_mnt_data/forvia/private_datasets/recordings_forvia_30_01/mpeg/joseph_defretin1_40_M_183_70.mp4"
CSV_FILE="/media/jdefretin/DATA/PSA_domain_adaptation/sshfs_mountpoint_kirby_mnt_data/forvia/private_datasets/recordings_forvia_30_01/raw/joseph_defretin1_40_M_183_70.sec/log/tracking_000000.csv"
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --model_path "$CKPT_FILE" --video_path "$VIDEO_PATH" --csv_file "$CSV_FILE" --out_path "$OUT_PATH"


ROOT_DATA_DIR=/media/jdefretin/DATA/PSA_domain_adaptation/sshfs_mountpoint_kirby_mnt_data/forvia/private_datasets/recordings_forvia_23_05
REC=benjamin_31_ohc
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

REC=benjamin_40_ohc
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

REC=francois_48_ohc
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

REC=IHAB_24_CID
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

REC=Jacquemin_49_ohc
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

REC=OSCAR_29_cid_1
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

REC=samuel_36_cid
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

REC=Samuel_36_ohc
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

REC=Sebastien_46_ohc
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

REC=seb_CID_46_1
CUDA_VISIBLE_DEVICES=0 python "$(dirname ${BASH_SOURCE[0]})/../src/age_gender/viewer.py" --video_path "${ROOT_DATA_DIR}/mpeg/${REC}.mp4" --csv_file  "${ROOT_DATA_DIR}/raw/${REC}.sec/log/tracking_000000.csv" --out_path "$ROOT_DATA_DIR"

