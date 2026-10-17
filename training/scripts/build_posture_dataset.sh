ROOT_MPEG_DIR=/data/forvia/private_datasets/recordings_forvia_28_05/mpeg
ROOT_LABEL_DIR=/data/forvia/private_datasets/recordings_forvia_28_05/test/labels
ROOT_IMG_DIR=/data/forvia/private_datasets/recordings_forvia_28_05/test/images

python "$(dirname ${BASH_SOURCE[0]})/../src/posture/build_posture_dataset.py" --mpeg_root_dir "${ROOT_MPEG_DIR}" --label_root_dir "${ROOT_LABEL_DIR}" --out_path "${ROOT_IMG_DIR}"