ROOT_DATA_DIR=/data/forvia/private_datasets/recordings_pbys_9_4_25
REC=2025-02-09_12-08-45
python "$(dirname ${BASH_SOURCE[0]})/../../SDK/Ubuntu/sec2video.py" --input "${ROOT_DATA_DIR}/raw/${REC}.sec/" --framerate 30
