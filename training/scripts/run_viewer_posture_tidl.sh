VIDEO_FILE="ihab_haddad1.mp4"
PRED_FILE="ihab_haddad1_out_positions.csv"
GT_FILE="/data/forvia/recordings_forvia_31_01/test/labels/ihab_haddad1/default.txt"
OUT_DIR="/data/forvia/evals/posture/ihab_haddad1"
python "$(dirname ${BASH_SOURCE[0]})/../src/posture/viewer_tidl.py" --video_file "$VIDEO_FILE" --pred_file "$PRED_FILE" --gt_file "$GT_FILE" --out_dir "$OUT_DIR"

