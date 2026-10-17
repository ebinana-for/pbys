# Posture Inference

## Environment Setup

Create and activate your conda environment:

```bash
conda create -n posture_inference python=3.10 -y
conda activate posture_inference
```

Install required packages:

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy pandas
pip install tqdm fire
pip install scikit-learn
pip install onnxruntime-gpu
```

---

## Dataset

Download the **Forvia test dataset** from the following SharePoint folder:

[Forvia Test Dataset](https://faurecia.sharepoint.com/:f:/r/sites/IRYS-Research-Projects-OutOfPosition/Shared%20Documents/Probayes_OoP/forvia4testset?d=w3f309a9bd85d4737a8d3f4a1b5a1c550&csf=1&web=1&e=0fkt19)

Extract/copy the `forvia4testset` folder into:

```text
data/
└── forvia4testset/
```

The folder must contain the `.mp4` videos and their associated tracking `.csv` files.

## Run Inference

From the project root directory:

```bash
python3 src/inference_python/py_inference_v2_same_result_C_kpi_golden_ref_multiple_video.py \
    --model_path ./weights/onnx/epoch_40_batch_1.onnx \
    --videos_dir ./data/forvia4testset \
    --with_tatoo_keypoints False \
    --output_dir ./output/forvia4test2/golden_ref_kpi/onnx \
    --config_path ./src/config/eval_onnx_july_4_forvia_sequences.yaml \
    --gt_csv_dir ./gt
```

### Inputs

- `--model_path`: ONNX model file
- `--videos_dir`: folder containing `.mp4` files and associated tracking `.csv`
- `--config_path`: model configuration file
- `--gt_csv_dir`: ground-truth CSV folder used for KPI computation

### Outputs

- Per-video prediction CSV files
- `KPI_summary_videos.csv` containing Precision, Recall, F1-Score, TP, TN, FP, and FN metrics

---

## Compare Python vs C++ Results

Generate the final comparison report between:

- Ground Truth
- Python inference output
- C++ inference output

```bash
python3 src/kpi_comparison/comparison_3_inference_table_output_multiple_video_frame_align.py \
    --eval_csv_dir ./gt \
    --py_inf_csv_dir ./output/forvia4test2/golden_ref_kpi/onnx \
    --c_inf_csv_dir ./output/inference/forvia4test/ \
    --output_csv ./output/comparison_3_inference_multiple_vid_frame_align2.csv
```

### Output

The script generates:

```text
./output/comparison_3_inference_multiple_vid_frame_align2.csv
```

This file contains the final frame-aligned comparison between the Ground Truth, Python inference results, and C++ inference results, along with the associated KPI metrics.
