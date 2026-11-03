"""Run posture model inference on multiple video files (auto-detects tracking CSV) and export predictions.

This script mimics the embedded C++ pipeline (verified line-by-line against
foms_positions.cpp and samples_positions_evaluation.cpp): crop with a 10% 
width-only margin, proportional letterbox, centered black padding, INTER_LINEAR interpolation.

Modified to support:
- Batch processing (folder containing multiple videos)
- KPI calculation per video via centralized annotations.csv file
- Final GLOBAL metric calculation via concatenation
- Inference on MULTIPLE OCCUPANTS (Mimic C++ Array of Occupants)
- Exporting spatial bounding box center (bbox_cx) for stable seat assignment
"""

# This model is working with both fp32 and int8 , it can give you really close results to 
# the C++ output but some really similar that are not exactly the same too 

import os
import sys
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from fire import Fire
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Go up to the repository root then down into training to find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../training')))
from src.utils.config import get_config
from src.posture.train_posture import load_model

# Pour tous mes codes, j'utilise des environnements Conda.
# Never remove my personal comments from the code.

# Canonical order of classes, reused for prediction export AND KPIs.
CPP_CLASS_ORDER = [
    "closetoairbag",
    "legsondashboard",
    "torsodissimetry",
    "reachingback",
    "lateralbodycontact",
    "oneshoulderincontact",
    "pelvisrotation",
    "pelvisdrift",
]


def _compute_kpi_for_dataframe(df_merged, sequence_name, threshold, cpp_class_order):
    """Utility function to compute KPIs on a merged DataFrame"""
    kpi_rows = []
    for cls in cpp_class_order:
        col_gt = f"{cls}_gt"
        if col_gt not in df_merged.columns or cls not in df_merged.columns:
            continue

        gt_full = df_merged[col_gt]
        pred_full = df_merged[cls]

        # Invalid frames marked by -1, excluded independently for each class.
        valid_mask = (gt_full != -1) & (pred_full != -1.0)
        gt_valid = gt_full[valid_mask]
        pred_probs = pred_full[valid_mask]

        if len(gt_valid) == 0:
            continue

        y_pred = (pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(gt_valid, y_pred, labels=[0, 1]).ravel()

        kpi_rows.append({
            "Sequence": sequence_name,
            "Class": cls,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Precision": round(precision_score(gt_valid, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(gt_valid, y_pred, zero_division=0), 4),
            "F1_Score": round(f1_score(gt_valid, y_pred, zero_division=0), 4),
            "Total_Frames": len(gt_valid),
        })
    return kpi_rows


def compute_kpis_from_merged_dict(merged_dfs_dict, threshold=0.5, cpp_class_order=None):
    """
    Computes KPIs (TP/TN/FP/FN/Precision/Recall/F1) per class and per video, 
    then adds a GLOBAL metric on the concatenation of all videos.
    """
    if cpp_class_order is None:
        cpp_class_order = CPP_CLASS_ORDER

    all_kpi_rows = []
    
    # 1. Calculations per video
    for seq_name, df_merged in merged_dfs_dict.items():
        all_kpi_rows.extend(
            _compute_kpi_for_dataframe(df_merged, seq_name, threshold, cpp_class_order)
        )

    # 2. GLOBAL Calculation
    if merged_dfs_dict:
        global_df = pd.concat(list(merged_dfs_dict.values()), ignore_index=True)
        all_kpi_rows.extend(
            _compute_kpi_for_dataframe(global_df, "GLOBAL", threshold, cpp_class_order)
        )

    return pd.DataFrame(all_kpi_rows)


def run_video_inference(model_path, video_path, tracking_csv, output_csv, model, session, dlr_model, device, input_names, output_name, config, cpp_class_order, class_names):
    """
    Run inference on a single video frame by frame for MULTIPLE OCCUPANTS.
    Automatically loads the associated tracking CSV to crop each person, and exports posture predictions.
    """
    if not os.path.isfile(tracking_csv):
        print(f"⚠️ Tracking CSV not found for {os.path.basename(video_path)} at path: {tracking_csv}. Skipped.")
        return None

    try:
        df_input = pd.read_csv(tracking_csv)
    except pd.errors.EmptyDataError:
        print(f"⚠️ EMPTY tracking CSV for {os.path.basename(video_path)}. Video skipped.")
        return None

    # --- DYNAMIC OCCUPANT DETECTION ---
    # Scan columns to find all IDs (e.g., 0 for Driver, 1 for Passenger)
    occupant_ids = set()
    for col in df_input.columns:
        if col.startswith('/trackingState/occupants/'):
            try:
                occ_id = int(col.split('/')[3])
                occupant_ids.add(occ_id)
            except ValueError:
                pass
    occupant_ids = sorted(list(occupant_ids))
    
    if not occupant_ids:
        print(f"⚠️ No occupants found in tracking CSV header for {os.path.basename(video_path)}.")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n🎥 Processing video: {os.path.basename(video_path)} ({total_frames} frames)")
    print(f"👥 Detected occupants: {occupant_ids}")
    
    img_height = 256
    img_width = 192
    results = []
    
    is_dlr = dlr_model is not None
    is_onnx = session is not None

    frame_idx = 0
    max_iterations = min(total_frames, len(df_input))
    pbar = tqdm(total=max_iterations, colour="blue", desc="Inference")
    
    with torch.no_grad():
        while frame_idx < max_iterations:
            ret, frame = cap.read()
            if not ret:
                break 

            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row = df_input.iloc[frame_idx]
            
            # --- C++ LOOP: ITERATE OVER EACH OCCUPANT ---
            for occ_id in occupant_ids:
                # Dynamic CSV paths for current occupant
                col_x = f'/trackingState/occupants/{occ_id}/boundingBox/upperLeft/0'
                col_y = f'/trackingState/occupants/{occ_id}/boundingBox/upperLeft/1'
                col_w = f'/trackingState/occupants/{occ_id}/boundingBox/width'
                col_h = f'/trackingState/occupants/{occ_id}/boundingBox/height'

                val_x = row.get(col_x, pd.NA)
                val_y = row.get(col_y, pd.NA)
                val_w = row.get(col_w, pd.NA)
                val_h = row.get(col_h, pd.NA)

                # Base dictionary for this frame AND this occupant
                frame_data = {"index": frame_idx, "occupant_id": occ_id}

                # Tracking loss management (failsafe)
                if pd.isna(val_x) or pd.isna(val_y) or pd.isna(val_w) or pd.isna(val_h):
                    for class_name in cpp_class_order:
                        frame_data[class_name] = -1.0
                    frame_data["bbox_cx"] = -1.0
                    results.append(frame_data)
                    continue 

                x = int(float(val_x))
                y = int(float(val_y))
                w = int(float(val_w))
                h = int(float(val_h))

                # Calculate spatial center of bounding box for evaluation filtering
                bbox_cx = float(x + (w / 2.0))
                frame_data["bbox_cx"] = bbox_cx

                # Geometric Guard
                if w <= 0 or h <= 0 or x < 0 or y < 0:
                    for class_name in cpp_class_order:
                        frame_data[class_name] = -1.0
                    frame_data["bbox_cx"] = -1.0
                    results.append(frame_data)
                    continue

                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(image_gray.shape[1], x + w), min(image_gray.shape[0], y + h)
                
                delta_width = int(0.1 * float(x2 - x1))
                x1_margin = max(0, x1 - delta_width)
                x2_margin = min(image_gray.shape[1] - 1, x2 + delta_width)

                if x2_margin > x1_margin and y2 > y1:
                    roi_img = image_gray[y1:y2, x1_margin:x2_margin]
                else:
                    roi_img = image_gray 

                roi_height, roi_width = roi_img.shape[:2]
                scalex = img_width / float(roi_width)
                scaley = img_height / float(roi_height)
                scale = min(scalex, scaley)
                
                dst_cols = int(round(roi_width * scale))
                dst_rows = int(round(roi_height * scale))
                
                resized_img = cv2.resize(roi_img, (dst_cols, dst_rows), interpolation=cv2.INTER_LINEAR)
                
                top = int((img_height - dst_rows) / 2)
                bottom = img_height - top - dst_rows
                right = int((img_width - dst_cols) / 2)
                left = img_width - right - dst_cols
                
                final_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

                if is_dlr:
                    img_tensor_np = final_img.astype(np.uint8)
                else:
                    img_tensor_np = final_img.astype(np.float32) * 0.00392156862
                
                img_tensor_np = np.expand_dims(img_tensor_np, axis=0) 
                img_tensor_np = np.expand_dims(img_tensor_np, axis=0) 
                img_tensor = torch.from_numpy(img_tensor_np) 
                
                if is_dlr:
                    dlr_inputs = {input_names[0]: img_tensor_np}
                    dlr_outs = dlr_model.run(dlr_inputs)
                    pred_raw = np.squeeze(dlr_outs[0])
                    pos_probs = 1.0 / (1.0 + np.exp(-pred_raw)) 
                elif is_onnx:
                    ort_inputs = {input_names[0]: img_tensor_np}
                    ort_outs = session.run([output_name], ort_inputs)
                    pred_raw = np.squeeze(ort_outs[0])
                    pos_probs = 1.0 / (1.0 + np.exp(-pred_raw)) 
                else:
                    pred = model(img_tensor.to(device))
                    pos_probs = np.squeeze(torch.sigmoid(pred).cpu().numpy())

                if pos_probs.ndim == 0:
                    pos_probs = np.array([pos_probs])

                max_pos_prob = np.clip(np.max(pos_probs), 0.0, 1.0)
                correctly_seated_prob = 1.0 - max_pos_prob
                
                full_probs = [correctly_seated_prob] + pos_probs.tolist()
                
                for class_name_from_model, prob in zip(class_names, full_probs):
                    clean_name = class_name_from_model.lower().replace("_", "").replace(" ", "")
                    final_name = "closetoairbag" if clean_name == "tooclosetoairbag" else clean_name
                    frame_data[final_name] = float(prob)

                for cpp_col in cpp_class_order:
                    if cpp_col not in frame_data:
                        frame_data[cpp_col] = -1.0

                results.append(frame_data)
            
            frame_idx += 1
            pbar.update(1)

    cap.release()
    pbar.close()

    df_results = pd.DataFrame(results)
    
    # Adding bbox_cx to final export columns
    final_columns = ["index", "occupant_id", "bbox_cx"] + cpp_class_order
    df_results = df_results[final_columns]
    
    df_results.to_csv(output_csv, index=False)
    print(f"  -> Predictions saved: {output_csv}")

    return df_results


def main(model_path, videos_dir, output_dir=None, config_path=None,
         annotations_csv=None, kpi_csv_path=None, kpi_threshold=0.5):
    """
    Runs posture model inference on a batch of videos.

    Args:
        model_path (str): Path to the model (.onnx, .pt, or DLR folder).
        videos_dir (str): Folder containing input videos (.mp4).
        output_dir (str, optional): Output folder for individual prediction CSVs.
        config_path (str, optional): Path to config.yaml file.
        annotations_csv (str, optional): Centralized ground truth file (annotations.csv).
        kpi_csv_path (str, optional): Path for KPI summary file (CSV). 
                                       Default: <output_dir>/KPI_summary_videos.csv
        kpi_threshold (float): Binarization threshold for KPIs.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    if output_dir is None:
        output_dir = os.path.join(videos_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    if config_path is None:
        inference_folder = os.path.dirname(model_path.rstrip('/')) if os.path.isdir(model_path) else os.path.dirname(model_path)
        config_path = os.path.join(inference_folder, "../config.yaml")
    assert os.path.isfile(config_path), f"Config file not found: {config_path}"

    config = get_config()
    config.set_config(config_path)
    
    class_names = ["correctly_seated"] + list(config.input.positive_labels)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- MODEL LOADING (ONCE FOR ALL VIDEOS) ---
    is_dlr = os.path.isdir(model_path)
    is_onnx = model_path.endswith(".onnx")
    
    model = session = dlr_model = None
    input_names = output_name = None
    
    if is_dlr:
        import dlr
        print(f"⚙️ Loading quantified DLR model: {model_path}")
        dlr_model = dlr.DLRModel(model_path, 'cpu')
        input_names = dlr_model.get_input_names()
    elif is_onnx:
        import onnxruntime as ort
        print(f"⚙️ Loading ONNX model: {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        input_names = [inp.name for inp in session.get_inputs()]
        output_name = session.get_outputs()[0].name
    else:
        print(f"⚙️ Loading PyTorch model: {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        trained_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(trained_state_dict, "module.")
        model = load_model(config)
        model.load_state_dict(trained_state_dict)
        model.to(device)
        model.eval()

    # --- CENTRALIZED ANNOTATIONS FILE LOADING ---
    df_annotations = None
    if annotations_csv and os.path.isfile(annotations_csv):
        print(f"📊 Loading annotations file: {annotations_csv}")
        df_annotations = pd.read_csv(annotations_csv, sep=None, engine='python', encoding='latin1', header=2)

    # --- VIDEO BATCH PROCESSING ---
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    if not video_files:
        print(f"❌ No .mp4 videos found in: {videos_dir}")
        return

    print(f"🚀 Starting processing: {len(video_files)} video(s) found.")
    
    # Deduce 'tracking' folder (at the same level as the 'mp4' folder)
    parent_dir = os.path.dirname(videos_dir.rstrip('/\\'))
    tracking_dir = os.path.join(parent_dir, "tracking")

    merged_dfs_dict = {}

    for video_path in sorted(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_filename = os.path.basename(video_path)
        output_csv = os.path.join(output_dir, f"{video_name}_predictions.csv")
        
        # Exact path of bbox file in tracking folder
        tracking_csv = os.path.join(tracking_dir, f"{video_name}.csv")
        
        # --- RESUME MECHANISM ---
        if os.path.exists(output_csv):
            print(f"⏩ Video already processed: {video_name}. Loading existing predictions...")
            try:
                df_pred = pd.read_csv(output_csv)
            except pd.errors.EmptyDataError:
                print(f"⚠️ Corrupted predictions for {video_name}, restarting inference...")
                df_pred = run_video_inference(
                    model_path, video_path, tracking_csv, output_csv, 
                    model, session, dlr_model, device, input_names, output_name, 
                    config, CPP_CLASS_ORDER, class_names
                )
        else:
            # Run normal inference
            df_pred = run_video_inference(
                model_path, video_path, tracking_csv, output_csv, 
                model, session, dlr_model, device, input_names, output_name, 
                config, CPP_CLASS_ORDER, class_names
            )
        # -------------------------------------

        if df_pred is not None and df_annotations is not None:
            # NOTE : KPI calculation is not done here because evaluation 
            # is now managed by the separate evaluate_kpis.py script.
            pass

    # --- FINAL KPI REMINDER ---
    if df_annotations is not None:
        print("\nℹ️ Inference is complete. Please use the `evaluate_kpis.py` script to evaluate predictions.")


if __name__ == "__main__":
    Fire(main)