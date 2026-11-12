"""
Multi-Occupant Evaluation Script for Age & Gender ODMS2
Evaluates multiple IDs simultaneously using a 'long format' dataframe structure.
(e.g., index, id, age, gender...)
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import onnxruntime as ort
import math
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../../training/src"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.config import get_config

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_body_keypoint(df, frame_cnt, occupant_id, keypoint_number):
    """
    Simulates foms::BodyKeyPoint(CsvReader, ...) constructor.
    """
    quality_col = f"/trackingState/occupants/{occupant_id}/bodyKeyPoints/{keypoint_number}/quality"
    x_col = f"/trackingState/occupants/{occupant_id}/bodyKeyPoints/{keypoint_number}/value/0"
    y_col = f"/trackingState/occupants/{occupant_id}/bodyKeyPoints/{keypoint_number}/value/1"
    
    if quality_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return {"valid": False}
        
    try:
        quality = float(df.at[frame_cnt, quality_col])
        valid = quality > 0.1  # minBodyPtQuality = 0.1F in C++
        
        if not valid:
            return {"valid": False}
            
        x = int(df.at[frame_cnt, x_col])
        y = int(df.at[frame_cnt, y_col])
        return {"valid": True, "x": x, "y": y, "quality": quality}
    except (ValueError, KeyError, TypeError):
        return {"valid": False}

def compute_face_crop_from_keypoints(df, frame_img, frame_cnt, occupant_id):
    """
    Exact Python translation of foms::computeFaceCropFromkeyPoints.
    Returns: (cropped_image, face_center_x) or (None, -1) if invalid.
    """
    LkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 1)  # LeftEye
    RkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 2)  # RightEye
    LEarkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 3) # LeftEar
    REarkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 4) # RightEar
    shinkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 25) # Chin
    
    if not (LkeyPt["valid"] and RkeyPt["valid"] and shinkeyPt["valid"] and REarkeyPt["valid"] and LEarkeyPt["valid"]):
        return None, -1

    interEyeDistance = math.sqrt((LkeyPt["x"] - RkeyPt["x"])**2 + (LkeyPt["y"] - RkeyPt["y"])**2)
    
    if interEyeDistance < 5.0:
        return None, -1
        
    if shinkeyPt["x"] < LkeyPt["x"] or shinkeyPt["x"] > RkeyPt["x"]:
        return None, -1
        
    if (LEarkeyPt["x"] > RkeyPt["x"] and REarkeyPt["x"] > RkeyPt["x"]) or \
       (LEarkeyPt["x"] < LkeyPt["x"] and REarkeyPt["x"] < LkeyPt["x"]):
        return None, -1
        
    eye_to_chin_vert_distance = (shinkeyPt["y"] - (RkeyPt["y"] + LkeyPt["y"]) / 2.0)
    
    if eye_to_chin_vert_distance <= interEyeDistance * 1.5:
        return None, -1
        
    # Face BBox Center X based on Ears
    center_x = (REarkeyPt["x"] + LEarkeyPt["x"]) / 2.0
    center_y = (LkeyPt["y"] + RkeyPt["y"]) / 2.0
    
    width = interEyeDistance * 3.2
    height = eye_to_chin_vert_distance * 2.2
    
    xmin = int((center_x - width / 2.0) + 0.5)
    ymin = int((center_y - 0.8 * height / 2.0) + 0.5)
    xmax = int((center_x + width / 2.0) + 0.5)
    ymax = int((center_y + height / 2.0) + 0.5)
    
    img_h, img_w = frame_img.shape[:2]
    
    if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
        return None, -1
        
    final_width = float(xmax - xmin)
    final_height = float(ymax - ymin)
    
    if final_height <= 0 or final_width <= 0:
        return None, -1
        
    ratio = final_width / final_height
    
    if ratio < 0.6 or ratio > 2.0 or final_width < 130 or final_height < 190:
        return None, -1
        
    return frame_img[ymin:ymax, xmin:xmax], center_x

def evaluate(video_dir, csv_dir, output_path, config):
    
    onnx_model_path = config.model.checkpoint
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    video_roots = set(os.path.splitext(f)[0] for f in mp4_files)
    csv_roots = set(os.path.splitext(f)[0] for f in csv_files)
    
    files_to_process = sorted(list(video_roots.intersection(csv_roots)))

    if not files_to_process:
        print("[Error] No matching video/csv pairs found.")
        return

    with tqdm(total=len(files_to_process), desc="Testing Multi-Occupant Long Format") as pbar_files:
        for f_root in files_to_process:
            
            video_path = os.path.join(video_dir, f_root + ".mp4")
            csv_path = os.path.join(csv_dir, f_root + ".csv")
            
            cap = cv2.VideoCapture(video_path)
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except pd.errors.EmptyDataError:
                print(f"[Warning] Empty CSV, no detections: {csv_path}")
                df = pd.DataFrame()

            count = 0
            file_results = []
            
            # IDs to track per frame
            occupant_ids_to_track = [0, 1]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                for occ_id in occupant_ids_to_track:
                    head_crop, bbox_cx = compute_face_crop_from_keypoints(df, frame_gray, count, occ_id)

                    if head_crop is not None and head_crop.shape[0] > 0 and head_crop.shape[1] > 0:
                        
                        input_image = cv2.resize(head_crop, (config.input.input_width, config.input.input_height)) 
                        input_image = np.expand_dims(input_image, axis=2) 
                        input_image = input_image.transpose((2, 0, 1)) 
                        input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0
                        
                        outputs = session.run(None, {input_name: input_image})
                        
                        if len(outputs) == 1:
                            logits_gender, logits_age = outputs[0][0][:2], outputs[0][0][2:]
                        else:
                            logits_gender, logits_age = outputs[0][0], outputs[1][0]

                        gender = {"M": 0, "F": 1}[["M", "F"][int(np.argmax(logits_gender))]]
                        score_gender = float(np.max(softmax(logits_gender)))
                        age_pred = logits_age[0] if logits_age.size == 1 else np.sum(softmax(logits_age) * np.arange(0, len(logits_age)))
                        
                        file_results.append([count, occ_id, int(age_pred), gender, score_gender, bbox_cx])
                    else:
                        file_results.append([count, occ_id, -1, -1, -1, -1])

                count += 1

            # Convert to DataFrame
            df_file = pd.DataFrame(file_results, columns=["index", "id", "age", "gender", "score_gender", "bbox_cx"])
            
            # --- Apply C++ Logic independently for each Occupant ID ---
            processed_dfs = []
            
            for occ_id in occupant_ids_to_track:
                # Isolate the timeline for a single ID
                df_id = df_file[df_file["id"] == occ_id].copy()
                
                valid_mask = df_id["age"] != -1
                df_valid = df_id[valid_mask].copy()
                
                if not df_valid.empty:
                    df_valid["age_mean"] = df_valid["age"].expanding(min_periods=1).mean().astype(int)
                    df_valid["gender_mean"] = (df_valid["gender"].expanding(min_periods=1).mean() >= 0.5).astype(int)
                    df_valid["score_gender_mean"] = df_valid["score_gender"].expanding(min_periods=1).mean()
                    
                    df_id = df_id.merge(df_valid[["index", "age_mean", "gender_mean", "score_gender_mean"]], on="index", how="left")
                else:
                    df_id["age_mean"] = -1
                    df_id["gender_mean"] = -1
                    df_id["score_gender_mean"] = -1

                # C++ Boolean Trap
                invalid_mask = df_id["age"] == -1
                df_id.loc[invalid_mask, ["age_mean", "gender_mean", "score_gender_mean", "bbox_cx"]] = -1
                df_id.fillna(-1, inplace=True)
                
                # Enforce C++ Integer types
                df_id["age_mean"] = df_id["age_mean"].astype(int)
                df_id["gender_mean"] = df_id["gender_mean"].astype(int)
                
                processed_dfs.append(df_id)
                
            # Recombine and sort by frame index, then by occupant ID
            df_final = pd.concat(processed_dfs).sort_values(by=["index", "id"]).reset_index(drop=True)
            
            # Order the columns exactly as the C++ logic
            final_columns = ["index", "id", "age", "gender", "score_gender", "age_mean", "gender_mean", "score_gender_mean", "bbox_cx"]
            df_final[final_columns].to_csv(os.path.join(output_path, f"{f_root}_out_age.csv"), index=False)

            pbar_files.update(1)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser("C++ Replicator (Long Format): Age/Gender ONNX")
    parser.add_argument("-c", "--config_file_path", required=True, help="Path to your internal YAML/JSON config file")
    parser.add_argument("--output_dir", required=True, help="Directory where results will be saved")
    parser.add_argument("--video_dir", required=True, help="Directory containing input .mp4 files")
    parser.add_argument("--csv_dir", required=True, help="Directory containing input .csv tracking files")
    args = parser.parse_args()

    config = get_config()
    config.set_config(args.config_file_path)

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args.video_dir, args.csv_dir, args.output_dir, config)