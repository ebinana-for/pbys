"""Script d'évaluation adaptatif pour l'étude d'ablation (Color & Crop)"""

import argparse
import ast
import datetime
import logging
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import onnxruntime as ort
import math


from src.utils.viewer_utils import get_face_bbox
from src.utils.config import get_config

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_body_keypoint(df, frame_cnt, occupant_id, keypoint_number):
    """
    Simule le constructeur foms::BodyKeyPoint(CsvReader, ...)
    Lit la validité, qualité, X et Y depuis le DataFrame.
    """
    quality_col = f"/trackingState/occupants/{occupant_id}/bodyKeyPoints/{keypoint_number}/quality"
    x_col = f"/trackingState/occupants/{occupant_id}/bodyKeyPoints/{keypoint_number}/value/0"
    y_col = f"/trackingState/occupants/{occupant_id}/bodyKeyPoints/{keypoint_number}/value/1"
    
    if quality_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return {"valid": False}
        
    try:
        quality = float(df.at[frame_cnt, quality_col])
        valid = quality > 0.1 # minBodyPtQuality = 0.1F en C++
        
        if not valid:
            return {"valid": False}
            
        x = int(df.at[frame_cnt, x_col])
        y = int(df.at[frame_cnt, y_col])
        return {"valid": True, "x": x, "y": y, "quality": quality}
    except (ValueError, KeyError, TypeError):
        return {"valid": False}

def compute_face_crop_from_keypoints(df, frame_img, frame_cnt, occupant_id=0):
    """
    Traduction C++ exacte : foms::computeFaceCropFromkeyPoints (foms_utils.cpp).
    Corrigé pour correspondre à la troncature float -> int stricte du C++.
    """
    # 1. Récupération des 5 Keypoints requis
    LkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 1) # LeftEyeKeypointIndex
    RkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 2) # RightEyeKeypointIndex
    LEarkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 3) # LeftEarKeypointIndex
    REarkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 4) # RightEarKeypointIndex
    shinkeyPt = get_body_keypoint(df, frame_cnt, occupant_id, 25) # ShinKeypointsIndex (Menton)
    
    # 2. Vérification de validité basique
    if not (LkeyPt["valid"] and RkeyPt["valid"] and shinkeyPt["valid"] and REarkeyPt["valid"] and LEarkeyPt["valid"]):
        return None

    # 3. Mathématiques du crop
    interEyeDistance = math.sqrt((LkeyPt["x"] - RkeyPt["x"])**2 + (LkeyPt["y"] - RkeyPt["y"])**2)
    
    if interEyeDistance < 5.0: # minInterEyeDistance
        return None
        
    if shinkeyPt["x"] < LkeyPt["x"] or shinkeyPt["x"] > RkeyPt["x"]:
        return None
        
    if (LEarkeyPt["x"] > RkeyPt["x"] and REarkeyPt["x"] > RkeyPt["x"]) or \
       (LEarkeyPt["x"] < LkeyPt["x"] and REarkeyPt["x"] < LkeyPt["x"]):
        return None
        
    eye_to_chin_vert_distance = (shinkeyPt["y"] - (RkeyPt["y"] + LkeyPt["y"]) / 2.0)
    
    if eye_to_chin_vert_distance <= interEyeDistance * 1.5:
        return None
        
    center_x = (REarkeyPt["x"] + LEarkeyPt["x"]) / 2.0
    center_y = (LkeyPt["y"] + RkeyPt["y"]) / 2.0
    
    width = interEyeDistance * 3.2
    height = eye_to_chin_vert_distance * 2.2
    
    # --- CORRECTION MAJEURE : TRONCATURE C++ ---
    # Equivalent de static_cast<int>(val + 0.5) en C++
    xmin = int((center_x - width / 2.0) + 0.5)
    ymin = int((center_y - 0.8 * height / 2.0) + 0.5)
    xmax = int((center_x + width / 2.0) + 0.5)
    ymax = int((center_y + height / 2.0) + 0.5)
    
    # --- CORRECTION FINALE : REJET STRICT AU LIEU DU CLAMPING ---
    img_h, img_w = frame_img.shape[:2]
    
    # Si la boîte déborde de l'image, le C++ rejette la frame
    if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
        return None
        
    final_width = float(xmax - xmin)
    final_height = float(ymax - ymin)
    
    if final_height <= 0 or final_width <= 0:
        return None
        
    ratio = final_width / final_height
    
    # Validation finale des limites (minWidthFaceCrop = 130, minHeightFaceCrop = 190)
    if ratio < 0.6 or ratio > 2.0 or final_width < 130 or final_height < 190:
        return None
        
    return frame_img[ymin:ymax, xmin:xmax]

def evaluate(input_path, output_path, config, color_mode, crop_mode, resize_mode, smoothing_mode):
    
    onnx_model_path = config.model.checkpoint
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    all_files = os.listdir(input_path)
    mp4_files = [f for f in all_files if f.endswith('.mp4')]
    
    files_to_process = [os.path.splitext(f)[0] for f in mp4_files if os.path.splitext(f)[0] + ".csv" in all_files]
    files_to_process = sorted(list(set(files_to_process)))

    with tqdm(total=len(files_to_process), desc=f"Testing [Color: {color_mode} | Crop: {crop_mode}]") as pbar_files:
        for f_root in files_to_process:
            
            try:
                parts = f_root.split("_")
                gt_age = int(parts[2])
                gt_gender_str = parts[3].upper()
                gt_gender = {"M": 0, "F": 1}[gt_gender_str] if gt_gender_str in ["M", "F"] else -1 
            except (IndexError, ValueError, KeyError):
                gt_age, gt_gender = -1, -1

            cap = cv2.VideoCapture(os.path.join(input_path, f_root + ".mp4"))
            df = pd.read_csv(os.path.join(input_path, f_root + ".csv"), low_memory=False)

            count = 0
            file_results = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # --- CORRECTION : CONVERSION COULEUR AVANT LE CROP ---
                if color_mode == "grayscale":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif color_mode == "r_channel":
                    if len(frame.shape) == 3:
                        frame = frame[:, :, 2]

                # --- ABLATION : MODE DE CROP ---
                if crop_mode == "bbox":
                    head_crop0, _, _, _ = get_face_bbox(df, occupant_id=0, frame=frame, frame_cnt=count)
                elif crop_mode == "keypoints":
                    head_crop0 = compute_face_crop_from_keypoints(df, frame, count)

                if head_crop0 is not None and head_crop0.shape[0] > 0 and head_crop0.shape[1] > 0:
                    
                    # --- ABLATION : MODE DE REDIMENSIONNEMENT (GÉOMÉTRIE) ---
                    if resize_mode == "forced":
                        input_image = cv2.resize(head_crop0, (config.input.input_width, config.input.input_height)) 
                    
                    elif resize_mode == "letterbox":
                        in_cols = config.input.input_width
                        in_rows = config.input.input_height
                        roi_h, roi_w = head_crop0.shape[:2]

                        if roi_w != in_cols or roi_h != in_rows:
                            scale_x = in_cols / float(roi_w)
                            scale_y = in_rows / float(roi_h)
                            scale = min(scale_x, scale_y)
                            
                            dst_cols = int(round(roi_w * scale))
                            dst_rows = int(round(roi_h * scale))
                            
                            resized_image = cv2.resize(head_crop0, (dst_cols, dst_rows))
                            
                            top = int((in_rows - dst_rows) / 2)
                            bottom = in_rows - top - dst_rows
                            left = int((in_cols - dst_cols) / 2)
                            right = in_cols - left - dst_cols
                            
                            input_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                        else:
                            input_image = head_crop0.copy()
                                        
                    # Préparation du tenseur (l'image est déjà en 2D grâce à la conversion préalable)
                    input_image = np.expand_dims(input_image, axis=2) # (H, W, 1)
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
                    
                    # C++ utilise une troncature pure
                    file_results.append([count, int(age_pred), gender, score_gender, 1.0, gt_age, gt_gender])
                else:
                    file_results.append([count, -1, -1, -1, -1, gt_age, gt_gender])

                count += 1

            df_file = pd.DataFrame(file_results, columns=["index", "age", "gender", "score_gender", "score_age", "GT_age", "GT_gender"])
            
            # Application du lissage
            valid_mask = df_file["age"] != -1
            df_file.replace(-1, np.nan, inplace=True)
            
            # --- ABLATION : MODE DE LISSAGE ---
            if smoothing_mode == "rolling":
                df_file["age_mean"] = df_file["age"].rolling(window=20, min_periods=1).median()
                df_file["gender_mean"] = df_file["gender"].rolling(window=20, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                df_file["score_age_mean"] = df_file["score_age"].rolling(window=20, min_periods=1).mean()
                df_file["score_gender_mean"] = df_file["score_gender"].rolling(window=20, min_periods=1).mean()
            
            elif smoothing_mode == "cumulative":
                # CORRECTION : Troncature stricte int() au lieu de np.round()
                df_file["age_mean"] = df_file["age"].expanding(min_periods=1).mean().apply(lambda x: int(x) if pd.notnull(x) else np.nan)
                df_file["gender_mean"] = df_file["gender"].expanding(min_periods=1).mean().apply(lambda x: 1 if pd.notnull(x) and x > 0.5 else (0 if pd.notnull(x) else np.nan))
                df_file["score_age_mean"] = df_file["score_age"].expanding(min_periods=1).mean()
                df_file["score_gender_mean"] = df_file["score_gender"].expanding(min_periods=1).mean()
                
            # CORRECTION : On masque les moyennes calculées sur les frames vides
            df_file.loc[~valid_mask, ["age_mean", "gender_mean", "score_age_mean", "score_gender_mean"]] = np.nan
            
            df_file.fillna(-1, inplace=True)
            
            final_columns = ["index", "age", "gender", "score_gender", "age_mean", "gender_mean", "score_age_mean", "score_gender_mean", "GT_age", "GT_gender"]
            df_file[final_columns].to_csv(os.path.join(output_path, f"{f_root}_out_age.csv"), index=False)

            pbar_files.update(1)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ablation Study: Age/Gender ONNX")
    parser.add_argument("-c", "--config_file_path", required=True)
    parser.add_argument("--color_mode", choices=["r_channel", "grayscale"], default="r_channel")
    parser.add_argument("--crop_mode", choices=["bbox", "keypoints"], default="bbox")
    parser.add_argument("--resize_mode", choices=["forced", "letterbox"], default="forced")
    parser.add_argument("--smoothing_mode", choices=["rolling", "cumulative"], default="rolling") # <-- NOUVEAU
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_dir", required=True, help="Dossier contenant les vidéos et csv d'entrée")
    args = parser.parse_args()

    config = get_config()
    config.set_config(args.config_file_path)

    os.makedirs(args.output_dir, exist_ok=True)
    # N'oublie pas d'ajouter args.smoothing_mode dans l'appel :
    evaluate(args.input_dir, args.output_dir, config, args.color_mode, args.crop_mode, args.resize_mode, args.smoothing_mode)