"""This script allows for a quantitative evaluation of age & gender prediction
on a non-public dataset using an ONNX model. It outputs one CSV file per video, 
formatted identically to the C++ estimator.
"""

import argparse
import ast
import datetime
import logging
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import onnxruntime as ort

from src.utils.viewer_utils import get_face_bbox
from src.utils.config import get_config

# Fonction utilitaire d'activation softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def evaluate(input_path=None, output_path=None, config=None):
    # 1. Configuration des augmentations (Albumentations et Custom)
    extract_r_channel = False
    
    if config is not None:
        albumentation_ops = config["input"]["data_augmentations"]["test"]["pipeline"].get("albumentation_ops", {})
        p = config["input"]["data_augmentations"]["test"]["pipeline"].get("p", 1.0)
        
        custom_ops = config["input"]["data_augmentations"]["test"]["pipeline"].get("custom_ops", {})
        if "ExtractRChannel" in custom_ops:
            extract_r_channel = True

        list_data_aug = []
        for albumentation_op, params in albumentation_ops.items():
            if albumentation_op in A.__dict__:
                params2 = params.copy()
                for key in params2:
                    if isinstance(params2[key], str):
                        params2[key] = ast.literal_eval(params2[key])
                list_data_aug.append(A.__dict__[albumentation_op](**params2))

        albumentation_pipeline = A.Compose(list_data_aug, p=p) if list_data_aug else None
        
        try:
            center_reduce = config.model.heads_params.age.center_reduce
        except AttributeError:
            center_reduce = False
    else:
        albumentation_pipeline = None
        center_reduce = False

    # 2. Initialisation de la session ONNX via le yaml
    onnx_model_path = config.model.checkpoint
    logging.info(f"Chargement du modèle ONNX : {onnx_model_path}")
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Récupération des fichiers vidéos (vérifie que le mp4 ET le csv existent)
    all_files = os.listdir(input_path)
    mp4_files = [f for f in all_files if f.endswith('.mp4')]
    
    files_to_process = []
    for f in mp4_files:
        f_root = os.path.splitext(f)[0]
        if f_root + ".csv" in all_files:
            files_to_process.append(f_root)
        else:
            logging.warning(f"Fichier ignoré : {f} (aucun .csv correspondant trouvé)")
            
    files_to_process = sorted(list(set(files_to_process)))

    with tqdm(total=len(files_to_process), desc="Processing Files") as pbar_files:
        for f_root in files_to_process:
            
            # Extraction Ground Truth (Format prenom_nom_age_genre...)
            try:
                parts = f_root.split("_")
                gt_age = int(parts[2])
                
                gt_gender_str = parts[3].upper()
                if gt_gender_str in ["M", "F"]:
                    gt_gender = {"M": 0, "F": 1}[gt_gender_str]
                else:
                    gt_gender = -1 
                    
            except (IndexError, ValueError, KeyError):
                logging.warning(f"\nImpossible d'extraire la GT de '{f_root}'. Valeurs -1 appliquées.")
                gt_age = -1
                gt_gender = -1

            video_path = os.path.join(input_path, f_root + ".mp4")
            cap = cv2.VideoCapture(video_path)
            
            csv_path = os.path.join(input_path, f_root + ".csv")
            df = pd.read_csv(csv_path, low_memory=False)

            if len(df) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                logging.warning(f"Length mismatch for {f_root}.")
            else:
                count = 0
                file_results = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    head_crop0, head_crop_quality0, _, bbox_0 = get_face_bbox(
                        df, occupant_id=0, frame=frame, frame_cnt=count
                    )

                    if head_crop0 is not None:
                        if albumentation_pipeline:
                            head_crop0 = albumentation_pipeline(image=head_crop0)["image"]
                        
                        input_image = cv2.resize(head_crop0, (config.input.input_width, config.input.input_height)) 
                        
                        if extract_r_channel:
                            if len(input_image.shape) == 3 and input_image.shape[2] >= 3:
                                input_image = input_image[:, :, 2]
                            input_image = np.expand_dims(input_image, axis=2)
                            
                        input_image = input_image.transpose((2, 0, 1)) 
                        input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0
                        
                        outputs = session.run(None, {input_name: input_image})
                        
                        if len(outputs) == 1:
                            combined_logits = outputs[0][0]
                            logits_gender = combined_logits[:2]
                            logits_age = combined_logits[2:]
                        else:
                            logits_gender = outputs[0][0]
                            logits_age = outputs[1][0]

                        gender_class = int(np.argmax(logits_gender))
                        gender = {"M": 0, "F": 1}[["M", "F"][gender_class]]
                        score_gender = float(np.max(softmax(logits_gender)))

                        if logits_age.size == 1:  
                            age_pred = logits_age[0]
                        else:  
                            age_pred = np.sum(softmax(logits_age) * np.arange(0, len(logits_age)))

                        age = age_pred if not center_reduce else age_pred * 120 + 60
                        score_age = 1.0 
                        
                        file_results.append([
                            count, int(age), gender, score_gender, score_age, gt_age, gt_gender
                        ])
                    else:
                        file_results.append([
                            count, None, None, None, None, gt_age, gt_gender
                        ])

                    count += 1

                df_file = pd.DataFrame(
                    file_results,
                    columns=["index", "age", "gender", "score_gender", "score_age", "GT_age", "GT_gender"]
                )
                
                roll_window = 20
                df_file["age_mean"] = df_file["age"].rolling(window=roll_window, min_periods=1).median()
                df_file["gender_mean"] = df_file["gender"].rolling(window=roll_window, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else None)
                df_file["score_age_mean"] = df_file["score_age"].rolling(window=roll_window, min_periods=1).mean()
                df_file["score_gender_mean"] = df_file["score_gender"].rolling(window=roll_window, min_periods=1).mean()

                # Organisation des colonnes et remplacement des None par -1
                final_columns = [
                    "index", "age", "gender", "score_gender", 
                    "age_mean", "gender_mean", "score_age_mean", "score_gender_mean", 
                    "GT_age", "GT_gender"
                ]
                df_file = df_file[final_columns].fillna(-1)
                
                # Sauvegarde d'un fichier CSV par vidéo, suffixé _out_age.csv comme en C++
                out_csv = os.path.join(output_path, f"{f_root}_out_age.csv")
                df_file.to_csv(out_csv, index=False)

            pbar_files.update(1)

    logging.info(f"Tous les fichiers individuels ont été sauvegardés dans : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ablation Study: Age/Gender ONNX")
    parser.add_argument("-c", "--config_file_path", required=True)
    parser.add_argument("--color_mode", choices=["r_channel", "grayscale"], default="r_channel", help="Test de la colorimétrie")
    parser.add_argument("--crop_mode", choices=["bbox", "keypoints"], default="bbox", help="Test du recadrage")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie forcé par le script Bash")
    args = parser.parse_args()

    config = get_config()
    config.set_config(args.config_file_path)

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(config.general.non_public_video_path, args.output_dir, config, args.color_mode, args.crop_mode)