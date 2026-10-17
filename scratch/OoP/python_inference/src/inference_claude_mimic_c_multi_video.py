"""Run posture model inference on multiple video files (auto-detects tracking CSV) and export predictions.

Ce script imite le pipeline C++ embarqué (vérifié ligne à ligne contre
foms_positions.cpp et samples_positions_evaluation.cpp) : 
- Crop avec marge de 10% en largeur, letterbox proportionnel (c_lround)
- Traitement par lot (dossier contenant plusieurs vidéos)
- LISSAGE TEMPOREL STRICT (Sliding Window de 20 frames, avec gel lors des pertes de tracking)
"""

# Ne jamais supprimer les commentaires personnels du code.
# Pour tous mes codes, j'utilise des environnements Conda.
# This model is working with both fp32 and int8 , it can give you really close results to 
# the C++ output but some really similar that are not exactly the same too 

import os
import sys
import glob
import math
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from fire import Fire
from collections import deque
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Remonte à la racine source_code puis redescend dans training pour trouver 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../training')))
from src.utils.config import get_config
from src.posture.train_posture import load_model

F32 = np.float32

# Ordre canonique des classes
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

BBOX_COLS = (
    '/trackingState/occupants/0/boundingBox/upperLeft/0',
    '/trackingState/occupants/0/boundingBox/upperLeft/1',
    '/trackingState/occupants/0/boundingBox/width',
    '/trackingState/occupants/0/boundingBox/height',
)

STRICT_CROP_VALIDATION = False

# === CONSTANTES CALQUÉES SUR LE C++ ===
BUFFER_SIZE = 20

def c_lround(x):
    """Équivalent de std::lround."""
    x = float(x)
    return int(math.copysign(math.floor(abs(x) + 0.5), x))

def to_letterbox(v, offset, scale, pad):
    """Même calcul que drawBodyKeypointsonImage côté C++."""
    return c_lround(F32(v - offset) * F32(scale) + F32(pad))

def invalid_row(frame_idx, cpp_class_order=None):
    if cpp_class_order is None: cpp_class_order = CPP_CLASS_ORDER
    d = {"index": frame_idx}
    d.update({c: -1.0 for c in cpp_class_order})
    return d

def compute_crop_like_cpp(row, img_w, img_h):
    vals = [row.get(c, np.nan) for c in BBOX_COLS]
    if any(pd.isna(v) for v in vals): return None
    x, y, w, h = (int(float(v)) for v in vals)
    if w <= 0 or h <= 0: return None
    if STRICT_CROP_VALIDATION and not (x >= 0 and y >= 0 and x + w <= img_w and y + h <= img_h): return None

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    delta_width = int(F32(0.1) * F32(x2 - x1))
    x1_margin = max(0, x1 - delta_width)
    x2_margin = min(img_w - 1, x2 + delta_width)

    if not (x2_margin > x1_margin and y2 > y1): return None
    return x1, y1, x2, y2, x1_margin, x2_margin

def _compute_kpi_for_dataframe(df_merged, sequence_name, threshold, cpp_class_order):
    kpi_rows = []
    for cls in cpp_class_order:
        col_gt = f"{cls}_gt"
        if col_gt not in df_merged.columns or cls not in df_merged.columns: continue
        gt_full, pred_full = df_merged[col_gt], df_merged[cls]
        valid_mask = (gt_full != -1) & (pred_full != -1.0)
        gt_valid, pred_probs = gt_full[valid_mask], pred_full[valid_mask]
        if len(gt_valid) == 0: continue
        y_pred = (pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(gt_valid, y_pred, labels=[0, 1]).ravel()
        kpi_rows.append({
            "Sequence": sequence_name, "Class": cls,
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Precision": round(precision_score(gt_valid, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(gt_valid, y_pred, zero_division=0), 4),
            "F1_Score": round(f1_score(gt_valid, y_pred, zero_division=0), 4),
            "Total_Frames": len(gt_valid),
        })
    return kpi_rows

def compute_kpis_from_merged_dict(merged_dfs_dict, threshold=0.5, cpp_class_order=None):
    """Calcule les KPI par vidéo, puis ajoute une métrique GLOBALE."""
    if cpp_class_order is None: cpp_class_order = CPP_CLASS_ORDER
    all_kpi_rows = []
    for seq_name, df_merged in merged_dfs_dict.items():
        all_kpi_rows.extend(_compute_kpi_for_dataframe(df_merged, seq_name, threshold, cpp_class_order))
    if merged_dfs_dict:
        global_df = pd.concat(list(merged_dfs_dict.values()), ignore_index=True)
        all_kpi_rows.extend(_compute_kpi_for_dataframe(global_df, "GLOBAL", threshold, cpp_class_order))
    return pd.DataFrame(all_kpi_rows)

def run_video_inference(model_path, video_path, with_tatoo_keypoints, output_csv, model, session, dlr_model, device, input_names, output_name, config, cpp_class_order, class_names):
    video_base_path = os.path.splitext(video_path)[0]
    input_csv = f"{video_base_path}.csv"
    
    if not os.path.isfile(input_csv): 
        print(f"⚠️ CSV de tracking introuvable pour {os.path.basename(video_path)}. Ignoré.")
        return None

    df_input = pd.read_csv(input_csv)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_height, img_width = 256, 192
    results = []
    is_dlr, is_onnx = dlr_model is not None, session is not None

    print(f"\n🎥 Traitement de la vidéo : {os.path.basename(video_path)} ({total_frames} frames)")

    # === INITIALISATION DU BUFFER TEMP ===
    # Rempli de zéros au démarrage comme le C++
    temporal_buffers = {cls: deque([0.0]*BUFFER_SIZE, maxlen=BUFFER_SIZE) for cls in cpp_class_order}

    frame_idx = 0
    max_iterations = min(total_frames, len(df_input))
    pbar = tqdm(total=max_iterations, colour="blue", desc=f"Inférence {os.path.basename(video_path)}")
    
    with torch.no_grad():
        while frame_idx < max_iterations:
            ret, frame = cap.read()
            if not ret: break 
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            row, crop = None, None
            if frame_idx < len(df_input):
                row = df_input.iloc[frame_idx]
                crop = compute_crop_like_cpp(row, image_gray.shape[1], image_gray.shape[0])

            # === GESTION DE LA PERTE DE TRACKING ===
            if crop is None:
                results.append(invalid_row(frame_idx))
                # 🛑 AUCUNE RÉINITIALISATION DU BUFFER ICI : le C++ gèle la mémoire.
                frame_idx += 1
                pbar.update(1)
                continue 

            x1, y1, x2, y2, x1_margin, x2_margin = crop
            roi_img = image_gray[y1:y2, x1_margin:x2_margin]
            roi_height, roi_width = roi_img.shape[:2]

            if roi_width == img_width and roi_height == img_height:
                final_img = np.ascontiguousarray(roi_img)
                scale, left, bottom = F32(1.0), 0, 0
            else:
                scale = min(F32(img_width) / F32(roi_width), F32(img_height) / F32(roi_height))
                dst_cols, dst_rows = c_lround(F32(roi_width) * scale), c_lround(F32(roi_height) * scale)

                if dst_cols <= 0 or dst_rows <= 0:
                    results.append(invalid_row(frame_idx))
                    frame_idx += 1
                    pbar.update(1)
                    continue

                resized_img = cv2.resize(roi_img, (dst_cols, dst_rows), interpolation=cv2.INTER_LINEAR)
                top = (img_height - dst_rows) // 2
                bottom = img_height - top - dst_rows
                right = (img_width - dst_cols) // 2
                left = img_width - right - dst_cols
                final_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

            if with_tatoo_keypoints:
                keyPointsLines = [(5, 6), (7, 5), (8, 6), (9, 7), (10, 8), (11, 5), (12, 6), (12, 11), (12, 14), (11, 13)]
                for kp1_idx, kp2_idx in keyPointsLines:
                    c1_x = f'/trackingState/occupants/0/bodyKeyPoints/{kp1_idx}/value/0'
                    c1_y = f'/trackingState/occupants/0/bodyKeyPoints/{kp1_idx}/value/1'
                    c1_q = f'/trackingState/occupants/0/bodyKeyPoints/{kp1_idx}/quality'
                    c2_x = f'/trackingState/occupants/0/bodyKeyPoints/{kp2_idx}/value/0'
                    c2_y = f'/trackingState/occupants/0/bodyKeyPoints/{kp2_idx}/value/1'
                    c2_q = f'/trackingState/occupants/0/bodyKeyPoints/{kp2_idx}/quality'

                    kp1_x, kp1_y = float(row.get(c1_x, np.nan)), float(row.get(c1_y, np.nan))
                    kp2_x, kp2_y = float(row.get(c2_x, np.nan)), float(row.get(c2_y, np.nan))
                    
                    if not (np.isnan(kp1_x) or np.isnan(kp1_y) or np.isnan(kp2_x) or np.isnan(kp2_y)):
                        if float(row.get(c1_q, 0.0)) >= 0.1 and float(row.get(c2_q, 0.0)) >= 0.1:
                            p1x = to_letterbox(kp1_x, x1_margin, scale, left)
                            p1y = to_letterbox(kp1_y, y1, scale, bottom)
                            p2x = to_letterbox(kp2_x, x1_margin, scale, left)
                            p2y = to_letterbox(kp2_y, y1, scale, bottom)
                            if (0 <= p1x < img_width and 0 <= p1y < img_height and 0 <= p2x < img_width and 0 <= p2y < img_height):
                                cv2.line(final_img, (p1x, p1y), (p2x, p2y), 255, 3)

            img_tensor_np = final_img.astype(np.uint8) if is_dlr else final_img.astype(np.float32) * 0.00392156862
            img_tensor_np = np.expand_dims(np.expand_dims(img_tensor_np, axis=0), axis=0) 
            
            if is_dlr:
                pred_raw = np.squeeze(dlr_model.run({input_names[0]: img_tensor_np})[0])
                pos_probs = 1.0 / (1.0 + np.exp(-pred_raw)) 
            elif is_onnx:
                pred_raw = np.squeeze(session.run([output_name], {input_names[0]: img_tensor_np})[0])
                pos_probs = 1.0 / (1.0 + np.exp(-pred_raw)) 
            else:
                pred = model(img_tensor.to(device))
                pos_probs = np.squeeze(torch.sigmoid(pred).cpu().numpy())

            if pos_probs.ndim == 0: pos_probs = np.array([pos_probs])
            full_probs = [1.0 - np.clip(np.max(pos_probs), 0.0, 1.0)] + pos_probs.tolist()

            frame_data = {"index": frame_idx}
            
            for class_name_from_model, prob in zip(class_names, full_probs):
                clean_name = class_name_from_model.lower().replace("_", "").replace(" ", "")
                final_name = "closetoairbag" if clean_name == "tooclosetoairbag" else clean_name
                
                if final_name in cpp_class_order:
                    # 1. Pousser la probabilité ML
                    temporal_buffers[final_name].append(float(prob))
                    
                    # 2. Calculer et sauvegarder la moyenne glissante (Division stricte par 20)
                    frame_data[final_name] = sum(temporal_buffers[final_name]) / float(BUFFER_SIZE)

            for cpp_col in cpp_class_order:
                if cpp_col not in frame_data: frame_data[cpp_col] = -1.0

            results.append(frame_data)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    pbar.close()

    df_results = pd.DataFrame(results)[["index"] + cpp_class_order]
    df_results.to_csv(output_csv, index=False)
    print(f"  -> Prédictions sauvegardées : {output_csv}")
    return df_results

def main(model_path, videos_dir, with_tatoo_keypoints, output_dir=None, config_path=None,
         gt_csv_dir=None, kpi_csv_path=None, kpi_threshold=0.5):
    
    # Sécurisation du seuil contre les erreurs de typage
    try:
        kpi_threshold = float(kpi_threshold)
    except (ValueError, TypeError):
        kpi_threshold = 0.5
        
    if isinstance(with_tatoo_keypoints, str):
        with_tatoo_keypoints = with_tatoo_keypoints.lower() in ['true', '1', 'yes']

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
    
    is_dlr = os.path.isdir(model_path)
    is_onnx = model_path.endswith(".onnx")
    model = session = dlr_model = input_names = output_name = None
    
    if is_dlr:
        import dlr
        print(f"⚙️ Chargement du modèle quantifié DLR : {model_path}")
        dlr_model = dlr.DLRModel(model_path, 'cpu')
        input_names = dlr_model.get_input_names()
    elif is_onnx:
        import onnxruntime as ort
        print(f"⚙️ Chargement du modèle ONNX : {model_path}")
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_names = [inp.name for inp in session.get_inputs()]
        output_name = session.get_outputs()[0].name
    else:
        print(f"⚙️ Chargement du modèle PyTorch : {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        trained_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(trained_state_dict, "module.")
        model = load_model(config)
        model.load_state_dict(trained_state_dict)
        model.to(device)
        model.eval()

    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    
    if not video_files:
        print(f"❌ Aucune vidéo .mp4 trouvée dans : {videos_dir}")
        return

    print(f"🚀 Début du traitement : {len(video_files)} vidéo(s) trouvée(s).")
    
    merged_dfs_dict = {}

    for video_path in sorted(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_csv = os.path.join(output_dir, f"{video_name}_predictions.csv")
        
        df_pred = run_video_inference(
            model_path, video_path, with_tatoo_keypoints, output_csv, 
            model, session, dlr_model, device, input_names, output_name, 
            config, CPP_CLASS_ORDER, class_names
        )

        if df_pred is not None and gt_csv_dir is not None:
            gt_path_1 = os.path.join(gt_csv_dir, f"{video_name}_eval_with_gt.csv")
            gt_path_2 = os.path.join(gt_csv_dir, f"{video_name}.csv") 
            gt_csv_path = gt_path_1 if os.path.isfile(gt_path_1) else (gt_path_2 if os.path.isfile(gt_path_2) else None)
            
            if gt_csv_path:
                df_gt = pd.read_csv(gt_csv_path)
                df_gt = df_gt[[c for c in df_gt.columns if c == "index" or str(c).endswith("_gt")]]
                merged_dfs_dict[video_name] = pd.merge(df_pred, df_gt, on="index", how="inner")
            else:
                print(f"⚠️ Vérité terrain introuvable pour {video_name}. Ignoré pour le calcul des KPI.")

    if gt_csv_dir is not None and merged_dfs_dict:
        print(f"\n📊 Calcul des KPI globaux (seuil={kpi_threshold})...")
        df_kpi = compute_kpis_from_merged_dict(merged_dfs_dict, threshold=kpi_threshold, cpp_class_order=CPP_CLASS_ORDER)
        is_global = df_kpi['Sequence'] == 'GLOBAL'
        df_kpi = pd.concat([df_kpi[~is_global].sort_values(['Sequence', 'Class']), df_kpi[is_global].sort_values(['Class'])]).reset_index(drop=True)

        if kpi_csv_path is None: 
            kpi_csv_path = os.path.join(output_dir, "KPI_summary_videos.csv")
        df_kpi.to_csv(kpi_csv_path, index=False)
        print(f"\n✅ Tous les KPI ont été exportés (CSV) : {kpi_csv_path}")

if __name__ == "__main__":
    Fire(main)