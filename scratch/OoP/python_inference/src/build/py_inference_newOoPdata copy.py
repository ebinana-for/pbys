"""Run posture model inference on multiple video files (auto-detects tracking CSV) and export predictions.

Ce script imite le pipeline C++ embarqué (vérifié ligne à ligne contre
foms_positions.cpp et samples_positions_evaluation.cpp) : crop avec marge de
10% en largeur uniquement, letterbox proportionnel, padding noir centré, interpolation INTER_LINEAR.

Modifié pour supporter :
- Le traitement par lot (dossier contenant plusieurs vidéos)
- Le calcul de KPI par vidéo via le fichier annotations.csv centralisé
- Le calcul d'une métrique GLOBALE finale via concaténation
- L'inférence sur de MULTIPLES OCCUPANTS (Mimic C++ Array of Occupants)
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

# Remonte à la racine du dépôt puis redescend dans training pour trouver 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../training')))
from src.utils.config import get_config
from src.posture.train_posture import load_model

# Pour tous mes codes, j'utilise des environnements Conda.
# Never remove my personal comments from the code.

# Ordre canonique des classes, réutilisé pour l'export prédictions ET les KPI.
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
    """Fonction utilitaire pour calculer les KPIs sur un DataFrame mergé"""
    kpi_rows = []
    for cls in cpp_class_order:
        col_gt = f"{cls}_gt"
        if col_gt not in df_merged.columns or cls not in df_merged.columns:
            continue

        gt_full = df_merged[col_gt]
        pred_full = df_merged[cls]

        # Frames invalides marquées par -1, exclues indépendamment pour chaque classe.
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
    Calcule les KPI (TP/TN/FP/FN/Precision/Recall/F1) par classe et par vidéo, 
    puis ajoute une métrique GLOBALE sur la concaténation de toutes les vidéos.
    """
    if cpp_class_order is None:
        cpp_class_order = CPP_CLASS_ORDER

    all_kpi_rows = []
    
    # 1. Calculs par vidéo
    for seq_name, df_merged in merged_dfs_dict.items():
        all_kpi_rows.extend(
            _compute_kpi_for_dataframe(df_merged, seq_name, threshold, cpp_class_order)
        )

    # 2. Calcul GLOBALE
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
        print(f"⚠️ CSV de tracking introuvable pour {os.path.basename(video_path)} à l'emplacement : {tracking_csv}. Ignoré.")
        return None

    try:
        df_input = pd.read_csv(tracking_csv)
    except pd.errors.EmptyDataError:
        print(f"⚠️ CSV de tracking VIDE pour {os.path.basename(video_path)}. Vidéo ignorée.")
        return None

    # --- DÉTECTION DYNAMIQUE DES OCCUPANTS ---
    # On scanne les colonnes pour trouver tous les ID (ex: 0 pour Driver, 1 pour Passager)
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
        print(f"⚠️ Aucun occupant trouvé dans le header du CSV de tracking pour {os.path.basename(video_path)}.")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n🎥 Traitement de la vidéo : {os.path.basename(video_path)} ({total_frames} frames)")
    print(f"👥 Occupants détectés : {occupant_ids}")
    
    img_height = 256
    img_width = 192
    results = []
    
    is_dlr = dlr_model is not None
    is_onnx = session is not None

    frame_idx = 0
    max_iterations = min(total_frames, len(df_input))
    pbar = tqdm(total=max_iterations, colour="blue", desc="Inférence")
    
    with torch.no_grad():
        while frame_idx < max_iterations:
            ret, frame = cap.read()
            if not ret:
                break 

            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row = df_input.iloc[frame_idx]
            
            # --- BOUCLE C++ : ITÉRATION SUR CHAQUE OCCUPANT ---
            for occ_id in occupant_ids:
                # Création dynamique des chemins d'accès CSV pour l'occupant en cours
                col_x = f'/trackingState/occupants/{occ_id}/boundingBox/upperLeft/0'
                col_y = f'/trackingState/occupants/{occ_id}/boundingBox/upperLeft/1'
                col_w = f'/trackingState/occupants/{occ_id}/boundingBox/width'
                col_h = f'/trackingState/occupants/{occ_id}/boundingBox/height'

                val_x = row.get(col_x, pd.NA)
                val_y = row.get(col_y, pd.NA)
                val_w = row.get(col_w, pd.NA)
                val_h = row.get(col_h, pd.NA)

                # Dictionnaire de base pour cette frame ET cet occupant
                frame_data = {"index": frame_idx, "occupant_id": occ_id}

                # Gestion de la perte de tracking (garde-fou)
                if pd.isna(val_x) or pd.isna(val_y) or pd.isna(val_w) or pd.isna(val_h):
                    for class_name in cpp_class_order:
                        frame_data[class_name] = -1.0
                    results.append(frame_data)
                    continue 

                x = int(float(val_x))
                y = int(float(val_y))
                w = int(float(val_w))
                h = int(float(val_h))

                # Le "Guard" géométrique
                if w <= 0 or h <= 0 or x < 0 or y < 0:
                    for class_name in cpp_class_order:
                        frame_data[class_name] = -1.0
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
    final_columns = ["index", "occupant_id"] + cpp_class_order
    df_results = df_results[final_columns]
    
    df_results.to_csv(output_csv, index=False)
    print(f"  -> Prédictions sauvegardées : {output_csv}")

    return df_results


def main(model_path, videos_dir, output_dir=None, config_path=None,
         annotations_csv=None, kpi_csv_path=None, kpi_threshold=0.5):
    """
    Lance l'inférence d'un modèle de posture sur un lot de vidéos.

    Args:
        model_path (str): Chemin vers le modèle (.onnx, .pt, ou dossier DLR).
        videos_dir (str): Dossier contenant les vidéos d'entrée (.mp4).
        output_dir (str, optional): Dossier de sortie pour les CSV de prédictions individuels.
        config_path (str, optional): Chemin vers le fichier config.yaml.
        annotations_csv (str, optional): Fichier de vérité terrain centralisé (annotations.csv).
        kpi_csv_path (str, optional): Chemin du fichier récapitulatif des KPI (CSV). 
                                       Par défaut : <output_dir>/KPI_summary_videos.csv
        kpi_threshold (float): Seuil de binarisation pour les KPI.
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
    
    # --- CHARGEMENT DU MODÈLE (UNE SEULE FOIS POUR TOUTES LES VIDÉOS) ---
    is_dlr = os.path.isdir(model_path)
    is_onnx = model_path.endswith(".onnx")
    
    model = session = dlr_model = None
    input_names = output_name = None
    
    if is_dlr:
        import dlr
        print(f"⚙️ Chargement du modèle quantifié DLR : {model_path}")
        dlr_model = dlr.DLRModel(model_path, 'cpu')
        input_names = dlr_model.get_input_names()
    elif is_onnx:
        import onnxruntime as ort
        print(f"⚙️ Chargement du modèle ONNX : {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
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

    # --- CHARGEMENT DU FICHIER ANNOTATIONS CENTRALISÉ ---
    df_annotations = None
    if annotations_csv and os.path.isfile(annotations_csv):
        print(f"📊 Chargement du fichier d'annotations : {annotations_csv}")
        df_annotations = pd.read_csv(annotations_csv, sep=None, engine='python', encoding='latin1', header=2)

    # --- TRAITEMENT DU LOT DE VIDÉOS ---
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    if not video_files:
        print(f"❌ Aucune vidéo .mp4 trouvée dans : {videos_dir}")
        return

    print(f"🚀 Début du traitement : {len(video_files)} vidéo(s) trouvée(s).")
    
    # Déduire le dossier 'tracking' (au même niveau que le dossier 'mp4')
    parent_dir = os.path.dirname(videos_dir.rstrip('/\\'))
    tracking_dir = os.path.join(parent_dir, "tracking")

    merged_dfs_dict = {}

    for video_path in sorted(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_filename = os.path.basename(video_path)
        output_csv = os.path.join(output_dir, f"{video_name}_predictions.csv")
        
        # Le chemin exact du fichier de bbox dans le dossier tracking
        tracking_csv = os.path.join(tracking_dir, f"{video_name}.csv")
        
        # --- MÉCANISME DE REPRISE (RESUME) ---
        if os.path.exists(output_csv):
            print(f"⏩ Vidéo déjà traitée : {video_name}. Chargement des prédictions existantes...")
            try:
                df_pred = pd.read_csv(output_csv)
            except pd.errors.EmptyDataError:
                print(f"⚠️ Prédictions corrompues pour {video_name}, on relance l'inférence...")
                df_pred = run_video_inference(
                    model_path, video_path, tracking_csv, output_csv, 
                    model, session, dlr_model, device, input_names, output_name, 
                    config, CPP_CLASS_ORDER, class_names
                )
        else:
            # Lancer l'inférence normale
            df_pred = run_video_inference(
                model_path, video_path, tracking_csv, output_csv, 
                model, session, dlr_model, device, input_names, output_name, 
                config, CPP_CLASS_ORDER, class_names
            )
        # -------------------------------------

        if df_pred is not None and df_annotations is not None:
            # NOTE : Le calcul des KPI n'est pas fait ici car l'évaluation
            # est maintenant gérée par le script evaluate_kpis.py séparé.
            pass

    # --- CALCUL FINAL DES KPI ---
    if df_annotations is not None:
        print("\nℹ️ L'inférence est terminée. Veuillez utiliser le script `evaluate_kpis.py` pour évaluer les prédictions.")


if __name__ == "__main__":
    Fire(main)