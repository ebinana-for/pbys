"""Run posture model inference on a video file (auto-detects tracking CSV) and export predictions."""

# This model is working with both fp32 and int8 , it can give you really close results to 
# the C++ output but some really similar that are not exactly the same too 

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from fire import Fire

# Remonte à la racine source_code puis redescend dans training pour trouver 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../training')))
from src.utils.config import get_config
from src.posture.train_posture import load_model


def run_video_inference(model_path, video_path, with_tatoo_keypoints, output_csv=None, config_path=None):
    """
    Run inference on a video frame by frame. Automatically loads the associated 
    tracking CSV to crop the person, and exports posture predictions.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle ou dossier d'artefacts introuvable : {model_path}")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    # --- DÉDUCTION AUTOMATIQUE DU CSV D'ENTRÉE ---
    video_base_path = os.path.splitext(video_path)[0]
    input_csv = f"{video_base_path}.csv"
    
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"CSV de tracking introuvable. Le script cherchait : {input_csv}")

    if output_csv is None:
        output_csv = f"{video_base_path}_predictions.csv"
    
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- 1. CHARGEMENT DE LA CONFIGURATION ET DES CLASSES ---
    if config_path is None:
        inference_folder = os.path.dirname(model_path.rstrip('/')) if os.path.isdir(model_path) else os.path.dirname(model_path)
        config_path = os.path.join(inference_folder, "../config.yaml")
    assert os.path.isfile(config_path), f"Config file not found: {config_path}"

    config = get_config()
    config.set_config(config_path)

    num_positive_labels = len(config.input.positive_labels)
    class_names = ["correctly_seated"] + list(config.input.positive_labels)
    
    cpp_class_order = [
        "closetoairbag",
        "legsondashboard",
        "torsodissimetry",           
        "reachingback",
        "lateralbodycontact",
        "oneshoulderincontact",
        "pelvisrotation",
        "pelvisdrift",
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    is_dlr = os.path.isdir(model_path)
    is_onnx = model_path.endswith(".onnx")
    
    # --- 2. CHARGEMENT DU MODÈLE ---
    model = None
    session = None
    dlr_model = None
    
    if is_dlr:
        import dlr
        print(f"Chargement du modèle quantifié DLR (INT8) depuis : {model_path}")
        dlr_model = dlr.DLRModel(model_path, 'cpu')
        input_names = dlr_model.get_input_names()
    elif is_onnx:
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        input_names = [inp.name for inp in session.get_inputs()]
        output_name = session.get_outputs()[0].name
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        trained_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(trained_state_dict, "module.")
        model = load_model(config)
        model.load_state_dict(trained_state_dict)
        model.to(device)
        model.eval()

    # --- 3. PRÉPARATION DE LA VIDÉO ET DU CSV D'ENTRÉE ---
    df_input = pd.read_csv(input_csv)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Vidéo chargée : {video_path} ({total_frames} frames, {fps} FPS)")
    print(f"Tracking CSV chargé : {input_csv} ({len(df_input)} lignes)")
    print(f"Mode Tatouage Visuel (Squelette) : {'ACTIVÉ' if with_tatoo_keypoints else 'DÉSACTIVÉ'}")
    
    img_height = 256
    img_width = 192

    results = []
    
    # --- 4. BOUCLE D'INFÉRENCE FRAME PAR FRAME ---
    frame_idx = 0
    max_iterations = min(total_frames, len(df_input))
    pbar = tqdm(total=max_iterations, colour="blue", desc="Inférence Vidéo")
    
    with torch.no_grad():
        while frame_idx < max_iterations:
            ret, frame = cap.read()
            if not ret:
                break 

            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row = df_input.iloc[frame_idx]
            
            val_x = row.get('/trackingState/occupants/0/boundingBox/upperLeft/0', 0)
            val_y = row.get('/trackingState/occupants/0/boundingBox/upperLeft/1', 0)
            val_w = row.get('/trackingState/occupants/0/boundingBox/width', image_gray.shape[1])
            val_h = row.get('/trackingState/occupants/0/boundingBox/height', image_gray.shape[0])

            if pd.isna(val_x) or pd.isna(val_y) or pd.isna(val_w) or pd.isna(val_h):
                frame_data = {"index": frame_idx}
                for class_name in cpp_class_order:
                    frame_data[class_name] = -1.0
                results.append(frame_data)
                frame_idx += 1
                pbar.update(1)
                continue 

            x = int(float(val_x))
            y = int(float(val_y))
            w = int(float(val_w))
            h = int(float(val_h))

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

            # --- DESSIN DU SQUELETTE (Conditionné par l'argument obligatoire) ---
            if with_tatoo_keypoints:
                keyPointsLines = [
                    (5, 6), (7, 5), (8, 6), (9, 7), (10, 8),
                    (11, 5), (12, 6), (12, 11), (12, 14), (11, 13)
                ]
                
                c_plus_plus_y_offset = bottom 
                
                for kp1_idx, kp2_idx in keyPointsLines:
                    c1_x = f'/trackingState/occupants/0/bodyKeyPoints/{kp1_idx}/value/0'
                    c1_y = f'/trackingState/occupants/0/bodyKeyPoints/{kp1_idx}/value/1'
                    c1_q = f'/trackingState/occupants/0/bodyKeyPoints/{kp1_idx}/quality'
                    
                    c2_x = f'/trackingState/occupants/0/bodyKeyPoints/{kp2_idx}/value/0'
                    c2_y = f'/trackingState/occupants/0/bodyKeyPoints/{kp2_idx}/value/1'
                    c2_q = f'/trackingState/occupants/0/bodyKeyPoints/{kp2_idx}/quality'

                    val1_x = row.get(c1_x, np.nan)
                    val1_y = row.get(c1_y, np.nan)
                    val2_x = row.get(c2_x, np.nan)
                    val2_y = row.get(c2_y, np.nan)
                    
                    kp1_x = float(int(float(row.get(c1_x, np.nan)))) if not pd.isna(row.get(c1_x)) else np.nan
                    kp1_y = float(int(float(row.get(c1_y, np.nan)))) if not pd.isna(row.get(c1_y)) else np.nan
                    kp1_q = float(row.get(c1_q, 0.0))
                    
                    kp2_x = float(int(float(val2_x))) if not pd.isna(val2_x) else np.nan
                    kp2_y = float(int(float(val2_y))) if not pd.isna(val2_y) else np.nan
                    kp2_q = float(row.get(c2_q, 0.0))
                    
                    if not (np.isnan(kp1_x) or np.isnan(kp1_y) or np.isnan(kp2_x) or np.isnan(kp2_y)):
                        if kp1_q >= 0.1 and kp2_q >= 0.1:
                            p1x = int(round((kp1_x - x1_margin) * scale + left))
                            p1y = int(round((kp1_y - y1) * scale + c_plus_plus_y_offset))
                            p2x = int(round((kp2_x - x1_margin) * scale + left))
                            p2y = int(round((kp2_y - y1) * scale + c_plus_plus_y_offset))
                            
                            if (0 <= p1x < img_width and 0 <= p1y < img_height and 
                                0 <= p2x < img_width and 0 <= p2y < img_height):
                                cv2.line(final_img, (p1x, p1y), (p2x, p2y), 255, 3)

            ## --- CRÉATION DU TENSEUR FAÇON C++ + NORMALISATION ---
            if is_dlr:
                img_tensor_np = final_img.astype(np.uint8)
            else:
                img_tensor_np = final_img.astype(np.float32) * 0.00392156862
            
            img_tensor_np = np.expand_dims(img_tensor_np, axis=0) 
            img_tensor_np = np.expand_dims(img_tensor_np, axis=0) 
            img_tensor = torch.from_numpy(img_tensor_np) 
            
            # --- INFÉRENCE ---
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
            pred_class_idx = np.argmax(full_probs)
            predicted_class = class_names[pred_class_idx]
            confidence_score = full_probs[pred_class_idx]

            frame_data = {"index": frame_idx}
            
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
    final_columns = ["index"] + cpp_class_order
    df_results = df_results[final_columns]
    
    df_results.to_csv(output_csv, index=False)
    print(f"\n✅ Inférence terminée ! Les prédictions ont été sauvegardées dans : {output_csv}")


def main(model_path, video_path, with_tatoo_keypoints, output_csv=None, config_path=None):
    """
    Lance l'inférence d'un modèle de posture (PyTorch, ONNX, DLR) en imitant 
    exactement le pipeline de production embarqué C++ (letterboxing strict).

    Args:
        model_path (str): Chemin vers le modèle (.onnx, .pt, ou dossier DLR).
        video_path (str): Chemin vers la vidéo d'entrée (.mp4). Le CSV de tracking associé doit s'y trouver.
        with_tatoo_keypoints (bool): [OBLIGATOIRE] Indique s'il faut tracer le squelette sur l'image 
                                     avant l'inférence (True) ou garder l'image pure (False).
        output_csv (str, optional): Chemin personnalisé pour le CSV de sortie.
        config_path (str, optional): Chemin vers le fichier config.yaml.
    """
    # Force la conversion en booléen si passé en chaîne par la CLI
    if isinstance(with_tatoo_keypoints, str):
        with_tatoo_keypoints = with_tatoo_keypoints.lower() in ['true', '1', 'yes']

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run_video_inference(model_path, video_path, with_tatoo_keypoints, output_csv, config_path)


if __name__ == "__main__":
    Fire(main)