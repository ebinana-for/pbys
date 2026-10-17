"""Run posture model inference on a video file (auto-detects tracking CSV) and export predictions.

Ce script réplique le pré-traitement du chemin PAR DÉFAUT de PostureDataset
(evaluation_inference_20_08.py avec use_cpp_letterboxing=False), à savoir :

    1. Crop sur la bounding box SANS marge de 10 % (box[1]:box[3], box[0]:box[2])
    2. Letterboxing proportionnel (équivalent de LongestMaxSize + PadIfNeeded
       d'albumentations : scale = min(scale_x, scale_y), padding noir centré)
    3. Normalisation /255 en fp32

C'est délibérément DIFFÉRENT de py_inference_v2_same_result_C.py, qui lui
ajoute la marge de 10 % façon C++ (foms_positions.cpp). Si vos comparaisons
montrent qu'evaluation_inference_20_08 colle mieux au C++/GT, c'est ce
script-ci qu'il faut utiliser comme référence Python pour la vidéo.

Le mode "avec tatouage/keypoints" n'est PAS supporté ici intentionnellement :
l'objectif est la comparaison en mode sans keypoints (draw_keypoints=false
dans le config.yaml, cohérent avec les deux pipelines PostureDataset).

Un flag --use_margin est fourni pour basculer facilement vers la logique
avec marge de 10 % (façon C++ / py_inference_v2) si vous voulez comparer
les deux hypothèses sur les mêmes vidéos sans dupliquer le script.
"""

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


def letterbox_like_eval2008(roi_img, img_width, img_height):
    """Reproduit LongestMaxSize(max_size_hw=(img_height,img_width)) + PadIfNeeded
    tel qu'appliqué par albumentations dans DataAugPipeline, mais en pur cv2
    pour ne pas dépendre de la version d'albumentations installée.

    Retourne l'image letterboxée + les offsets (top, bottom, left, right, scale)
    utiles si jamais vous voulez projeter des keypoints plus tard.
    """
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

    # NOTE: BORDER_CONSTANT / valeur 0, comme foms_positions.cpp et
    # py_inference_v2. Si albumentations utilise BORDER_REFLECT_101 par
    # défaut dans votre version, changez border_mode/value ici pour tester.
    final_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    return final_img, (top, bottom, left, right, scale)


def run_video_inference(model_path, video_path, output_csv=None, config_path=None,
                         use_margin=False, margin_ratio=0.1):
    """
    Run inference on a video frame by frame, en répliquant le pré-traitement
    par défaut de PostureDataset / evaluation_inference_20_08.py.

    Args:
        model_path: chemin vers le modèle (.onnx, .pt, ou dossier DLR).
        video_path: chemin vers la vidéo (.mp4). Le CSV de tracking associé
            (même basename, .csv) doit se trouver au même endroit.
        output_csv: chemin de sortie personnalisé (sinon <video>_predictions.csv).
        config_path: chemin vers config.yaml (sinon déduit du dossier du modèle).
        use_margin: si True, ajoute une marge de 10% comme le C++/py_inference_v2
            au lieu du crop brut sans marge d'evaluation_inference_20_08.
        margin_ratio: ratio de la marge appliquée si use_margin=True (défaut 0.1).
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
        output_csv = f"{video_base_path}_predictions_eval2008.csv"

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- 1. CHARGEMENT DE LA CONFIGURATION ---
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
    print(f"Mode marge : {'10% (style C++)' if use_margin else 'AUCUNE (style eval_20_08 par défaut)'}")

    img_height = 256
    img_width = 192

    results = []

    # --- 4. BOUCLE D'INFÉRENCE FRAME PAR FRAME ---
    frame_idx = 0
    max_iterations = min(total_frames, len(df_input))
    pbar = tqdm(total=max_iterations, colour="blue", desc="Inférence Vidéo (eval2008-style)")

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
                for class_name in CPP_CLASS_ORDER:
                    frame_data[class_name] = -1.0
                results.append(frame_data)
                frame_idx += 1
                pbar.update(1)
                continue

            x = int(float(val_x))
            y = int(float(val_y))
            w = int(float(val_w))
            h = int(float(val_h))

            # --- CROP : réplique exactement PostureDataset.__getitem__ (défaut) ---
            # xmin/ymin/xmax/ymax clampés à l'image, comme
            #   xmin = max(0, int(box/left)) ; xmax = min(img.shape[1], int(box/right))
            xmin = max(0, x)
            ymin = max(0, y)
            xmax = min(image_gray.shape[1], x + w)
            ymax = min(image_gray.shape[0], y + h)

            if use_margin:
                delta_width = int(margin_ratio * float(xmax - xmin))
                xmin = max(0, xmin - delta_width)
                xmax = min(image_gray.shape[1], xmax + delta_width)

            if xmax > xmin and ymax > ymin:
                roi_img = image_gray[ymin:ymax, xmin:xmax]
            else:
                roi_img = image_gray

            final_img, _offsets = letterbox_like_eval2008(roi_img, img_width, img_height)

            # --- NORMALISATION (identique à PostureDataset : img_crop / 255.0) ---
            if is_dlr:
                img_tensor_np = final_img.astype(np.uint8)
            else:
                img_tensor_np = final_img.astype(np.float32) / 255.0

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

            frame_data = {"index": frame_idx}

            for class_name_from_model, prob in zip(class_names, full_probs):
                clean_name = class_name_from_model.lower().replace("_", "").replace(" ", "")
                final_name = "closetoairbag" if clean_name == "tooclosetoairbag" else clean_name
                frame_data[final_name] = float(prob)

            for cpp_col in CPP_CLASS_ORDER:
                if cpp_col not in frame_data:
                    frame_data[cpp_col] = -1.0

            results.append(frame_data)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    pbar.close()

    df_results = pd.DataFrame(results)
    final_columns = ["index"] + CPP_CLASS_ORDER
    df_results = df_results[final_columns]

    df_results.to_csv(output_csv, index=False)
    print(f"\n✅ Inférence terminée ! Les prédictions ont été sauvegardées dans : {output_csv}")


def main(model_path, video_path, output_csv=None, config_path=None,
         use_margin=False, margin_ratio=0.1):
    """
    Lance l'inférence d'un modèle de posture (PyTorch, ONNX, DLR) en répliquant
    le pipeline par défaut de PostureDataset (evaluation_inference_20_08.py,
    use_cpp_letterboxing=False) : crop sans marge + letterbox proportionnel.

    Args:
        model_path (str): Chemin vers le modèle (.onnx, .pt, ou dossier DLR).
        video_path (str): Chemin vers la vidéo d'entrée (.mp4).
        output_csv (str, optional): Chemin personnalisé pour le CSV de sortie.
        config_path (str, optional): Chemin vers le fichier config.yaml.
        use_margin (bool): Ajoute une marge de 10% (style C++) si True.
            Par défaut False, pour coller au comportement d'evaluation_inference_20_08.
        margin_ratio (float): Ratio de marge si use_margin=True (défaut 0.1).
    """
    if isinstance(use_margin, str):
        use_margin = use_margin.lower() in ['true', '1', 'yes']

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run_video_inference(model_path, video_path, output_csv, config_path, use_margin, margin_ratio)


if __name__ == "__main__":
    Fire(main)