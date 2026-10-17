"""
Générateur de CSV d'inférence via l'évaluateur PyTorch - 20/08/2026
Utilise PostureDataset pour évaluer le dataset et exporte les résultats 
au format strict C++ avec les vérités terrains (GT) associées.

Modifié pour calculer les KPI par vidéo (séquence) ET de manière globale.
Exporte 1 CSV par vidéo pour les prédictions, et 1 CSV récapitulatif pour les KPIs.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
from fire import Fire
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from utils.config import get_config
from utils.posture_dataset import PostureDataset
from src import PROJECT_DIR


def load_model(config):
    """
    Load the posture classification model based on the configuration.
    Args:
        config (Config): Configuration object containing model parameters.
    Returns:
        model (torch.nn.Module): The loaded posture classification model.
    """
    head_param = []
    for key, val in config.model.heads_params.items():
        head = val.copy()
        head["name"] = key
        head_param.append(head)

    if config.model.name == "mobilenet_v2":
        assert (
            len(head_param) == 1
        ), "You defined {} heads, but {} cannot handle more than 1.".format(
            len(head_param), config.model.name
        )
        model = MobileNetV2Posture(
            head_param,
            in_chans=config.model.input_channels,
            width_mult=config.model.width_multiplier,
        )
    else:
        logger.error(f"Unknown model {config.model.name}")
        raise NotImplementedError(f"Model {config.model.name} is not supported.")   

    

    if "checkpoint" in config.model and config.model.checkpoint is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading models weight from the checkpoint : {config.model.checkpoint}")
        
        checkpoint_data = torch.load(config.model.checkpoint, map_location=device)
        
        # Gestion si c'est un dico complet ou juste les poids
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        else:
            state_dict = checkpoint_data

        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)

    return model


def compute_detailed_kpis(pred_dict, label_dict, class_names, thresh=0.5):
    """
    Calcule TP, FP, TN, FN, Precision, Recall et F1 pour CHAQUE séquence individuellement,
    puis ajoute une métrique GLOBALE regroupant toutes les séquences[cite: 7].
    """
    results = []

    # 1. Calculs par vidéo (séquence individuelle)
    for seq in pred_dict.keys():
        seq_preds = pred_dict[seq]
        seq_labels = label_dict[seq]

        for ii, cls in enumerate(class_names):
            y_true = seq_labels[:, ii]
            y_pred = (seq_preds[:, ii] >= thresh).astype(int)

            # Ignorer les frames invalides (NaN) pour cette classe précisément[cite: 7]
            valid_idx = ~np.isnan(seq_preds[:, ii])
            y_true = y_true[valid_idx]
            y_pred = y_pred[valid_idx]

            if len(y_true) == 0:
                continue

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            results.append({
                "Sequence": seq,
                "Class": cls,
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "Total_Frames": len(y_true),
            })

    # 2. Calculs GLOBAUX (toutes séquences confondues)[cite: 7]
    sequence_list = list(pred_dict.keys())
    all_preds = np.concatenate([pred_dict[seq] for seq in sequence_list])
    all_labels = np.concatenate([label_dict[seq] for seq in sequence_list])

    for ii, cls in enumerate(class_names):
        y_true = all_labels[:, ii]
        y_pred = (all_preds[:, ii] >= thresh).astype(int)

        valid_idx = ~np.isnan(all_preds[:, ii])
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]

        if len(y_true) == 0:
            continue

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        results.append({
            "Sequence": "GLOBAL",
            "Class": cls,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "Total_Frames": len(y_true),
        })

    return pd.DataFrame(results)


def run(model_path, config, eval_side, output_dir, use_cpp_letterboxing, kpi_threshold=0.5, kpi_csv_path=None):
    """
    Run the evaluation of the posture model and export results to CSV with Ground Truth[cite: 7].
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    os.environ["USE_CPP_LETTERBOXING"] = "1" if use_cpp_letterboxing else "0"
    mode_str = "RATIO C++ (Letterboxing)" if use_cpp_letterboxing else "DISTORSION (Étirement par défaut)"
    print(f"📐 Mode Redimensionnement (Dataset) : {mode_str}")
    if use_cpp_letterboxing:
        print("⚠️  ATTENTION : ce mode diverge de posture_evaluation_new_metrics.py et "
              "comparison_avg_10.py. Les KPI produits ici ne seront pas directement comparables[cite: 7].")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        root_dir = os.path.dirname(model_path)
        output_dir = os.path.join(root_dir, "csv_exports_with_gt")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    is_onnx = model_path.endswith(".onnx")
    
    # --- CHARGEMENT DU MODÈLE (.ONNX OU .PT) ---
    if is_onnx:
        import onnxruntime as ort
        print(f"⚙️ Chargement du modèle ONNX : {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = ort.InferenceSession(model_path, providers=providers)
    else:
        print(f"⚙️ Chargement du modèle PyTorch : {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            trained_state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            trained_state_dict = checkpoint["state_dict"]
        else:
            trained_state_dict = checkpoint
        
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            trained_state_dict, "module."
        )
        model = load_model(config)
        model.load_state_dict(trained_state_dict)
        model.to(device)
        model.eval()

    eval_dataset = PostureDataset(config, task="TEST", side=eval_side)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    sequence_df = eval_dataset.img_df.copy()
    sequence_df["seq_name"] = sequence_df["file"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    num_positive_labels = len(config.input.positive_labels)
    
    print("🚀 Démarrage de l'inférence sur le dataset...")
    pred_dict, label_dict = infer(
        eval_loader,
        sequence_df,
        model,
        device,
        cache=False,
        num_positive_labels=num_positive_labels,
        is_onnx=is_onnx
    )

    class_names = ["correctly_seated"] + [
        eval_dataset.label_map[key] for key in range(num_positive_labels)
    ]
    
    # --- Remplir la colonne 0 (correctly_seated) ---
    for seq in pred_dict.keys():
        pos_preds = pred_dict[seq][:, 1:]
        valid_mask = ~np.isnan(pos_preds).any(axis=1)
        pred_dict[seq][valid_mask, 0] = 1.0 - np.clip(np.nanmax(pos_preds[valid_mask], axis=1), 0, 1)

    # =====================================================================
    # CALCUL DES KPI (Par Vidéo + GLOBAL) + EXPORT CSV
    # =====================================================================
    print(f"\n📊 Calcul des KPI (seuil={kpi_threshold})...")
    df_kpi = compute_detailed_kpis(pred_dict, label_dict, class_names, thresh=kpi_threshold)

    # Réordonne selon l'ordre canonique des classes[cite: 7]
    df_kpi["Class"] = pd.Categorical(df_kpi["Class"], categories=class_names, ordered=True)
    
    # Séparation pour forcer les lignes "GLOBAL" à s'afficher à la toute fin du fichier
    is_global = df_kpi['Sequence'] == 'GLOBAL'
    df_kpi_seq = df_kpi[~is_global].sort_values(['Sequence', 'Class'])
    df_kpi_global = df_kpi[is_global].sort_values(['Class'])
    df_kpi = pd.concat([df_kpi_seq, df_kpi_global]).reset_index(drop=True)

    if kpi_csv_path is None:
        kpi_csv_path = os.path.join(output_dir, "KPI_summary_videos.csv")
    elif kpi_csv_path.endswith(".xlsx"):
        kpi_csv_path = kpi_csv_path.replace(".xlsx", ".csv")

    df_kpi.to_csv(kpi_csv_path, index=False)
    print(f"📊 KPI exportés (CSV) : {kpi_csv_path}")

    # =====================================================================
    # EXPORT DES PRÉDICTIONS + GT EN CSV (Format strict C++) — INCHANGÉ
    # =====================================================================
    cpp_class_order = [
        "closetoairbag", "legsondashboard", "torsodissimetry",           
        "reachingback", "lateralbodycontact", "oneshoulderincontact",
        "pelvisrotation", "pelvisdrift"
    ]
    
    print(f"\n💾 Exportation des {len(pred_dict.keys())} séquences au format CSV (Pred + GT)...")
    
    for seq in pred_dict.keys():
        seq_preds = pred_dict[seq]
        seq_labels = label_dict[seq]
        df_export = pd.DataFrame()
        
        df_export["index"] = np.arange(seq_preds.shape[0])
        
        for ii, class_name_from_model in enumerate(class_names):
            clean_name = class_name_from_model.lower().replace("_", "").replace(" ", "")
            final_name = "closetoairbag" if clean_name == "tooclosetoairbag" else clean_name
                
            df_export[final_name] = seq_preds[:, ii]
            df_export[f"{final_name}_gt"] = seq_labels[:, ii]

        for cpp_col in cpp_class_order:
            if cpp_col not in df_export.columns:
                df_export[cpp_col] = -1.0
                df_export[f"{cpp_col}_gt"] = -1
                
        final_columns = ["index"]
        for cpp_col in cpp_class_order:
            final_columns.append(cpp_col)
            final_columns.append(f"{cpp_col}_gt")
            
        df_export = df_export[final_columns]
        
        csv_out_path = os.path.join(output_dir, f"{seq}_eval_with_gt.csv")
        df_export.to_csv(csv_out_path, index=False)
        print(f"  -> Sauvegardé : {csv_out_path}")
        
    print("\n✅ Inférence et export terminés avec succès !")


def infer(eval_loader, sequence_df, model, device, cache, num_positive_labels, is_onnx):
    """
    Run inference on the evaluation dataset and compute predictions and labels[cite: 7].
    """
    compute = True
    if cache:
        cache_dir = PROJECT_DIR / "cache"
        if not os.path.isdir(str(cache_dir)):
            os.makedirs(str(cache_dir))
        cache_file = cache_dir / "posture_eval_tmp.pkl"
        if os.path.isfile(str(cache_file)):
            with open(str(cache_file), "rb") as handle:
                data = pickle.load(handle)
                pred_dict, label_dict = data
                compute = False

    if compute:
        sequence_list = np.unique(sequence_df["seq_name"].values).tolist()
        sequence_len = [
            sequence_df[sequence_df["seq_name"] == name]["frame_index"].max() + 1
            for name in sequence_list
        ]
        out_dim = num_positive_labels + 1
        pred_dict = {
            key: np.empty((nrows, out_dim), dtype=np.float64) * np.nan
            for key, nrows in zip(sequence_list, sequence_len)
        }
        label_dict = {
            key: np.zeros((nrows, out_dim), dtype=np.int64)
            for key, nrows in zip(sequence_list, sequence_len)
        }
        dataloader_pbar = tqdm(
            eval_loader, total=len(eval_loader), colour="green", leave=None
        )
        with torch.no_grad():
            cnt = 0
            for (
                image_tensors,
                posture_labels,
                posture_kpts,
                seq_name,
                frame_idx,
            ) in dataloader_pbar:
                assert seq_name[0] in pred_dict, "Sequence {} not is result_dict".format(seq_name[0])
                labels = np.squeeze(posture_labels.numpy())
                
                # --- INFÉRENCE HYBRIDE (ONNX ou PyTorch) ---
                if is_onnx:
                    input_names = [inp.name for inp in model.get_inputs()]
                    ort_inputs = {input_names[0]: image_tensors.cpu().numpy().astype(np.float32)}
                    
                    if len(input_names) > 1:
                        ort_inputs[input_names[1]] = posture_kpts.cpu().numpy().astype(np.float32)
                        
                    ort_outs = model.run(None, ort_inputs)
                    pred_raw = np.squeeze(ort_outs[0])
                    pos_probs = 1.0 / (1.0 + np.exp(-pred_raw))
                    
                    if pos_probs.ndim == 0:
                        pos_probs = np.array([pos_probs])
                    pred_dict[seq_name[0]][frame_idx.item(), 1:] = pos_probs
                else:
                    if model.n_inputs == 2:
                        pred = model(image_tensors.to(device), posture_kpts.to(device))
                    else:
                        pred = model(image_tensors.to(device))
                    
                    pos_probs = np.squeeze(torch.sigmoid(pred).cpu().numpy())
                    if pos_probs.ndim == 0:
                        pos_probs = np.array([pos_probs])
                    pred_dict[seq_name[0]][frame_idx.item(), 1:] = pos_probs
                
                label_dict[seq_name[0]][frame_idx.item(), :] = labels
                cnt += 1

        if cache:
            data = [pred_dict, label_dict]
            with open(str(cache_file), "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pred_dict, label_dict


def main(model_path, output_dir=None, config_path=None, eval_side=None, use_cpp_letterboxing=False,
         kpi_threshold=0.5, kpi_csv_path=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if config_path is None:
        inference_folder = os.path.dirname(model_path)
        config_path = os.path.join(inference_folder, "../config.yaml")
    assert os.path.isfile(config_path), f"Config file not found: {config_path}"

    config = get_config()
    config.set_config(config_path)

    possible_eval_sides = ["driver", "passenger", None]
    if eval_side not in possible_eval_sides:
        raise ValueError(
            f"incorrect eval_side: {eval_side}. Must be one of: {possible_eval_sides}"
        )

    run(model_path, config, eval_side, output_dir, use_cpp_letterboxing,
        kpi_threshold=kpi_threshold, kpi_csv_path=kpi_csv_path)


if __name__ == "__main__":
    Fire(main)