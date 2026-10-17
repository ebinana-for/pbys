"""
Évaluation par lot automatisée pour dossiers d'entraînements.
Génère les KPIs individuels ultra-détaillés (TP, FP, TN, FN, Precision, Recall, F1, SW_Metrics),
la moyenne globale des matrices de confusion, et le tableau détaillé moyen.
"""
import os
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from fire import Fire

from posture_evaluation_new_metrics import infer, post_process_sequences
from src.posture import sequence_evaluation
from src.utils.config import get_config
from src.posture.train_posture import load_model
from src.posture.dataset.posture_dataset import PostureDataset

# Pour tous mes codes, j'utilise des environnements Conda.
# Never remove my personal comments from the code.
def find_best_models(root_dir):
    root_path = Path(root_dir)
    models_dict = {}
    
    for file_path in root_path.rglob("*.pth"):
        if "ckpt" in file_path.parts and file_path.name.startswith("best"):
            folder_name = file_path.parent.parent.name
            model_name = f"{folder_name}_{file_path.stem}"
            models_dict[model_name] = str(file_path)
            
    for file_path in root_path.rglob("*.pt"):
        if "ckpt" in file_path.parts and file_path.name.startswith("best"):
            folder_name = file_path.parent.parent.name
            model_name = f"{folder_name}_{file_path.stem}"
            models_dict[model_name] = str(file_path)

    return models_dict

def get_raw_confusion_matrix(pred_dict_orig, label_dict_orig, class_names, thresh):
    pred_dict = copy.deepcopy(pred_dict_orig)
    label_dict = copy.deepcopy(label_dict_orig)
    pred_class = []
    label_class = []
    
    for seq in label_dict.keys():
        valid_rows = np.all(pred_dict[seq][:, 1:] >= 0, axis=1)
        pred_dict[seq] = (pred_dict[seq] > thresh).astype(np.float64)
        pred_dict[seq][np.sum(pred_dict[seq][:, 1:], axis=1) == 0, 0] = 1
        pred_dict[seq][~valid_rows, :] = 0
        
        pred_class_tmp = [(np.where(vec == 1)[0]).tolist() for vec in pred_dict[seq]]
        label_class_tmp = [(np.where(vec == 1)[0]).tolist() for vec in label_dict[seq]]
        
        pred_class += [elt for elt, valid in zip(pred_class_tmp, valid_rows) if valid]
        label_class += [elt for elt, valid in zip(label_class_tmp, valid_rows) if valid]
        
    y_true_list = []
    y_pred_list = []
    for y_true, y_pred in zip(label_class, pred_class):
        y_true_list += [yt for yt in y_true for yp in y_pred]
        y_pred_list += [yp for yt in y_true for yp in y_pred]
        
    cm = confusion_matrix(y_true_list, y_pred_list, labels=np.arange(len(class_names)))
    return cm

def evaluate_single_model(model_name, model_path, device, output_root_dir, eval_side, seuil, global_config_path):
    print(f"\n⚙️ Évaluation en cours : {model_name}")
    
    if global_config_path is None:
        inference_folder = os.path.dirname(model_path)
        current_config_path = os.path.join(inference_folder, "../config.yaml")
    else:
        current_config_path = global_config_path
        
    config = get_config()
    config.set_config(current_config_path)
    
    model_output_dir = os.path.join(output_root_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    checkpoint = torch.load(model_path, map_location=device)
    trained_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(trained_state_dict, "module.")
    
    model = load_model(config)
    model.load_state_dict(trained_state_dict)
    model.to(device)
    model.eval()

    eval_dataset = PostureDataset(config, task="TEST", side=eval_side)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    num_positive_labels = len(config.input.positive_labels)
    class_names = ["correctly_seated"] + [eval_dataset.label_map[key] for key in range(num_positive_labels)]

    sequence_df = eval_dataset.img_df.copy()
    sequence_df["seq_name"] = sequence_df["file"].apply(lambda x: os.path.basename(os.path.dirname(x)))

    pred_dict, label_dict = infer(
        eval_loader, sequence_df, model, device, cache=False, num_positive_labels=num_positive_labels
    )

    for seq in pred_dict.keys():
        pos_preds = pred_dict[seq][:, 1:]
        valid_mask = ~np.isnan(pos_preds).any(axis=1)
        pred_dict[seq][valid_mask, 0] = 1.0 - np.clip(np.nanmax(pos_preds[valid_mask], axis=1), 0, 1)

    all_pred_df = pd.DataFrame()
    all_gt_df = pd.DataFrame()
    labels_dict_adapter = {}
    sequence_list = list(pred_dict.keys())
    
    for ii, pos_name in enumerate(class_names):
        seq_eval_name = pos_name.lower().replace("_", "").replace(" ", "").replace("too", "")
        all_pred_df[seq_eval_name] = np.concatenate([pred_dict[seq][:, ii] for seq in sequence_list])
        all_gt_df[seq_eval_name] = np.concatenate([label_dict[seq][:, ii] for seq in sequence_list])
        labels_dict_adapter[ii] = seq_eval_name

    valid_labels_adapter = {}
    for idx, (col_idx, seq_name) in enumerate(labels_dict_adapter.items()):
        if all_gt_df[seq_name].sum() > 0 or seq_name == "correctlyseated":
            valid_labels_adapter[col_idx] = seq_name
        else:
            print(f"   ⚠️ Classe ignorée (0 frame GT) : {seq_name}")

    results_dict = sequence_evaluation.evaluate_sliding_window(
        all_gt_df, all_pred_df, labels=valid_labels_adapter, sliding_window=10, 
        threshold=seuil, detection_warmup=10, min_pos_length=25,
    )
    
    sequence_evaluation.print_table_results(
        results_dict, out_file=os.path.join(model_output_dir, f"threshold_{seuil}_results.txt")
    )

    raw_cm = get_raw_confusion_matrix(pred_dict, label_dict, class_names, seuil)
    post_process_sequences(pred_dict.copy(), label_dict.copy(), class_names, thresh=seuil, out_dir=model_output_dir)

    all_preds = np.concatenate([pred_dict[seq] for seq in sequence_list])
    all_labels = np.concatenate([label_dict[seq] for seq in sequence_list])
    
    # === CONSTRUCTION DU DICTIONNAIRE DE MÉTRIQUES EXHAUSTIF (SAFE MODE) ===
    metrics = {}
    for ii, cls in enumerate(class_names):
        y_true = all_labels[:, ii]
        y_pred = (all_preds[:, ii] >= seuil).astype(int)
        
        # Extraction de TP, TN, FP, FN depuis confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Extraction des métriques d'évaluation de séquence (Sliding Window)
        seq_eval_name = cls.lower().replace("_", "").replace(" ", "").replace("too", "")
        sw_perf = results_dict.get(seq_eval_name, {})
        
        sw_fps = sw_perf.get("fp_segments", [])
        
        metrics[cls] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "SW_Mean_Recall": sw_perf.get("mean_recall", 0.0),
            "SW_FP_Segments": len(sw_fps),
            "SW_FP_Frames": sum(len(seg) for seg in sw_fps),
            "SW_Total_Frames": sw_perf.get("n_frames", 0),
            "SW_Total_Segments": sw_perf.get("n_segments", 0)
        }
        
    # Sauvegarde locale pour ce modèle
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.to_csv(os.path.join(model_output_dir, "detailed_metrics.csv"))
        
    return metrics, results_dict, raw_cm, class_names


def main(
    trainings_root_dir="/mnt/data/OoP/data/source_code/training/saved_models/OoP/BSeed/same_config_trainings",
    output_dir="/mnt/data/OoP/data/source_code/training/saved_evals/OoP/Bseed_eval",
    config_path=None, 
    eval_side=None,
    seuil=0.5
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["USE_CPP_LETTERBOXING"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models_to_evaluate = find_best_models(trainings_root_dir)
    
    if not models_to_evaluate:
        print(f"❌ Aucun fichier trouvé dans {trainings_root_dir}")
        return
        
    print(f"🔎 {len(models_to_evaluate)} modèles 'best' détectés. Démarrage de l'évaluation batch...")

    all_metrics = {}
    all_sliding_results = {}
    cms = []
    class_names_ref = None

    for model_name, model_path in sorted(models_to_evaluate.items()):
        metrics, res_dict, raw_cm, class_names = evaluate_single_model(
            model_name, model_path, device, output_dir, eval_side, seuil, config_path
        )
        all_metrics[model_name] = metrics
        all_sliding_results[model_name] = res_dict
        cms.append(raw_cm)
        if class_names_ref is None:
            class_names_ref = class_names

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. MATRICE DE CONFUSION MOYENNE
    # ==========================================
    if cms:
        total_cm = np.sum(cms, axis=0)
        norm = total_cm.sum(axis=1)[:, np.newaxis]
        norm[norm == 0] = 1.0
        cm_norm = total_cm.astype("float") / norm
        cm_norm = np.nan_to_num(cm_norm, copy=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names_ref)
        cm_disp.plot(xticks_rotation="vertical", values_format=".2f", text_kw={"fontsize": 14}, ax=ax)
        fig.tight_layout()
        avg_cm_path = os.path.join(output_dir, "average_conf_matrix.png")
        fig.savefig(avg_cm_path, dpi=300)
        plt.close(fig)
        print(f"\n🖼️ Matrice de confusion moyenne sauvegardée : {avg_cm_path}")

    # ==========================================
    # 2. CALCUL ET EXPORT DES MÉTRIQUES DÉTAILLÉES (MOYENNE GLOBALE)
    # ==========================================
    if all_metrics:
        avg_metrics = {}
        metric_keys = list(all_metrics[list(all_metrics.keys())[0]][class_names_ref[0]].keys())
        
        for cls in class_names_ref:
            avg_metrics[cls] = {}
            for key in metric_keys:
                avg_metrics[cls][key] = np.mean([all_metrics[m][cls][key] for m in all_metrics])
                
        df_avg_metrics = pd.DataFrame(avg_metrics).T
        avg_csv_path = os.path.join(output_dir, "average_detailed_metrics.csv")
        df_avg_metrics.to_csv(avg_csv_path)
        print(f"📊 Tableau complet des KPIs moyens (TP, FP, SW, etc.) sauvegardé : {avg_csv_path}")

    # ==========================================
    # 3. GÉNÉRATION DU TABLEAU DE SÉQUENCE MOYEN (SAFE MODE TXT)
    # ==========================================
    if all_sliding_results:
        avg_sliding = {}
        first_model = list(all_sliding_results.keys())[0]
        positions = list(all_sliding_results[first_model].keys())
        
        for pos in positions:
            avg_sliding[pos] = {}
            recalls, fp_segs, fp_frames, n_frames, n_segments = [], [], [], [], []
            
            for m in all_sliding_results:
                if pos in all_sliding_results[m]:
                    perf = all_sliding_results[m][pos]
                    
                    recalls.append(perf.get("mean_recall", 0.0))
                    n_frames.append(perf.get("n_frames", 0))
                    n_segments.append(perf.get("n_segments", 0))
                    
                    fps = perf.get("fp_segments", [])
                    fp_segs.append(len(fps))
                    fp_frames.append(sum(len(seg) for seg in fps))
            
            avg_sliding[pos]["mean_recall"] = np.mean(recalls) if recalls else 0.0
            avg_sliding[pos]["n_fp_segments"] = np.mean(fp_segs) if fp_segs else 0.0
            avg_sliding[pos]["n_fp_frames"] = np.mean(fp_frames) if fp_frames else 0.0
            avg_sliding[pos]["n_frames"] = np.mean(n_frames) if n_frames else 0.0
            avg_sliding[pos]["n_segments"] = np.mean(n_segments) if n_segments else 0.0
        
        avg_txt_path = os.path.join(output_dir, "average_sequence_results.txt")
        
        with open(avg_txt_path, "w") as f:
            f.write("+----------------------+-------------+-------------+-----------+--------------+----------------+\n")
            f.write("|       Position       | Mean recall | FP Segments | FP Frames | Total Frames | Total Segments |\n")
            f.write("+----------------------+-------------+-------------+-----------+--------------+----------------+\n")
            for pos, perf in avg_sliding.items():
                recall_str = f"{perf['mean_recall']:.2f}" if pos != "correctlyseated" else "---"
                fp_seg_str = f"{int(round(perf['n_fp_segments']))}" if pos != "correctlyseated" else "---"
                fp_frame_str = f"{int(round(perf['n_fp_frames']))}" if pos != "correctlyseated" else "---"
                tot_frames = int(round(perf["n_frames"]))
                tot_seg = int(round(perf["n_segments"]))
                
                f.write(f"| {pos:<20} | {recall_str:^11} | {fp_seg_str:^11} | {fp_frame_str:^9} | {tot_frames:^12} | {tot_seg:^14} |\n")
            f.write("+----------------------+-------------+-------------+-----------+--------------+----------------+\n")

    print("\n✅ Évaluation par lot terminée avec succès.")

if __name__ == "__main__":
    Fire(main)