"""
Automatisation de la comparaison entre un modèle ONNX (Baseline)
et les meilleurs checkpoints .pth ("best_" / "2nd_best_") de plusieurs
dossiers d'entraînement (same_config_trainings/10 à 18).
Génère un CSV complet des KPIs pour analyse sur Jupyter Notebook,
avec un run "average" résumant tous les checkpoints sélectionnés.
"""
import sys
import os
import re
import glob
import torch
import numpy as np
import pandas as pd
from fire import Fire
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader

# Pour tous mes codes, j'utilise des environnements Conda.
# Never remove my personal comments from the code.

# Ajout dynamique du chemin vers les modules d'entraînement et de posture
training_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))
posture_dir = os.path.join(training_root, 'src', 'posture')
if training_root not in sys.path:
    sys.path.append(training_root)
if posture_dir not in sys.path:
    sys.path.append(posture_dir)

from src.utils.config import get_config
from src.posture.dataset.posture_dataset import PostureDataset
from src.posture.train_posture import load_model

# Import des fonctions d'inférence
from posture_evaluation_new_metrics import infer
from posture_model_evaluation_onnx import infer_onnx


def compute_detailed_kpis(pred_dict, label_dict, class_names, thresh=0.5):
    """Calcule TP, FP, TN, FN, Recall, Precision et F1 pour toutes les séquences."""
    sequence_list = list(pred_dict.keys())

    # Remplir la classe 0 (correctly_seated)
    for seq in sequence_list:
        pos_preds = pred_dict[seq][:, 1:]
        valid_mask = ~np.isnan(pos_preds).any(axis=1)
        pred_dict[seq][valid_mask, 0] = 1.0 - np.clip(np.nanmax(pos_preds[valid_mask], axis=1), 0, 1)

    all_preds = np.concatenate([pred_dict[seq] for seq in sequence_list])
    all_labels = np.concatenate([label_dict[seq] for seq in sequence_list])

    metrics = {}
    for ii, cls in enumerate(class_names):
        y_true = all_labels[:, ii]
        y_pred = (all_preds[:, ii] >= thresh).astype(int)

        # Ignorer les frames invalides (NaN)
        valid_idx = ~np.isnan(all_preds[:, ii])
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]

        if len(y_true) == 0:
            continue

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        metrics[cls] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1_Score": f1_score(y_true, y_pred, zero_division=0),
            "Total_Frames": len(y_true)
        }
    return metrics


def find_best_checkpoints(ckpt_dir):
    """
    Cherche, dans un dossier ckpt donné, les checkpoints dont le nom commence
    par 'best' ou '2nd_best' (ex: best_epoch_48.pth, 2nd_best_epoch_98.pth),
    et extrait le numéro d'époque contenu dans le nom du fichier.
    Retourne une liste de tuples (epoch_num, checkpoint_type, path), triée
    par numéro d'époque.
    """
    files = []
    for ext in ("pth", "pt"):
        files += glob.glob(os.path.join(ckpt_dir, f"best*.{ext}"))
        files += glob.glob(os.path.join(ckpt_dir, f"2nd_best*.{ext}"))

    results = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        match = re.search(r"epoch_(\d+)", stem)
        if not match:
            print(f"⚠️  Numéro d'époque introuvable dans le nom, fichier ignoré : {f}")
            continue
        epoch_num = int(match.group(1))
        checkpoint_type = "2nd_best" if stem.startswith("2nd_best") else "best"
        results.append((epoch_num, checkpoint_type, f))

    results.sort(key=lambda x: x[0])
    return results


def main(
    onnx_path,
    config_path,
    root_dir="/mnt/data/OoP/data/source_code/training/saved_models/OoP/BSeed/same_config_trainings",
    n_start=10,
    n_end=18,
    output_csv="comparaison_epochs.csv",
    eval_side=None
):
    # Never remove my personal comments from the code.

    n_start = int(n_start)
    n_end = int(n_end)

    # Filtre de sécurité strict pour empêcher l'AssertionError du PostureDataset
    if str(eval_side).lower() not in ["driver", "passenger"]:
        eval_side = None

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = get_config()
    config.set_config(config_path)

    eval_dataset = PostureDataset(config, task="TEST", side=eval_side)

    # Personal comment: DataLoader universel avec batch_size=1 pour correspondre à l'ONNX et garantir la stabilité
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    sequence_df = eval_dataset.img_df.copy()
    sequence_df["seq_name"] = sequence_df["file"].apply(lambda x: os.path.basename(os.path.dirname(x)))

    num_positive_labels = len(config.input.positive_labels)
    class_names = ["correctly_seated"] + [eval_dataset.label_map[key] for key in range(num_positive_labels)]

    results_data = []

    # ==========================================
    # 1. ÉVALUATION DE LA BASELINE ONNX
    # ==========================================
    print(f"\n🚀 Évaluation du modèle de référence ONNX : {os.path.basename(onnx_path)}")
    pred_onnx, label_onnx = infer_onnx(eval_loader, sequence_df, onnx_path, num_positive_labels)
    onnx_metrics = compute_detailed_kpis(pred_onnx, label_onnx, class_names)

    for cls, kpis in onnx_metrics.items():
        row = {"Training_Run": "Baseline", "Model": "ONNX_Baseline", "Epoch": "Baseline",
               "Checkpoint_Type": "Baseline", "Class": cls}
        row.update(kpis)
        results_data.append(row)

    # ==========================================
    # 2. ÉVALUATION DES MEILLEURS CHECKPOINTS, POUR CHAQUE DOSSIER N
    # ==========================================
    model = load_model(config)
    model.to(device)

    for n in range(n_start, n_end + 1):
        ckpt_dir = os.path.join(root_dir, str(n), "ckpt")
        if not os.path.isdir(ckpt_dir):
            print(f"⚠️  Dossier introuvable, on saute : {ckpt_dir}")
            continue

        best_checkpoints = find_best_checkpoints(ckpt_dir)

        print(f"\n🔎 Training {n} : {len(best_checkpoints)} checkpoint(s) 'best'/'2nd_best' trouvé(s).")

        for epoch_num, checkpoint_type, pth_path in best_checkpoints:
            print(f"⚙️ Évaluation en cours : Training {n} / {checkpoint_type} (epoch {epoch_num})")

            checkpoint = torch.load(pth_path, map_location=device)
            trained_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(trained_state_dict, "module.")

            model.load_state_dict(trained_state_dict)
            model.eval()

            pred_pth, label_pth = infer(eval_loader, sequence_df, model, device, False, num_positive_labels)
            pth_metrics = compute_detailed_kpis(pred_pth, label_pth, class_names)

            for cls, kpis in pth_metrics.items():
                row = {"Training_Run": n, "Model": "PyTorch", "Epoch": epoch_num,
                       "Checkpoint_Type": checkpoint_type, "Class": cls}
                row.update(kpis)

                onnx_kpis = onnx_metrics[cls]
                row["Delta_F1_vs_ONNX"] = kpis["F1_Score"] - onnx_kpis["F1_Score"]
                row["Delta_FP_vs_ONNX"] = kpis["FP"] - onnx_kpis["FP"]
                row["Delta_TP_vs_ONNX"] = kpis["TP"] - onnx_kpis["TP"]
                row["Delta_TN_vs_ONNX"] = kpis["TN"] - onnx_kpis["TN"]
                row["Delta_FN_vs_ONNX"] = kpis["FN"] - onnx_kpis["FN"]
                row["Delta_Precision_vs_ONNX"] = kpis["Precision"] - onnx_kpis["Precision"]
                row["Delta_Recall_vs_ONNX"] = kpis["Recall"] - onnx_kpis["Recall"]

                results_data.append(row)

    # ==========================================
    # 3. RUN "AVERAGE" : MOYENNE DE TOUS LES CHECKPOINTS SÉLECTIONNÉS (HORS BASELINE)
    # ==========================================
    df_all = pd.DataFrame(results_data)
    non_baseline_mask = df_all["Training_Run"] != "Baseline"
    df_non_baseline = df_all[non_baseline_mask]

    numeric_cols = [
        "TP", "TN", "FP", "FN", "Precision", "Recall", "F1_Score", "Total_Frames",
        "Delta_F1_vs_ONNX", "Delta_FP_vs_ONNX", "Delta_TP_vs_ONNX",
        "Delta_TN_vs_ONNX", "Delta_FN_vs_ONNX", "Delta_Precision_vs_ONNX", "Delta_Recall_vs_ONNX"
    ]
    numeric_cols = [c for c in numeric_cols if c in df_non_baseline.columns]

    if not df_non_baseline.empty:
        avg_by_class = df_non_baseline.groupby("Class")[numeric_cols].mean()
        # Réordonne selon l'ordre de class_names (correctly_seated, puis classes positives),
        # au lieu de l'ordre alphabétique par défaut de groupby
        avg_by_class = avg_by_class.reindex(class_names)
        for cls in class_names:
            if cls not in avg_by_class.index or avg_by_class.loc[cls].isna().all():
                continue
            row = {"Training_Run": "average", "Model": "Average", "Epoch": "Average",
                   "Checkpoint_Type": "Average", "Class": cls}
            for col in numeric_cols:
                row[col] = avg_by_class.loc[cls, col]
            results_data.append(row)

    if not df_non_baseline.empty:
        stats = (
            df_non_baseline.groupby("Class")[["Precision", "Recall", "F1_Score"]]
            .agg(["min", "max", "std"])
            .reindex(class_names)
        )
        # Aplatit les colonnes multi-niveaux (ex: ("Precision","min") -> "Precision_Min")
        stats.columns = [f"{metric}_{stat.capitalize()}" for metric, stat in stats.columns]

        for cls in class_names:
            if cls not in stats.index or stats.loc[cls].isna().all():
                continue
            row = {"Training_Run": "variability", "Model": "Variability", "Epoch": "Variability",
                   "Checkpoint_Type": "Variability", "Class": cls}
            for col in stats.columns:
                row[col] = stats.loc[cls, col]
            results_data.append(row)

    # ==========================================
    # 4. EXPORTATION
    # ==========================================
    df = pd.DataFrame(results_data)
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Données exportées avec succès dans {output_csv}")
    print("➡️ Vous pouvez maintenant charger ce CSV dans votre Notebook !")


if __name__ == "__main__":
    Fire(main)