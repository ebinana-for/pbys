"""
Évaluation automatisée pour dossiers d'entraînements.
Calcule UNIQUEMENT les KPIs frame-level (TP, FP, TN, FN, Precision, Recall, F1).
Suit exactement la méthodologie du script de référence (seuillage strict ">",
DataFrames construits avant seuillage, évaluation de toutes les classes).

Deux modes :
- mode="batch" (par défaut) : évalue tous les checkpoints "best" trouvés sous trainings_root_dir.
- mode="epoch" : évalue un seul checkpoint correspondant à un numéro d'epoch donné.

⚠️ Hypothèse : find_epoch_model() suppose que le nom du fichier checkpoint contient le numéro
d'epoch sous forme d'entier (ex: "epoch_12.pth"). Si ta convention diffère,
passe `checkpoint_pattern="ton_pattern_{epoch}.pth"`.
"""
import os
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from fire import Fire

from posture_evaluation_new_metrics import infer
from src.utils.config import get_config
from src.posture.train_posture import load_model
from src.posture.dataset.posture_dataset import PostureDataset

# Pour tous mes codes, j'utilise des environnements Conda.
# Never remove my personal comments from the code.


# ==========================================================================
# Recherche de checkpoints
# ==========================================================================
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


def find_epoch_model(root_dir, epoch, checkpoint_pattern=None):
    """Cherche, sous root_dir/**/ckpt/, le checkpoint correspondant à un epoch donné."""
    root_path = Path(root_dir)
    candidates = []

    for ext in ("*.pth", "*.pt"):
        for file_path in root_path.rglob(ext):
            if "ckpt" not in file_path.parts:
                continue
            if checkpoint_pattern is not None:
                if file_path.name == checkpoint_pattern.format(epoch=epoch):
                    candidates.append(file_path)
                continue
            # Heuristique par défaut : on cherche l'entier "epoch" dans le nom du fichier
            numbers = [int(n) for n in re.findall(r"\d+", file_path.stem)]
            if epoch in numbers:
                candidates.append(file_path)

    if not candidates:
        raise FileNotFoundError(
            f"Aucun checkpoint pour l'epoch {epoch} trouvé sous {root_dir}. "
            f"Vérifie la convention de nommage ou passe checkpoint_pattern=..."
        )
    if len(candidates) > 1:
        print(f"⚠️ Plusieurs checkpoints correspondent à l'epoch {epoch}, on prend le premier : {candidates}")

    return str(candidates[0])


# ==========================================================================
# Chargement modèle + dataset
# ==========================================================================
def _load_model_and_data(model_path, device, eval_side, global_config_path):
    if global_config_path is None:
        inference_folder = os.path.dirname(model_path)
        current_config_path = os.path.join(inference_folder, "../config.yaml")
    else:
        current_config_path = global_config_path

    config = get_config()
    config.set_config(current_config_path)

    checkpoint = torch.load(model_path, map_location=device)
    trained_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(trained_state_dict, "module.")

    model = load_model(config)
    model.load_state_dict(trained_state_dict)
    model.to(device)
    model.eval()

    # Le PostureDataset se charge de lire les images et d'appliquer les transformations 
    # (redimensionnement, normalisation, Tensors) définies dans ton config.yaml.
    eval_dataset = PostureDataset(config, task="TEST", side=eval_side)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    return model, config, eval_dataset, eval_loader


# ==========================================================================
# Calcul des KPIs (cœur du script)
# ==========================================================================
def compute_kpis(model, config, eval_dataset, eval_loader, device, seuil, model_output_dir):
    """
    Calcule les KPIs frame-level (TP, FP, TN, FN, Precision, Recall, F1)
    en reproduisant EXACTEMENT la méthodologie de l'ancienne matrice de confusion
    (post_process_sequences dans posture_evaluation_new_metrics.py) :
    - seuillage strict ">"
    - repli de la colonne 0 (correctly_seated) à 1 si aucune classe positive
      n'est prédite
    - construction de la matrice de confusion par produit cartésien (multi-label)
      des labels vrais actifs x labels prédits actifs, par frame
    TP/FP/FN/TN sont dérivés en entiers depuis cette matrice de confusion NON
    normalisée (contrairement à l'ancien script qui la normalisait par ligne
    pour l'affichage).
    """
    os.makedirs(model_output_dir, exist_ok=True)

    num_positive_labels = len(config.input.positive_labels)
    class_names = ["correctly_seated"] + [eval_dataset.label_map[key] for key in range(num_positive_labels)]

    sequence_df = eval_dataset.img_df.copy()
    sequence_df["seq_name"] = sequence_df["file"].apply(lambda x: os.path.basename(os.path.dirname(x)))

    pred_dict, label_dict = infer(
        eval_loader, sequence_df, model, device, cache=False, num_positive_labels=num_positive_labels
    )

    # --- Remplir la colonne 0 (correctly_seated) ---
    for seq in pred_dict.keys():
        pos_preds = pred_dict[seq][:, 1:]
        valid_mask = ~np.isnan(pos_preds).any(axis=1)
        pred_dict[seq][valid_mask, 0] = 1.0 - np.clip(np.nanmax(pos_preds[valid_mask], axis=1), 0, 1)

    n_classes = len(class_names)
    pred_class_all = []
    label_class_all = []

    for seq in label_dict.keys():
        preds = pred_dict[seq].copy()
        labels = label_dict[seq]

        # Lignes valides : comme dans l'ancien script (>=0 sur les colonnes positives,
        # ce qui exclut les NaN car sigmoid ne produit jamais de négatif)
        valid_rows = np.all(preds[:, 1:] >= 0, axis=1)

        # Seuillage strict ">" sur toutes les colonnes, y compris correctly_seated
        preds = (preds > seuil).astype(np.float64)

        # Règle de repli : si aucune classe positive n'est prédite -> correctly_seated = 1
        preds[np.sum(preds[:, 1:], axis=1) == 0, 0] = 1

        # Lignes invalides -> tout à 0
        preds[~valid_rows, :] = 0

        pred_class_tmp = [(np.where(vec == 1)[0]).tolist() for vec in preds]
        label_class_tmp = [(np.where(vec == 1)[0]).tolist() for vec in labels]

        # On ne garde que les lignes valides (comme l'ancien script)
        pred_class_tmp = [elt for elt, v in zip(pred_class_tmp, valid_rows) if v]
        label_class_tmp = [elt for elt, v in zip(label_class_tmp, valid_rows) if v]

        pred_class_all += pred_class_tmp
        label_class_all += label_class_tmp

    # Produit cartésien vrai x prédit par frame, comme dans l'ancien script
    y_true_list = []
    y_pred_list = []
    for y_true, y_pred in zip(label_class_all, pred_class_all):
        y_true_list += [yt for yt in y_true for yp in y_pred]
        y_pred_list += [yp for yt in y_true for yp in y_pred]

    cm_counts = confusion_matrix(y_true_list, y_pred_list, labels=list(range(n_classes)))
    total = cm_counts.sum()

    metrics = {}
    for ii, cls in enumerate(class_names):
        tp = int(cm_counts[ii, ii])
        fn = int(cm_counts[ii, :].sum() - tp)
        fp = int(cm_counts[:, ii].sum() - tp)
        tn = int(total - tp - fn - fp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics[cls] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        }

    df_metrics = pd.DataFrame(metrics).T
    df_metrics.to_csv(os.path.join(model_output_dir, "detailed_metrics.csv"))

    return metrics, class_names
def evaluate_single_model(
    model_name, model_path, device, output_root_dir, eval_side, seuil, global_config_path
):
    print(f"\n⚙️ Évaluation en cours : {model_name}")
    model, config, eval_dataset, eval_loader = _load_model_and_data(model_path, device, eval_side, global_config_path)
    model_output_dir = os.path.join(output_root_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    metrics, class_names = compute_kpis(
        model, config, eval_dataset, eval_loader, device, seuil, model_output_dir
    )
    return metrics, class_names


# ==========================================================================
# Main
# ==========================================================================
def main(
    trainings_root_dir="/mnt/data/OoP/data/source_code/training/saved_models/OoP/BSeed/same_config_trainings",
    output_dir="/mnt/data/OoP/data/source_code/training/saved_evals/OoP/Bseed_eval",
    config_path=None,
    eval_side=None,
    seuil=0.5,
    mode="batch",               # "batch" ou "epoch"
    epoch=None,                 # requis si mode="epoch"
    checkpoint_pattern=None     # ex: "epoch_{epoch}.pth" si ta convention diffère de l'heuristique par défaut
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["USE_CPP_LETTERBOXING"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- mode == "epoch" : un seul checkpoint ----------------
    if mode == "epoch":
        if epoch is None:
            raise ValueError("mode='epoch' nécessite de préciser epoch=<numéro>.")

        model_path = find_epoch_model(trainings_root_dir, epoch, checkpoint_pattern)
        model_name = f"epoch_{epoch}"
        print(f"🔎 Checkpoint sélectionné pour l'epoch {epoch} : {model_path}")

        evaluate_single_model(
            model_name, model_path, device, output_dir, eval_side, seuil, config_path
        )
        print(f"\n✅ KPIs sauvegardés dans {os.path.join(output_dir, model_name, 'detailed_metrics.csv')}")
        return

    # ---------------- mode == "batch" : tous les checkpoints "best" ----------------
    models_to_evaluate = find_best_models(trainings_root_dir)
    if not models_to_evaluate:
        print(f"❌ Aucun fichier trouvé dans {trainings_root_dir}")
        return

    print(f"🔎 {len(models_to_evaluate)} modèles 'best' détectés. Démarrage de l'évaluation batch...")

    all_metrics = {}
    class_names_ref = None

    for model_name, model_path in sorted(models_to_evaluate.items()):
        metrics, class_names = evaluate_single_model(
            model_name, model_path, device, output_dir, eval_side, seuil, config_path
        )
        all_metrics[model_name] = metrics
        if class_names_ref is None:
            class_names_ref = class_names

    # ---- Moyenne globale des KPIs (CSV uniquement, aucune figure) ----
    if all_metrics:
        avg_metrics = {}
        metric_keys = list(all_metrics[list(all_metrics.keys())[0]][class_names_ref[0]].keys())
        for cls in class_names_ref:
            avg_metrics[cls] = {
                key: np.mean([all_metrics[m][cls][key] for m in all_metrics])
                for key in metric_keys
            }
        df_avg_metrics = pd.DataFrame(avg_metrics).T
        avg_csv_path = os.path.join(output_dir, "average_detailed_metrics.csv")
        df_avg_metrics.to_csv(avg_csv_path)
        print(f"📊 KPIs moyens sauvegardés : {avg_csv_path}")

    print("\n✅ Évaluation par lot terminée avec succès.")


if __name__ == "__main__":
    Fire(main)
