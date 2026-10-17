"""
Automatisation de la comparaison entre un modèle ONNX (Baseline) 
et des checkpoints .pth (avec un pas de 5 époques).
Génère un CSV complet des KPIs pour analyse sur Jupyter Notebook.
"""
import sys
import os
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
# Ajoute la racine 'training' pour trouver 'src.utils' et 'src.posture'
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


def main(
    onnx_path, 
    pth_dir, 
    config_path, 
    output_csv="comparaison_epochs.csv", 
    eval_side=None, 
    step=5,
    start_epoch=20
):
    # Never remove my personal comments from the code.
    
    # Conversion explicite en entiers pour corriger le TypeError de Fire
    step = int(step)
    start_epoch = int(start_epoch)
    
    # Filtre de sécurité strict pour empêcher l'AssertionError du PostureDataset
    if str(eval_side).lower() not in ["driver", "passenger"]:
        eval_side = None

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # ... le reste du code reste exactement identique
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
        row = {"Model": "ONNX_Baseline", "Epoch": "Baseline", "Class": cls}
        row.update(kpis)
        results_data.append(row)

    # ==========================================
    # 2. SÉLECTION ET ÉVALUATION DES ÉPOQUES (Pas de 5, >= 20)
    # ==========================================
    pth_files = glob.glob(os.path.join(pth_dir, "epoch_*.pth"))
    pth_files = sorted(pth_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    # Filtre par pas (step) et par époque minimum (start_epoch)
    selected_pths = [
        f for f in pth_files 
        if int(os.path.basename(f).split('_')[1].split('.')[0]) % step == 0 
        and int(os.path.basename(f).split('_')[1].split('.')[0]) >= start_epoch
    ]
    
    print(f"\n🔎 {len(selected_pths)} époques sélectionnées (Saut de {step} en {step}, à partir de l'époque {start_epoch}).")

    model = load_model(config)
    model.to(device)

    for pth_path in selected_pths:
        epoch_num = int(os.path.basename(pth_path).split('_')[1].split('.')[0])
        print(f"⚙️ Évaluation en cours : Epoch {epoch_num}")
        
        checkpoint = torch.load(pth_path, map_location=device)
        trained_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(trained_state_dict, "module.")
        
        model.load_state_dict(trained_state_dict)
        model.eval()

        pred_pth, label_pth = infer(eval_loader, sequence_df, model, device, False, num_positive_labels)
        pth_metrics = compute_detailed_kpis(pred_pth, label_pth, class_names)

        for cls, kpis in pth_metrics.items():
            row = {"Model": "PyTorch", "Epoch": epoch_num, "Class": cls}
            row.update(kpis)
            
            onnx_f1 = onnx_metrics[cls]["F1_Score"]
            row["Delta_F1_vs_ONNX"] = abs(onnx_f1 - kpis["F1_Score"])
            row["Delta_FP_vs_ONNX"] = kpis["FP"] - onnx_metrics[cls]["FP"]
            
            results_data.append(row)

    # ==========================================
    # 3. EXPORTATION
    # ==========================================
    df = pd.DataFrame(results_data)
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Données exportées avec succès dans {output_csv}")
    print("➡️ Vous pouvez maintenant charger ce CSV dans votre Notebook !")

if __name__ == "__main__":
    Fire(main)