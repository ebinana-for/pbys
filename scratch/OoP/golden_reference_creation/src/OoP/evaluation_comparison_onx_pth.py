"""
Comparaison directe entre 3 modèles :
1. proBayes Baseline (ONNX original)
2. Irystec Epoch 40 (nouvel ONNX)
3. Irystec Reference (PyTorch Epoch 40)
"""
import sys
import os
import torch
import numpy as np
import pandas as pd
from fire import Fire
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader

# Pour tous mes codes, j'utilise des environnements Conda.
# Never remove my personal comments from the code.

training_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))
posture_dir = os.path.join(training_root, 'src', 'posture')
if training_root not in sys.path:
    sys.path.append(training_root)
if posture_dir not in sys.path:
    sys.path.append(posture_dir)

from src.utils.config import get_config
from src.posture.dataset.posture_dataset import PostureDataset
from src.posture.train_posture import load_model
from posture_evaluation_new_metrics import infer
from posture_model_evaluation_onnx import infer_onnx


def compute_detailed_kpis(pred_dict, label_dict, class_names, thresh=0.5):
    sequence_list = list(pred_dict.keys())
    
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
    baseline_onnx_path,
    epoch40_onnx_path, 
    pth_path, 
    config_path, 
    output_csv="comparaison_trois_modeles.csv", 
    eval_side=None
):
    # Never remove my personal comments from the code.
    
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
    # 1. ÉVALUATION DE LA BASELINE ONNX (proBayes)
    # ==========================================
    print(f"\n🚀 [1/3] Évaluation de la proBayes Baseline : {os.path.basename(baseline_onnx_path)}")
    pred_baseline, label_baseline = infer_onnx(eval_loader, sequence_df, baseline_onnx_path, num_positive_labels)
    baseline_metrics = compute_detailed_kpis(pred_baseline, label_baseline, class_names)
    
    for cls, kpis in baseline_metrics.items():
        row = {"Model": "proBayes_Baseline", "Epoch": "Baseline", "Class": cls}
        row.update(kpis)
        results_data.append(row)

    # ==========================================
    # 2. ÉVALUATION DE L'ONNX EPOCH 40
    # ==========================================
    print(f"\n🚀 [2/3] Évaluation de l'ONNX Epoch 40 : {os.path.basename(epoch40_onnx_path)}")
    pred_onnx40, label_onnx40 = infer_onnx(eval_loader, sequence_df, epoch40_onnx_path, num_positive_labels)
    onnx40_metrics = compute_detailed_kpis(pred_onnx40, label_onnx40, class_names)
    
    for cls, kpis in onnx40_metrics.items():
        row = {"Model": "ONNX_Irystec", "Epoch": "40_ONNX", "Class": cls}
        row.update(kpis)
        
        baseline_f1 = baseline_metrics[cls]["F1_Score"]
        row["Delta_F1_vs_Baseline"] = abs(baseline_f1 - kpis["F1_Score"])
        row["Delta_FP_vs_Baseline"] = kpis["FP"] - baseline_metrics[cls]["FP"]
        
        results_data.append(row)

    # ==========================================
    # 3. ÉVALUATION DE LA RÉFÉRENCE PYTORCH (Irystec - Epoch 40)
    # ==========================================
    print(f"\n⚙️ [3/3] Évaluation de la Irystec Reference (PyTorch) : {os.path.basename(pth_path)}")
    model = load_model(config)
    model.to(device)

    checkpoint = torch.load(pth_path, map_location=device)
    trained_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(trained_state_dict, "module.")
    
    model.load_state_dict(trained_state_dict)
    model.eval()

    pred_pth, label_pth = infer(eval_loader, sequence_df, model, device, False, num_positive_labels)
    pth_metrics = compute_detailed_kpis(pred_pth, label_pth, class_names)

    for cls, kpis in pth_metrics.items():
        row = {"Model": "PyTorch_Irystec", "Epoch": "40_PTH", "Class": cls}
        row.update(kpis)
        
        baseline_f1 = baseline_metrics[cls]["F1_Score"]
        row["Delta_F1_vs_Baseline"] = abs(baseline_f1 - kpis["F1_Score"])
        row["Delta_FP_vs_Baseline"] = kpis["FP"] - baseline_metrics[cls]["FP"]
        
        results_data.append(row)

    # ==========================================
    # 4. EXPORTATION
    # ==========================================
    df = pd.DataFrame(results_data)
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Données exportées avec succès dans {output_csv}")

if __name__ == "__main__":
    Fire(main)