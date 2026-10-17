"""Compute posture metrics using PyTorch or ONNX Runtime"""
import os
import matplotlib
matplotlib.use('Agg') # Force le mode headless pour éviter le crash tkinter (workers/threads)
from tqdm import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fire import Fire
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from src.utils.config import get_config
from src.posture.train_posture import load_model
from src.posture.dataset.posture_dataset import PostureDataset
from src import PROJECT_DIR

# Import du module d'évaluation de séquence pour la génération des nouveaux plots
from src.posture import sequence_evaluation

import pdb


def run(model_path, config, eval_side, output_dir):
    """
    Run the evaluation of the posture model (supports .pth and .onnx).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        root_dir = os.path.dirname(model_path)
        output_dir = os.path.join(root_dir, "metrics")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    is_onnx = model_path.endswith(".onnx")

    # Load PyTorch Model if not ONNX
    model = None
    if not is_onnx:
        checkpoint = torch.load(model_path)
        trained_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(trained_state_dict, "module.")
        model = load_model(config)
        model.load_state_dict(trained_state_dict)
        model.to(device)
        model.eval()

    # Create dataloader
    eval_dataset = PostureDataset(config, task="TEST", side=eval_side)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    sequence_df = eval_dataset.img_df.copy()
    sequence_df["seq_name"] = sequence_df["file"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    num_positive_labels = len(config.input.positive_labels)

    # Inférence dynamique (ONNX ou PyTorch)
    if is_onnx:
        pred_dict, label_dict = infer_onnx(
            eval_loader, sequence_df, model_path, num_positive_labels
        )
    else:
        pred_dict, label_dict = infer_pytorch(
            eval_loader, sequence_df, model, device, cache=False, num_positive_labels=num_positive_labels
        )

    class_names = ["correctly_seated"] + [
        eval_dataset.label_map[key] for key in range(num_positive_labels)
    ]

    # --- CORRECTION : Remplir la colonne 0 (correctly_seated) ---
    for seq in pred_dict.keys():
        pos_preds = pred_dict[seq][:, 1:]
        valid_mask = ~np.isnan(pos_preds).any(axis=1)
        # La probabilité d'être correctement assis est le complémentaire de la classe positive max détectée
        pred_dict[seq][valid_mask, 0] = 1.0 - np.clip(np.nanmax(pos_preds[valid_mask], axis=1), 0, 1)

    # --- Adaptation des données pour les modules de séquence ---
    # INITIALISATION SÉCURISÉE DES DATAFRAMES
    all_pred_df = pd.DataFrame()
    all_gt_df = pd.DataFrame()
    labels_dict_adapter = {}
    
    for ii, pos_name in enumerate(class_names):
        seq_eval_name = pos_name.lower().replace("_", "").replace(" ", "").replace("too", "")
        
        all_pred_df[seq_eval_name] = np.concatenate([pred_dict[seq][:, ii] for seq in pred_dict.keys()])
        all_gt_df[seq_eval_name] = np.concatenate([label_dict[seq][:, ii] for seq in label_dict.keys()])
        labels_dict_adapter[ii] = seq_eval_name
    # -------------------------------------------------------------------------------------

    label_convert = None
    
    # --- PARTIE 1 : ANCIENNES MÉTRIQUES (Matrice de Confusion) ---
    pred_segment_dict, label_segment_dict = post_process_sequences(
        pred_dict,
        label_dict,
        class_names, # Utilise les beaux noms originaux pour la matrice
        thresh=0.5,
        label_convert=label_convert,
        out_dir=output_dir
    )

    # Compute metrics & standard plots (Histogramme de chevauchement)
    metrics = compute_metrics(pred_segment_dict, label_segment_dict, class_names)
    plot_metrics(metrics, out_dir=output_dir)

    # --- PARTIE 2 : NOUVEAUX PLOTS DE SÉQUENCES ---
    
    # 1. Génération des graphiques _start.png et _end.png
    sequence_evaluation.plot_start_end_annotated_segments(
        all_gt_df,
        all_pred_df,
        labels=labels_dict_adapter,
        sliding_window=10,
        threshold=0.5,
        min_pos_length=25,
        save_dir=output_dir
    )

    # 2. Évaluation avec fenêtre glissante (Seuil 0.5) et sauvegarde du fichier .txt
    results_dict = sequence_evaluation.evaluate_sliding_window(
        all_gt_df,
        all_pred_df,
        labels=labels_dict_adapter,
        sliding_window=10,
        threshold=0.5,
        detection_warmup=10,
        min_pos_length=25,
    )
    sequence_evaluation.print_table_results(
        results_dict,
        out_file=os.path.join(output_dir, "threshold_0.5_results.txt")
    )

    # 3. Évaluation sur l'ensemble des seuils et sauvegarde des courbes (_metrics_threshold.png)
    var_threshold_results_dict = sequence_evaluation.evaluate_varying_threshold(
        all_gt_df,
        all_pred_df,
        labels=labels_dict_adapter,
        step=0.05,
        sliding_window=10,
        detection_warmup=10,
        min_pos_length=25,
    )
    sequence_evaluation.plot_varying_thresholds(
        var_threshold_results_dict,
        save_dir=output_dir
    )


def infer_onnx(eval_loader, sequence_df, onnx_model_path, num_positive_labels):
    """Exécute l'inférence directement à partir d'un fichier ONNX."""
    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

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

    dataloader_pbar = tqdm(eval_loader, total=len(eval_loader), colour="green", leave=None)
    
    for image_tensors, posture_labels, posture_kpts, seq_name, frame_idx in dataloader_pbar:
        labels = np.squeeze(posture_labels.numpy())
        
        # Préparation des entrées ONNX
        if len(input_names) == 2:
            ort_inputs = {
                input_names[0]: image_tensors.cpu().numpy().astype(np.float32),
                input_names[1]: posture_kpts.cpu().numpy().astype(np.float32)
            }
        else:
            ort_inputs = {
                input_names[0]: image_tensors.cpu().numpy().astype(np.float32)
            }

        ort_outs = session.run([output_name], ort_inputs)
        pred_raw = np.squeeze(ort_outs[0])
        
        # Sigmoïde NumPy si le modèle retourne des logits
        pred = 1.0 / (1.0 + np.exp(-pred_raw))

        pred_dict[seq_name[0]][frame_idx.item(), 1:] = pred
        label_dict[seq_name[0]][frame_idx.item(), :] = labels

    return pred_dict, label_dict


def infer_pytorch(eval_loader, sequence_df, model, device, cache, num_positive_labels):
    """Exécute l'inférence PyTorch standard."""
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
    dataloader_pbar = tqdm(eval_loader, total=len(eval_loader), colour="green", leave=None)
    with torch.no_grad():
        for image_tensors, posture_labels, posture_kpts, seq_name, frame_idx in dataloader_pbar:
            labels = np.squeeze(posture_labels.numpy())
            if model.n_inputs == 2:
                pred = model(image_tensors.to(device), posture_kpts.to(device))
            else:
                pred = model(image_tensors.to(device))
            pred = np.squeeze(torch.sigmoid(pred).cpu().numpy())
            pred_dict[seq_name[0]][frame_idx.item(), 1:] = pred
            label_dict[seq_name[0]][frame_idx.item(), :] = labels

    return pred_dict, label_dict


def post_process_sequences(pred_dict, label_dict, class_names, thresh=0.5, label_convert=None, out_dir=None):
    pred_class = []
    label_class = []
    for seq in label_dict.keys():
        valid_rows = np.all(pred_dict[seq][:, 1:] >= 0, axis=1)
        pred_dict[seq] = (pred_dict[seq] > thresh).astype(np.float64)
        pred_dict[seq][np.sum(pred_dict[seq][:, 1:], axis=1) == 0, 0] = 1
        pred_dict[seq][~valid_rows, :] = 0
        pred_class_tmp = [(np.where(vec == 1)[0]).tolist() for vec in pred_dict[seq]]
        label_class_tmp = [(np.where(vec == 1)[0]).tolist() for vec in label_dict[seq]]

        if label_convert is not None:
            for ii in range(len(pred_class_tmp)):
                for src, tgt in zip(label_convert["src"], label_convert["tgt"]):
                    if sorted(pred_class_tmp[ii]) == sorted(src):
                        pred_class_tmp[ii] = tgt
                        pred_dict[seq][ii][:] = 0
                        for jj in tgt:
                            pred_dict[seq][ii][jj] = 1

        pred_class_tmp = [elt for elt, valid in zip(pred_class_tmp, valid_rows) if valid]
        label_class_tmp = [elt for elt, valid in zip(label_class_tmp, valid_rows) if valid]
        pred_class += pred_class_tmp
        label_class += label_class_tmp

    y_true_list = []
    y_pred_list = []
    for y_true, y_pred in zip(label_class, pred_class):
        y_true_list += [yt for yt in y_true for yp in y_pred]
        y_pred_list += [yp for yt in y_true for yp in y_pred]
    cm = confusion_matrix(
        y_true_list,
        y_pred_list,
        labels=np.array([ii for ii in range(len(class_names))]),
    )
    norm = cm.sum(axis=1)[:, np.newaxis]
    norm[norm == 0] = 1.0
    cm = cm.astype("float") / norm
    cm = np.nan_to_num(cm, copy=True)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    text_kw = {"fontsize": 8}
    cm_disp.plot(xticks_rotation="vertical", values_format=".2f", text_kw=text_kw)
    cm_disp.figure_.tight_layout()
    out_path = os.path.join(out_dir, "conf_matrix_all_classes.png") if out_dir else "conf_matrix_all_classes.png"
    cm_disp.figure_.savefig(out_path, dpi=300)
    plt.close(cm_disp.figure_)

    fill_mask = [1, 0, 1]
    discard_mask = [-1, 1, -1]
    pred_segment_dict = {key: np.zeros_like(val) for key, val in pred_dict.items()}
    label_segment_dict = {key: np.zeros_like(val) for key, val in label_dict.items()}
    for seq in label_dict.keys():
        for ii in range(label_dict[seq].shape[1]):
            conv_res = np.convolve(label_dict[seq][:, ii], fill_mask, mode="same")
            label_dict[seq][conv_res == 2, ii] = 1
            conv_res = np.convolve(label_dict[seq][:, ii], discard_mask, mode="same")
            label_dict[seq][conv_res == 1, ii] = 0

            conv_res = np.convolve(pred_dict[seq][:, ii], fill_mask, mode="same")
            pred_dict[seq][conv_res == 2, ii] = 1
            conv_res = np.convolve(pred_dict[seq][:, ii], discard_mask, mode="same")
            pred_dict[seq][conv_res == 1, ii] = 0

        for ii in range(label_dict[seq].shape[1]):
            conv_res = np.convolve(label_dict[seq][:, ii], [1, -1], mode="same")
            start_idx = np.where(conv_res == 1)[0]
            stop_idx = np.where(conv_res == -1)[0]
            for offset, idx in enumerate(start_idx):
                conv_res[idx] += offset
            for offset, idx in enumerate(stop_idx):
                conv_res[idx] -= offset
            label_segment_dict[seq][:, ii] = np.cumsum(conv_res)

            conv_res = np.convolve(pred_dict[seq][:, ii], [1, -1], mode="same")
            start_idx = np.where(conv_res == 1)[0]
            stop_idx = np.where(conv_res == -1)[0]
            for offset, idx in enumerate(start_idx):
                conv_res[idx] += offset
            for offset, idx in enumerate(stop_idx):
                conv_res[idx] -= offset
            pred_segment_dict[seq][:, ii] = np.cumsum(conv_res)

    return pred_segment_dict, label_segment_dict


def compute_metrics(pred_segment_dict, label_segment_dict, class_names):
    metrics = {key: {"gt_overlap": [], "pred_overlap": []} for key in label_segment_dict.keys()}
    for seq in label_segment_dict.keys():
        seq_data_gt = label_segment_dict[seq]
        seq_data_pred = pred_segment_dict[seq]
        metrics[seq]["gt_overlap"] = [[] for _ in range(seq_data_gt.shape[1])]
        metrics[seq]["pred_overlap"] = [[] for _ in range(seq_data_gt.shape[1])]
        out_dim = seq_data_gt.shape[1]
        for ii in range(out_dim):
            seg_instances_gt = [elt for elt in np.unique(seq_data_gt[:, ii]).tolist() if elt > 0]
            seg_instances_pred = [elt for elt in np.unique(seq_data_pred[:, ii]).tolist() if elt > 0]
            for seg_idx in seg_instances_gt:
                seg_gt = seq_data_gt[:, ii] == seg_idx
                seg_gt_len = np.sum(seg_gt)
                overlap = np.sum(seg_gt * (seq_data_pred[:, ii] > 0))
                metrics[seq]["gt_overlap"][ii].append([seg_gt_len, overlap])

            for seg_idx in seg_instances_pred:
                seg_pred = seq_data_pred[:, ii] == seg_idx
                seg_pred_len = np.sum(seg_pred)
                overlap = np.sum(seg_pred * (seq_data_gt[:, ii] > 0))
                gt_cover = 1 if any(gt > 0 and np.sum(seq_data_gt[seg_pred, ii] > 0) > 0 for gt in np.unique(seq_data_gt[seg_pred, ii])) else 0
                metrics[seq]["pred_overlap"][ii].append([seg_pred_len, overlap, gt_cover])

    min_seg_len = 10
    gt_overlap = [elt for seq in label_segment_dict.keys() for ii in range(seq_data_gt.shape[1]) for elt in metrics[seq]["gt_overlap"][ii] if elt[0] >= min_seg_len]
    pred_overlap = [elt for seq in label_segment_dict.keys() for ii in range(seq_data_gt.shape[1]) for elt in metrics[seq]["pred_overlap"][ii] if elt[0] >= min_seg_len]
    overlap_thresh = [0.2, 0.4, 0.6, 0.8, 1.0]
    TP, FN, FP0, FP1 = [], [], [], []
    TP_population_all, FP_population_all = len(gt_overlap), len(pred_overlap)
    for thresh in overlap_thresh:
        gt_ok = [elt for elt in gt_overlap if (elt[1] / elt[0]) >= thresh]
        fp0 = [elt for elt in pred_overlap if (1.0 - (elt[1] / elt[0])) >= thresh and elt[2] == 0]
        fp1 = [elt for elt in pred_overlap if (1.0 - (elt[1] / elt[0])) >= thresh and elt[2] == 1]
        TP.append((len(gt_ok) / max(1, len(gt_overlap))) * 100)
        FN.append(100.0 - TP[-1])
        FP0.append((len(fp0) / max(1, len(pred_overlap))) * 100)
        FP1.append((len(fp1) / max(1, len(pred_overlap))) * 100)

    TP_per_class, FN_per_class, FP0_per_class, FP1_per_class = [], [], [], []
    TP_population_per_class, FP_population_per_class = [], []
    for ii in range(out_dim):
        gt_o = [elt for seq in label_segment_dict.keys() for elt in metrics[seq]["gt_overlap"][ii] if elt[0] >= min_seg_len]
        pr_o = [elt for seq in label_segment_dict.keys() for elt in metrics[seq]["pred_overlap"][ii] if elt[0] >= min_seg_len]
        tp_t, fn_t, fp0_t, fp1_t = [], [], [], []
        for thresh in overlap_thresh:
            tp_t.append((len([e for e in gt_o if (e[1] / e[0]) >= thresh]) / max(1, len(gt_o))) * 100)
            fn_t.append(100.0 - tp_t[-1])
            fp0_t.append((len([e for e in pr_o if (1.0 - (e[1] / e[0])) >= thresh and e[2] == 0]) / max(1, len(pr_o))) * 100)
            fp1_t.append((len([e for e in pr_o if (1.0 - (e[1] / e[0])) >= thresh and e[2] == 1]) / max(1, len(pr_o))) * 100)
        TP_per_class.append(tp_t)
        FN_per_class.append(fn_t)
        FP0_per_class.append(fp0_t)
        FP1_per_class.append(fp1_t)
        TP_population_per_class.append(len(gt_o))
        FP_population_per_class.append(len(pr_o))

    return {
        "TP_all": TP, "FN_all": FN, "FP0_all": FP0, "FP1_all": FP1,
        "TP_per_class": TP_per_class, "FN_per_class": FN_per_class,
        "FP0_per_class": FP0_per_class, "FP1_per_class": FP1_per_class,
        "TP_population_all": TP_population_all, "FP_population_all": FP_population_all,
        "TP_population_per_class": TP_population_per_class, "FP_population_per_class": FP_population_per_class,
        "class_names": class_names, "overlap_thresh": overlap_thresh,
    }


def plot_metrics(metrics_dict, out_dir=None):
    plt.figure(figsize=(18, 12))
    n_classes = len(metrics_dict["TP_per_class"])
    width = 0.06
    x = np.array(metrics_dict["overlap_thresh"]) - 4 * width / 5
    fontsize = 10
    
    ax = plt.subplot(n_classes // 2 + 3, 3, 1)
    rects = ax.bar(x, metrics_dict["TP_all"], width, label="TP")
    ax.bar_label(rects, labels=["{}/{}".format(int(v * metrics_dict["TP_population_all"] / 100.0 + 0.5), metrics_dict["TP_population_all"]) for v in metrics_dict["TP_all"]], padding=len(x), fontsize=fontsize)
    rects = ax.bar(x + width, metrics_dict["FP1_all"], width, label="almost FP")
    ax.bar_label(rects, labels=["{}/{}".format(int(v * metrics_dict["FP_population_all"] / 100.0 + 0.5), metrics_dict["FP_population_all"]) for v in metrics_dict["FP1_all"]], padding=len(x), fontsize=fontsize)
    rects = ax.bar(x + 2 * width, metrics_dict["FP0_all"], width, label="FP")
    ax.bar_label(rects, labels=["{}/{}".format(int(v * metrics_dict["FP_population_all"] / 100.0 + 0.5), metrics_dict["FP_population_all"]) for v in metrics_dict["FP0_all"]], padding=len(x), fontsize=fontsize)
    ax.set_xlabel("min overlap, %", fontsize=fontsize)
    ax.set_ylabel("%", fontsize=fontsize)
    ax.set_title("All classes", fontsize=fontsize + 1)
    ax.legend(bbox_to_anchor=(-0.2, 1.3), loc="upper left", ncols=3, fontsize=fontsize)
    ax.set_ylim(0, 140)

    for ii in range(n_classes):
        ax = plt.subplot(n_classes // 3 + 2, 3, ii + 4)
        rects = ax.bar(x, metrics_dict["TP_per_class"][ii], width, label="TP")
        ax.bar_label(rects, labels=["{}/{}".format(int(v * metrics_dict["TP_population_per_class"][ii] / 100.0 + 0.5), metrics_dict["TP_population_per_class"][ii]) for v in metrics_dict["TP_per_class"][ii]], padding=len(x), fontsize=fontsize)
        rects = ax.bar(x + width, metrics_dict["FP1_per_class"][ii], width, label="almost FP")
        ax.bar_label(rects, labels=["{}/{}".format(int(v * metrics_dict["FP_population_per_class"][ii] / 100.0 + 0.5), metrics_dict["FP_population_per_class"][ii]) for v in metrics_dict["FP1_per_class"][ii]], padding=len(x), fontsize=fontsize)
        rects = ax.bar(x + 2 * width, metrics_dict["FP0_per_class"][ii], width, label="FP")
        ax.bar_label(rects, labels=["{}/{}".format(int(v * metrics_dict["FP_population_per_class"][ii] / 100.0 + 0.5), metrics_dict["FP_population_per_class"][ii]) for v in metrics_dict["FP0_per_class"][ii]], padding=len(x), fontsize=fontsize)
        ax.set_ylim(0, 140)
        ax.set_title(metrics_dict["class_names"][ii], fontsize=fontsize + 1)

    out_path = os.path.join(out_dir, "metrics_fig.png") if out_dir else "metrics_fig.png"
    plt.tight_layout(h_pad=1.2, w_pad=3.5)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main(model_path, output_dir=None, config_path=None, eval_side=None):
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
    run(model_path, config, eval_side, output_dir)


if __name__ == "__main__":
    Fire(main)