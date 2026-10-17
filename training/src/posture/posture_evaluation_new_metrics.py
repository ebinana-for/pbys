"""Compute posture metrics using the pytorch version of the classifier"""
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
    Run the evaluation of the posture model.
    Args:
        model_path (str): Path to the trained posture model.
        config (Config): Configuration object containing model and evaluation parameters.
        eval_side (str): Side of the vehicle to evaluate ('driver', 'passenger', or None).
        output_dir (str): Directory to save the evaluation metrics and plots.
    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If eval_side is not one of the expected values.
    """
    # Check if model path exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # GPU settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        root_dir = os.path.dirname(model_path)
        output_dir = os.path.join(root_dir, "metrics")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load model
    checkpoint = torch.load(model_path)

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

    # Create dataloader
    eval_dataset = PostureDataset(config, task="TEST", side=eval_side)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Result
    sequence_df = eval_dataset.img_df.copy()
    sequence_df["seq_name"] = sequence_df["file"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    num_positive_labels = len(config.input.positive_labels)
    pred_dict, label_dict = infer(
        eval_loader,
        sequence_df,
        model,
        device,
        cache=False,
        num_positive_labels=num_positive_labels
    )

    # Compute posture video segmentation masks
    class_names = ["correctly_seated"] + [
        eval_dataset.label_map[key] for key in range(num_positive_labels)
    ]
    
    # --- Remplir la colonne 0 (correctly_seated) ---
    for seq in pred_dict.keys():
        pos_preds = pred_dict[seq][:, 1:]
        valid_mask = ~np.isnan(pos_preds).any(axis=1)
        # La probabilité d'être correctement assis est le complémentaire de la classe positive max détectée
        pred_dict[seq][valid_mask, 0] = 1.0 - np.clip(np.nanmax(pos_preds[valid_mask], axis=1), 0, 1)

    # --- Adaptation des données pour les Safety Metrics AVANT le post-processing ---
    # On convertit les dictionnaires de séquences en DataFrames continus (format attendu par le notebook)
    all_pred_df = pd.DataFrame()
    all_gt_df = pd.DataFrame()
    labels_dict_adapter = {}
    
    for ii, pos_name in enumerate(class_names):
        seq_eval_name = pos_name.lower().replace("_", "").replace(" ", "").replace("too", "")
        
        all_pred_df[seq_eval_name] = np.concatenate([pred_dict[seq][:, ii] for seq in pred_dict.keys()])
        all_gt_df[seq_eval_name] = np.concatenate([label_dict[seq][:, ii] for seq in label_dict.keys()])
        labels_dict_adapter[ii] = seq_eval_name
    # -------------------------------------------------------------------------------------

    # label_convert = {
    #     "src": [
    #         [
    #             class_names.index(elt)
    #             for elt in ["Pelvis rotation", "One_shoulder_in_contact"]
    #         ]
    #     ],
    #     "tgt": [[class_names.index(elt) for elt in ["Lateral_body_contact"]]],
    # }
    label_convert = None
    
    # --- PARTIE 1 : GÉNÉRATION DE L'ANCIENNE MATRICE DE CONFUSION ---
    pred_segment_dict, label_segment_dict = post_process_sequences(
        pred_dict,
        label_dict,
        class_names,
        thresh=0.5,
        label_convert=label_convert,
        out_dir=output_dir
    )

    # --- PARTIE 2 : GÉNÉRATION DE L'ANCIEN METRICS_FIG.PNG ---
    # Compute metrics
    metrics = compute_metrics(pred_segment_dict, label_segment_dict, class_names)
    # Plot results
    plot_metrics(metrics, out_dir=output_dir)

    # --- PARTIE 3 : GÉNÉRATION DES NOUVEAUX PLOTS DE SÉQUENCES ---
    
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


def infer(eval_loader, sequence_df, model, device, cache, num_positive_labels):
    """
    Run inference on the evaluation dataset and compute predictions and labels.
    Args:
        eval_loader (DataLoader): DataLoader for the evaluation dataset.
        sequence_df (DataFrame): DataFrame containing sequence information.
        model (torch.nn.Module): The posture model to evaluate.
        device (torch.device): Device to run the model on.
        cache (bool): Whether to use cached results or compute them.
        num_positive_labels (int): Number of positive labels in the dataset.
    Returns:
        tuple: Predicted logits and ground truth labels as dictionaries.
    """

    # Check if cache exists
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
                seq_names,
                frame_idxs,
            ) in dataloader_pbar:
                
                # Inférence sur tout le batch d'un coup
                if model.n_inputs == 2:
                    pred = model(image_tensors.to(device), posture_kpts.to(device))
                else:
                    pred = model(image_tensors.to(device))
                
                # On récupère les prédictions du batch (N, Classes)
                preds = torch.sigmoid(pred).cpu().numpy()
                labels = posture_labels.numpy()

                # On dispatche chaque image du batch dans les bons dictionnaires
                for i in range(len(seq_names)):
                    s_name = seq_names[i]
                    f_idx = frame_idxs[i].item()
                    
                    assert s_name in pred_dict, "Sequence {} not in result_dict".format(s_name)
                    
                    pred_dict[s_name][f_idx, 1:] = np.squeeze(preds[i])
                    label_dict[s_name][f_idx, :] = np.squeeze(labels[i])
                    cnt += 1

        if cache:
            data = [pred_dict, label_dict]
            with open(str(cache_file), "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pred_dict, label_dict


def post_process_sequences(
    pred_dict, label_dict, class_names, thresh=0.5, label_convert=None, out_dir=None
):
    """
    Post-process predicted and label sequences to extract valid segments and compute confusion matrix.
    Args:
        pred_dict (dict): Dictionary containing predicted logits for each sequence.
        label_dict (dict): Dictionary containing ground truth labels for each sequence.
        class_names (list): List of class names for the segments.
        thresh (float): Threshold to convert logits to binary labels.
        label_convert (dict, optional): Mapping to convert labels if required.
        out_dir (str, optional): Directory to save the confusion matrix plot.
    Returns:
        tuple: Processed predicted and label segment dictionaries.
    """
    # Extract valid label segments
    # Convert predicted logits to labels
    pred_class = []
    label_class = []
    for seq in label_dict.keys():
        valid_rows = np.all(pred_dict[seq][:, 1:] >= 0, axis=1)
        pred_dict[seq] = (pred_dict[seq] > thresh).astype(np.float64)
        pred_dict[seq][np.sum(pred_dict[seq][:, 1:], axis=1) == 0, 0] = 1
        pred_dict[seq][~valid_rows, :] = 0
        pred_class_tmp = [(np.where(vec == 1)[0]).tolist() for vec in pred_dict[seq]]
        label_class_tmp = [(np.where(vec == 1)[0]).tolist() for vec in label_dict[seq]]

        # Transform labels if required
        if label_convert is not None:
            for ii in range(len(pred_class_tmp)):
                for src, tgt in zip(label_convert["src"], label_convert["tgt"]):
                    if sorted(pred_class_tmp[ii]) == sorted(src):
                        pred_class_tmp[ii] = tgt
                        pred_dict[seq][ii][:] = 0
                        for jj in tgt:
                            pred_dict[seq][ii][jj] = 1

        pred_class_tmp = [
            elt for elt, valid in zip(pred_class_tmp, valid_rows) if valid
        ]
        label_class_tmp = [
            elt for elt, valid in zip(label_class_tmp, valid_rows) if valid
        ]
        pred_class += pred_class_tmp
        label_class += label_class_tmp

    # Compute frame-wise performance
    y_true_list = []
    y_pred_list = []
    for y_true, y_pred in zip(label_class, pred_class):
        y_true_list += [
            yt for yt in y_true for yp in y_pred
        ]  # duplicate pred-label pairs for confusion matrix
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
    if out_dir is not None:
        out_path = os.path.join(out_dir, "conf_matrix_all_classes.png")
    else:
        out_path = "conf_matrix_all_classes.png"
    cm_disp.figure_.savefig(out_path, dpi=300)
    plt.close(cm_disp.figure_)

    # Compute posture video segment detection performance
    fill_mask = [1, 0, 1]
    discard_mask = [-1, 1, -1]
    pred_segment_dict = {key: np.zeros_like(val) for key, val in pred_dict.items()}
    label_segment_dict = {key: np.zeros_like(val) for key, val in label_dict.items()}
    for seq in label_dict.keys():
        # 1. fill gaps of width 1, and remove isolated labels
        for ii in range(label_dict[seq].shape[1]):
            conv_res = np.convolve(label_dict[seq][:, ii], fill_mask, mode="same")
            label_dict[seq][conv_res == 2, ii] = 1
            conv_res = np.convolve(label_dict[seq][:, ii], discard_mask, mode="same")
            label_dict[seq][conv_res == 1, ii] = 0

            conv_res = np.convolve(pred_dict[seq][:, ii], fill_mask, mode="same")
            pred_dict[seq][conv_res == 2, ii] = 1
            conv_res = np.convolve(pred_dict[seq][:, ii], discard_mask, mode="same")
            pred_dict[seq][conv_res == 1, ii] = 0

        # 2. segment labels and predictions into posture sub-sequences
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
    """
    Compute metrics for posture video segment detection.
    Args:
        pred_segment_dict (dict): Dictionary containing predicted segment data.
        label_segment_dict (dict): Dictionary containing ground truth segment data.
        class_names (list): List of class names for the segments.
    Returns:
        dict: Dictionary containing computed metrics.
    """
    # Initialize metrics dictionary
    metrics = {
        key: {"gt_overlap": [], "pred_overlap": []} for key in label_segment_dict.keys()
    }
    for seq in label_segment_dict.keys():
        seq_data_gt = label_segment_dict[seq]
        seq_data_pred = pred_segment_dict[seq]
        metrics[seq]["gt_overlap"] = [[] for ii in range(seq_data_gt.shape[1])]
        metrics[seq]["pred_overlap"] = [[] for ii in range(seq_data_gt.shape[1])]
        out_dim = seq_data_gt.shape[1]
        for ii in range(out_dim):
            seg_instances_gt = np.unique(seq_data_gt[:, ii]).tolist()
            seg_instances_gt = [elt for elt in seg_instances_gt if elt > 0]
            seg_instances_pred = np.unique(seq_data_pred[:, ii]).tolist()
            seg_instances_pred = [elt for elt in seg_instances_pred if elt > 0]
            # Compute overlap for TP and FN
            for seg_idx in seg_instances_gt:
                seg_gt = seq_data_gt[:, ii] == seg_idx
                seg_gt_len = np.sum(seg_gt)
                overlap = np.sum(
                    seg_gt * (seq_data_pred[:, ii] > 0)
                )  # compute overlap with pred
                metrics[seq]["gt_overlap"][ii].append([seg_gt_len, overlap])
                if overlap > seg_gt_len:
                    pdb.set_trace()

            # Compute overlap for FP
            for seg_idx in seg_instances_pred:
                seg_pred = seq_data_pred[:, ii] == seg_idx
                seg_pred_len = np.sum(seg_pred)
                overlap = np.sum(
                    seg_pred * (seq_data_gt[:, ii] > 0)
                )  # compute overlap with gt
                gt_cover = 0
                for gt in np.unique(seq_data_gt[seg_pred, ii]):
                    if gt > 0 and (np.sum(seq_data_gt[seg_pred, ii] > 0) > 0):
                        gt_cover = 1
                metrics[seq]["pred_overlap"][ii].append(
                    [seg_pred_len, overlap, gt_cover]
                )
                if overlap > seg_pred_len:
                    pdb.set_trace()

    # Compute histograms
    min_seg_len = 10
    # 1. histogram of gt & pred overlaps, all class included
    gt_overlap = [
        elt
        for seq in label_segment_dict.keys()
        for ii in range(seq_data_gt.shape[1])
        for elt in metrics[seq]["gt_overlap"][ii]
        if elt[0] >= min_seg_len
    ]
    pred_overlap = [
        elt
        for seq in label_segment_dict.keys()
        for ii in range(seq_data_gt.shape[1])
        for elt in metrics[seq]["pred_overlap"][ii]
        if elt[0] >= min_seg_len
    ]
    overlap_thresh = [0.2, 0.4, 0.6, 0.8, 1.0]
    TP = []
    FN = []
    FP0 = []
    FP1 = []
    TP_population_all = len(gt_overlap)
    FP_population_all = len(pred_overlap)
    for thresh in overlap_thresh:
        gt_overlap_ok = [elt for elt in gt_overlap if (elt[1] / elt[0]) >= thresh]
        pred_overlap_ok0 = [
            elt
            for elt in pred_overlap
            if (1.0 - (elt[1] / elt[0])) >= thresh
            if elt[2] == 0
        ]
        pred_overlap_ok1 = [
            elt
            for elt in pred_overlap
            if (1.0 - (elt[1] / elt[0])) >= thresh
            if elt[2] == 1
        ]  # FP, but a gt segment is covered, so it's less worse
        TP.append((len(gt_overlap_ok) / len(gt_overlap)) * 100)
        FN.append(100.0 - TP[-1])
        FP0.append((len(pred_overlap_ok0) / len(pred_overlap)) * 100)
        FP1.append((len(pred_overlap_ok1) / len(pred_overlap)) * 100)

    # 2. histogram of gt & pred overlaps, per class
    TP_per_class = []
    FN_per_class = []
    FP0_per_class = []
    FP1_per_class = []
    TP_population_per_class = []
    FP_population_per_class = []
    for ii in range(out_dim):
        gt_overlap = [
            elt
            for seq in label_segment_dict.keys()
            for elt in metrics[seq]["gt_overlap"][ii]
            if elt[0] >= min_seg_len
        ]
        pred_overlap = [
            elt
            for seq in label_segment_dict.keys()
            for elt in metrics[seq]["pred_overlap"][ii]
            if elt[0] >= min_seg_len
        ]
        TP_tmp = []
        FN_tmp = []
        FP0_tmp = []
        FP1_tmp = []
        for thresh in overlap_thresh:
            gt_overlap_ok = [elt for elt in gt_overlap if (elt[1] / elt[0]) >= thresh]
            pred_overlap_ok0 = [
                elt
                for elt in pred_overlap
                if (1.0 - (elt[1] / elt[0])) >= thresh
                if elt[2] == 0
            ]
            pred_overlap_ok1 = [
                elt
                for elt in pred_overlap
                if (1.0 - (elt[1] / elt[0])) >= thresh
                if elt[2] == 1
            ]
            TP_tmp.append((len(gt_overlap_ok) / max(1, len(gt_overlap))) * 100)
            FN_tmp.append(100.0 - TP_tmp[-1])
            FP0_tmp.append((len(pred_overlap_ok0) / max(1, len(pred_overlap))) * 100)
            FP1_tmp.append((len(pred_overlap_ok1) / max(1, len(pred_overlap))) * 100)
        TP_per_class.append(TP_tmp)
        FN_per_class.append(FN_tmp)
        FP0_per_class.append(FP0_tmp)
        FP1_per_class.append(FP1_tmp)
        TP_population_per_class.append(len(gt_overlap))
        FP_population_per_class.append(len(pred_overlap))

    metrics = {
        "TP_all": TP,
        "FN_all": FN,
        "FP0_all": FP0,
        "FP1_all": FP1,
        "TP_per_class": TP_per_class,
        "FN_per_class": FN_per_class,
        "FP0_per_class": FP0_per_class,
        "FP1_per_class": FP1_per_class,
        "TP_population_all": TP_population_all,
        "FP_population_all": FP_population_all,
        "TP_population_per_class": TP_population_per_class,
        "FP_population_per_class": FP_population_per_class,
        "class_names": class_names,
        "overlap_thresh": overlap_thresh,
    }

    return metrics


def plot_metrics(metrics_dict, out_dir=None):

    plt.figure(figsize=(18, 12))

    n_classes = len(metrics_dict["TP_per_class"])
    width = 0.06  # the width of the bars
    x = np.array(metrics_dict["overlap_thresh"]) - 4 * width / 5
    fontsize = 10
    # Plot all-class metrics
    ax = plt.subplot(n_classes // 2 + 3, 3, 1)
    rects = ax.bar(x, metrics_dict["TP_all"], width, label="TP")
    bar_labels_TP = [
        "{}/{}".format(
            int(val * metrics_dict["TP_population_all"] / 100.0 + 0.5),
            metrics_dict["TP_population_all"],
        )
        for val in metrics_dict["TP_all"]
    ]
    ax.bar_label(rects, labels=bar_labels_TP, padding=len(x), fontsize=fontsize)

    rects = ax.bar(x + width, metrics_dict["FP1_all"], width, label="almost FP")
    bar_labels_FP1 = [
        "{}/{}".format(
            int(val * metrics_dict["FP_population_all"] / 100.0 + 0.5),
            metrics_dict["FP_population_all"],
        )
        for val in metrics_dict["FP1_all"]
    ]
    ax.bar_label(rects, labels=bar_labels_FP1, padding=len(x), fontsize=fontsize)
    rects = ax.bar(x + 2 * width, metrics_dict["FP0_all"], width, label="FP")
    bar_labels_FP1 = [
        "{}/{}".format(
            int(val * metrics_dict["FP_population_all"] / 100.0 + 0.5),
            metrics_dict["FP_population_all"],
        )
        for val in metrics_dict["FP0_all"]
    ]
    ax.bar_label(rects, labels=bar_labels_FP1, padding=len(x), fontsize=fontsize)

    ax.set_xlabel("min overlap, %", fontsize=fontsize)
    ax.set_ylabel("%", fontsize=fontsize)
    ax.set_title("All classes", fontsize=fontsize + 1)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.legend(bbox_to_anchor=(-0.2, 1.3), loc="upper left", ncols=3, fontsize=fontsize)
    ax.set_ylim(0, 140)
    plt.tight_layout(h_pad=0.5, w_pad=3.5)

    for ii in range(n_classes):
        ax = plt.subplot(n_classes // 3 + 2, 3, ii + 4)
        rects = ax.bar(x, metrics_dict["TP_per_class"][ii], width, label="TP")
        bar_labels_TP = [
            "{}/{}".format(
                int(val * metrics_dict["TP_population_per_class"][ii] / 100.0 + 0.5),
                metrics_dict["TP_population_per_class"][ii],
            )
            for val in metrics_dict["TP_per_class"][ii]
        ]
        ax.bar_label(rects, labels=bar_labels_TP, padding=len(x), fontsize=fontsize)
        rects = ax.bar(
            x + width, metrics_dict["FP1_per_class"][ii], width, label="almost FP"
        )
        bar_labels_FP1 = [
            "{}/{}".format(
                int(val * metrics_dict["FP_population_per_class"][ii] / 100.0 + 0.5),
                metrics_dict["FP_population_per_class"][ii],
            )
            for val in metrics_dict["FP1_per_class"][ii]
        ]
        ax.bar_label(rects, labels=bar_labels_FP1, padding=len(x), fontsize=fontsize)
        rects = ax.bar(
            x + 2 * width, metrics_dict["FP0_per_class"][ii], width, label="FP"
        )
        bar_labels_FP0 = [
            "{}/{}".format(
                int(val * metrics_dict["FP_population_per_class"][ii] / 100.0 + 0.5),
                metrics_dict["FP_population_per_class"][ii],
            )
            for val in metrics_dict["FP0_per_class"][ii]
        ]
        ax.bar_label(rects, labels=bar_labels_FP0, padding=len(x), fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.legend(
            bbox_to_anchor=(-0.2, 1.2), loc="upper left", ncols=3, fontsize=fontsize
        )
        ax.set_ylim(0, 140)
        ax.set_title(metrics_dict["class_names"][ii], fontsize=fontsize + 1)
        plt.tight_layout(h_pad=1.2, w_pad=3.5)

    if out_dir is not None:
        out_path = os.path.join(out_dir, "metrics_fig.png")
    else:
        out_path = "metrics_fig.png"
    plt.subplots_adjust(left=0.05)
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