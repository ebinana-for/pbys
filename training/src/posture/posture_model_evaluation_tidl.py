"""Compute posture metrics using prediction of the embedded version of the classifier
    Prediction should be built beforhand, using the C++ API
"""
import os
import pdb
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from src import PROJECT_DIR
from src.posture.dataset.posture_dataset import PostureDataset
from src.utils.config import get_config

PRED_LABEL_MAP = {
    "closetoairbag": "Too_close_to_airbag",
    "legsondashboard": "Legs_on_dashboard",
    "torsodissimetry": "Torso_dissimetry",
    "reachingback": "Reaching_back",
    "lateralbodycontact": "Lateral_body_contact",
    "oneshoulderincontact": "One_shoulder_in_contact",
    "pelvisrotation": "Pelvis rotation",
    "pelvisdrift": "Pelvis_drift",
}


def run(root_dir, config, eval_side, out_dir=None):
    """
    Run the evaluation of the posture model using TIDL predictions.
    Args:
        root_dir: Directory containing the TIDL predictions.
        config: Configuration object containing model and evaluation parameters.
        eval_side: Side to evaluate, either 'driver', 'passenger', or None.
        out_dir: Directory to save the output metrics and plots.
    Raises:
        ValueError: If the root directory does not exist or if eval_side is invalid.
    """
    # Check if the root directory exists
    if not os.path.isdir(root_dir):
        raise ValueError(f"Root directory does not exist: {root_dir}")

    # Create dataloader
    eval_dataset = PostureDataset(config, task="TEST", return_img=False, side=eval_side)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Result
    sequence_df = eval_dataset.img_df.copy()
    sequence_df["seq_name"] = sequence_df["file"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    num_positive_labels = len(config.input.positive_labels)
    pred_dict, label_dict = load_data(
        root_dir,
        eval_loader,
        sequence_df,
        cache=False,
        num_positive_labels=num_positive_labels,
    )

    # Create output directory
    if out_dir is None:
        out_dir = os.path.join(root_dir, "metrics")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Compute posture video segmentation masks
    class_names = ["correctly_seated"] + [
        eval_dataset.label_map[key] for key in range(num_positive_labels)
    ]

    label_convert = None
    pred_segment_dict, label_segment_dict = post_process_sequences(
        pred_dict,
        label_dict,
        class_names,
        thresh=0.5,
        label_convert=label_convert,
        out_dir=out_dir,
    )

    # Compute metrics
    metrics = compute_metrics(pred_segment_dict, label_segment_dict, class_names)

    # Plot results
    plot_metrics(metrics, out_dir=out_dir)


def load_data(pred_dir, eval_loader, sequence_df, cache, num_positive_labels):
    compute = True
    if cache:
        cache_dir = PROJECT_DIR / "cache"
        if not os.path.isdir(str(cache_dir)):
            os.makedirs(str(cache_dir))
        cache_file = cache_dir / "posture_eval_tidl_tmp.pkl"
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

        # Load the prediction file
        raw_pred_dict = {}
        for seq in sequence_list:
            pred_file = os.path.join(pred_dir, seq + "_out_positions.csv")
            if not os.path.isfile(pred_file):
                print("Warning: cannot find {}.".format(pred_file))
            else:
                data = pd.read_csv(pred_file)
                raw_pred_dict[seq] = data

        dataloader_pbar = tqdm(
            eval_loader, total=len(eval_loader), colour="green", leave=None
        )
        cnt = 0
        for (
            image_tensors,
            posture_labels,
            posture_kpts,
            seq_name,
            frame_idx,
        ) in dataloader_pbar:
            if seq_name[0] in pred_dict and seq_name[0] in raw_pred_dict:
                labels = np.squeeze(posture_labels.numpy())
                pred = raw_pred_dict[seq_name[0]].iloc[frame_idx.item()].values[1:]
                pred_dict[seq_name[0]][frame_idx.item(), 1:] = pred
                label_dict[seq_name[0]][frame_idx.item(), :] = labels
                cnt += 1
    if cache:
        data = [pred_dict, label_dict]
        with open(str(cache_file), "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pred_dict, label_dict


def post_process_sequences(
    pred_dict, label_dict, class_names, thresh=0.5, label_convert=None, out_dir=None
):
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
        TP.append((len(gt_overlap_ok) / max(1, len(gt_overlap))) * 100)
        FN.append(100.0 - TP[-1])
        FP0.append((len(pred_overlap_ok0) / max(1, len(pred_overlap))) * 100)
        FP1.append((len(pred_overlap_ok1) / max(1, len(pred_overlap))) * 100)

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
    n_classes = len(metrics_dict["TP_per_class"])
    width = 0.06  # the width of the bars
    x = np.array(metrics_dict["overlap_thresh"]) - 4 * width / 5
    fontsize = 3
    # Plot all-class metrics
    plt.figure()
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
    plt.savefig(out_path, dpi=300)


def main(inference_folder, output_dir=None, config_path=None, eval_side=None):
    """
    Main function to run the posture model evaluation using TIDL predictions.
    Args:
        inference_folder: Directory containing the TIDL predictions.
        output_dir: Directory to save the output metrics and plots.
        config_path: Path to the configuration file.
        eval_side: Side to evaluate, either 'driver', 'passenger', or None.
    """
    if config_path is None:
        config_path = os.path.join(inference_folder, "model/config.yaml")
    config = get_config()
    assert os.path.isfile(config_path), f"Config file not found: {config_path}"
    config.set_config(config_path)

    possible_eval_sides = ["driver", "passenger", None]
    if eval_side not in possible_eval_sides:
        raise ValueError(
            f"incorrect eval_side: {eval_side}. Must be one of: {possible_eval_sides}"
        )

    run(inference_folder, config, eval_side, output_dir)


if __name__ == "__main__":
    Fire(main)
