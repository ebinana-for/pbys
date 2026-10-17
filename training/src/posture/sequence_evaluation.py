"""Evaluate position model performance on multiple sequences in the sequence-manner:
plot the sequence plots and print out the table of sequence-wise metrics.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire
from prettytable import PrettyTable
from tqdm import tqdm

def plot_start_end_annotated_segments(
    gt, pred, labels, sliding_window=10, threshold=0.5, min_pos_length=25, save_dir=None
):
    """
    Plot the beginning and end of annotated segments for each position.
    Args:
        gt (pd.DataFrame): Ground truth annotations.
        pred (pd.DataFrame): Model predictions.
        labels (dict): Dictionary mapping position names to their labels.
        sliding_window (int): Size of the sliding window for smoothing predictions.
        threshold (float): Threshold for considering a prediction as positive.
        min_pos_length (int): Minimum length of the annotated segment to be considered valid.
        save_dir (str, optional): Directory to save the plots. If None, plots are shown.
    Raises:
        ValueError: If gt or pred are not pandas DataFrames or if labels are not present in the DataFrames.
    """
    if not isinstance(gt, pd.DataFrame) or not isinstance(pred, pd.DataFrame):
        raise ValueError("Both gt and pred should be pandas DataFrames.")
    if not all(label in gt.columns for label in labels.values()):
        raise ValueError(
            "All labels must be present in the ground truth DataFrame columns."
        )
    if not all(
        label in pred.columns for label in labels.values() if label != "correctlyseated"
    ):
        raise ValueError(
            "All labels must be present in the prediction DataFrame columns."
        )

    discarded_seqs = {x: 0 for x in labels.values()}
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
    for position in labels.values():
        if position == "correctlyseated":
            continue
            
        segments = (gt[position].shift(1) != gt[position]).astype(int).cumsum()
        
        # --- CORRECTION 1 : Boucher les NaN (trous) avant la moyenne glissante pour unifier les courbes ---
        prediction = pred[position].ffill().fillna(0).rolling(sliding_window).mean()
        
        start_seqs = pd.DataFrame()
        end_seqs = pd.DataFrame()
        
        fig1, (ax1, ax1_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig1.subplots_adjust(hspace=0.05)  # Remove space between plots

        fig2, (ax2, ax2_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig2.subplots_adjust(hspace=0.05)  # Remove space between plots
        
        for segid, segment in segments.groupby(segments):
            start = segment.index[0]
            stop = segment.index[-1]
            mid = (stop + start) // 2
            
            # Check if we are inside the positive annotation sequence
            if not gt[position][start]:
                continue
            # Check if the annotation is long enough
            if stop - start < min_pos_length:
                discarded_seqs[position] += 1
                continue

            # For vizu purposes we start a bit before 0 and end a bit after
            d_back = min((20, start))
            d_front = min((20, prediction.index[-1] - stop))

            # --- CORRECTION 2 : Aligner la collecte de données sur les limites des graphiques ---
            newseq = pd.Series(
                index=range(-d_back, mid - start),
                data=prediction[start - d_back : mid].values,
                name=segid,
            )
            start_seqs = pd.concat([start_seqs, newseq], axis=1)
            
            newseq = pd.Series(
                index=range(mid - stop, d_front),
                data=prediction[mid : stop + d_front].values,
                name=segid,
            )
            end_seqs = pd.concat([end_seqs, newseq], axis=1)

            # ++++++++++++++++++++++++++++++++++++++++++++++
            ax1.plot(
                range(-d_back, mid - start),
                prediction[start - d_back : mid],
            )
            ax1.vlines(0, -0.05, 1.05, colors="black", linestyles="dashed")
            ax1.set_xlabel("Time from beginning of annotation (frames)")
            ax1.set_ylabel("Model prediction score")
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_title(f"Beginning of {position} position")
            
            ax2.plot(range(mid - stop, d_front), prediction[mid : stop + d_front])
            ax2.vlines(0, -0.05, 1.05, colors="black", linestyles="dashed")
            ax2.set_xlabel("Time from end of annotation (frames)")
            ax2.set_ylabel("Model prediction score")
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_title(f"End of {position} position")
            # ++++++++++++++++++++++++++++++++++++++++++++++

        start_seqs.sort_index(axis=0, inplace=True)
        end_seqs.sort_index(axis=0, inplace=True)
        num_start = start_seqs.count(axis=1)
        num_end = end_seqs.count(axis=1)
        start_seqs["Recall"] = (start_seqs >= threshold).sum(axis=1) / num_start
        start_seqs["Num_sequences"] = num_start
        end_seqs["Recall"] = (end_seqs >= threshold).sum(axis=1) / num_end
        end_seqs["Num_sequences"] = num_end

        # ---------------------------------------
        start_seqs["Num_sequences"].plot(
            ax=ax1_bottom,
            label="Number of Events",
            color="royalblue",
            drawstyle="steps-post",
        )

        ax1_bottom.set_xlabel("Frame after start of annotated segment")
        ax1_bottom.set_ylabel("Number of sequences", color="royalblue")

        ax1_bottom_right = ax1_bottom.twinx()  # Second y-axis
        start_seqs["Recall"].plot(
            ax=ax1_bottom_right, label="Recall", color="red", drawstyle="steps-post"
        )
        ax1_bottom_right.set_ylabel("Recall", color="red")
        ax1_bottom_right.tick_params(axis="y", labelcolor="red")
        ax1_bottom_right.set_ylim((-0.05, 1.05))

        # ---------------------------------------

        end_seqs["Num_sequences"].plot(
            ax=ax2_bottom,
            label="Number of Events",
            color="royalblue",
            drawstyle="steps-post",
        )

        ax2_bottom.set_xlabel("Frame around the end of annotated segment")
        ax2_bottom.set_ylabel("Number of sequences", color="royalblue")

        ax2_bottom_right = ax2_bottom.twinx()  # Second y-axis
        end_seqs["Recall"].plot(
            ax=ax2_bottom_right, label="Recall", color="red", drawstyle="steps-post"
        )
        ax2_bottom_right.set_ylabel("Recall", color="red")
        ax2_bottom_right.tick_params(axis="y", labelcolor="red")
        ax2_bottom_right.set_ylim((-0.05, 1.05))

        ax1.axhspan(-1, 0.5, facecolor="0.2", alpha=0.2)
        ax2.axhspan(-1, 0.5, facecolor="0.2", alpha=0.2)

        plt.tight_layout()
        if save_dir is not None:
            fig1.savefig(f"{save_dir}/{position}_start.png")
            fig2.savefig(f"{save_dir}/{position}_end.png")
            plt.close(fig1)
            plt.close(fig2)
        else:
            plt.show()
            print(
                f"Discarded {discarded_seqs[position]} segments for {position} position"
            )

def plot_start_end_annotated_segments_init(
    gt, pred, labels, sliding_window=10, threshold=0.5, min_pos_length=25, save_dir=None
):
    """
    Plot the beginning and end of annotated segments for each position.
    Args:
        gt (pd.DataFrame): Ground truth annotations.
        pred (pd.DataFrame): Model predictions.
        labels (dict): Dictionary mapping position names to their labels.
        sliding_window (int): Size of the sliding window for smoothing predictions.
        threshold (float): Threshold for considering a prediction as positive.
        min_pos_length (int): Minimum length of the annotated segment to be considered valid.
        save_dir (str, optional): Directory to save the plots. If None, plots are shown.
    Raises:
        ValueError: If gt or pred are not pandas DataFrames or if labels are not present in the DataFrames.
    """
    if not isinstance(gt, pd.DataFrame) or not isinstance(pred, pd.DataFrame):
        raise ValueError("Both gt and pred should be pandas DataFrames.")
    if not all(label in gt.columns for label in labels.values()):
        raise ValueError(
            "All labels must be present in the ground truth DataFrame columns."
        )
    if not all(
        label in pred.columns for label in labels.values() if label != "correctlyseated"
    ):
        raise ValueError(
            "All labels must be present in the prediction DataFrame columns."
        )

    discarded_seqs = {x: 0 for x in labels.values()}
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for position in labels.values():
        if position == "correctlyseated":
            continue
        segments = (gt[position].shift(1) != gt[position]).astype(int).cumsum()
        prediction = pred[position].rolling(sliding_window).mean()
        start_seqs = pd.DataFrame()
        end_seqs = pd.DataFrame()
        fig1, (ax1, ax1_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig1.subplots_adjust(hspace=0.05)  # Remove space between plots

        fig2, (ax2, ax2_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig2.subplots_adjust(hspace=0.05)  # Remove space between plots
        for segid, segment in segments.groupby(segments):
            start = segment.index[0]
            stop = segment.index[-1]
            mid = (stop + start) // 2
            # Check if we are inside the positive annotation sequence
            if not gt[position][start]:
                continue
            # Check if the annotation is long enough
            if stop - start < min_pos_length:
                discarded_seqs[position] += 1
                continue

            newseq = pd.Series(
                index=range(0, mid - start),
                data=prediction[start:mid].values,
                name=segid,
            )
            start_seqs = pd.concat([start_seqs, newseq], axis=1)
            newseq = pd.Series(
                index=range(mid - stop, 0),
                data=prediction[mid:stop].values,
                name=segid,
            )
            end_seqs = pd.concat([end_seqs, newseq], axis=1)

            # For vizu purposes we start a bit before 0 and end a bit after
            d_back = min((20, start))
            d_front = min((20, prediction.index[-1] - stop))
            # ++++++++++++++++++++++++++++++++++++++++++++++
            ax1.plot(
                range(-d_back, mid - start),
                prediction[start - d_back : mid],
            )
            ax1.vlines(0, -0.05, 1.05, colors="black", linestyles="dashed")
            ax1.set_xlabel("Time from beginning of annotation (frames)")
            ax1.set_ylabel("Model prediction score")
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_title(f"Beginning of {position} position")
            ax2.plot(range(mid - stop, d_front), prediction[mid : stop + d_front])
            ax2.vlines(0, -0.05, 1.05, colors="black", linestyles="dashed")
            ax2.set_xlabel("Time from end of annotation (frames)")
            ax2.set_ylabel("Model prediction score")
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_title(f"End of {position} position")
            # ++++++++++++++++++++++++++++++++++++++++++++++

        start_seqs.sort_index(axis=0, inplace=True)
        end_seqs.sort_index(axis=0, inplace=True)
        num_start = start_seqs.count(axis=1)
        num_end = end_seqs.count(axis=1)
        start_seqs["Recall"] = (start_seqs >= threshold).sum(axis=1) / num_start
        start_seqs["Num_sequences"] = num_start
        end_seqs["Recall"] = (end_seqs >= threshold).sum(axis=1) / num_end
        end_seqs["Num_sequences"] = num_end

        # ---------------------------------------
        start_seqs["Num_sequences"].plot(
            ax=ax1_bottom,
            label="Number of Events",
            color="royalblue",
            drawstyle="steps-post",
        )

        ax1_bottom.set_xlabel("Frame after start of annotated segment")
        ax1_bottom.set_ylabel("Number of sequences", color="royalblue")

        ax1_bottom_right = ax1_bottom.twinx()  # Second y-axis
        start_seqs["Recall"].plot(
            ax=ax1_bottom_right, label="Recall", color="red", drawstyle="steps-post"
        )
        ax1_bottom_right.set_ylabel("Recall", color="red")
        ax1_bottom_right.tick_params(axis="y", labelcolor="red")
        ax1_bottom_right.set_ylim((-0.05, 1.05))

        # ---------------------------------------

        end_seqs["Num_sequences"].plot(
            ax=ax2_bottom,
            label="Number of Events",
            color="royalblue",
            drawstyle="steps-post",
        )

        ax2_bottom.set_xlabel("Frame around the end of annotated segment")
        ax2_bottom.set_ylabel("Number of sequences", color="royalblue")

        ax2_bottom_right = ax2_bottom.twinx()  # Second y-axis
        end_seqs["Recall"].plot(
            ax=ax2_bottom_right, label="Recall", color="red", drawstyle="steps-post"
        )
        ax2_bottom_right.set_ylabel("Recall", color="red")
        ax2_bottom_right.tick_params(axis="y", labelcolor="red")
        ax2_bottom_right.set_ylim((-0.05, 1.05))

        ax1.axhspan(-1, 0.5, facecolor="0.2", alpha=0.2)
        ax2.axhspan(-1, 0.5, facecolor="0.2", alpha=0.2)

        plt.tight_layout()
        if save_dir is not None:
            fig1.savefig(f"{save_dir}/{position}_start.png")
            fig2.savefig(f"{save_dir}/{position}_end.png")
        else:
            plt.show()
            print(
                f"Discarded {discarded_seqs[position]} segments for {position} position"
            )


def evaluate_prediction(
    gt,
    prediction,
    labels,
    detection_warmup=10,
    min_pos_length=25,
    debug=False,
):
    out_dict = {}
    # Recall / false negatives
    for position in labels.values():
        if position == "correctlyseated":
            continue
        out_dict[position] = {
            "n_tp": 0,
            "n_fn": 0,
            "n_frames": 0,
            "n_segments": 0,
            "fp_segments": [],
        }
        segments = (gt[position].shift(1) != gt[position]).astype(int).cumsum()
        for _, segment in segments.groupby(segments):
            # Positive annotations only
            if not gt[position][segment.index[0]]:
                continue
            # Short segments are removed
            if segment.index[-1] - segment.index[0] < max(
                min_pos_length, detection_warmup
            ):
                continue
            out_dict[position]["n_segments"] += 1
            out_dict[position]["n_frames"] += (
                segment.index[-1] - segment.index[detection_warmup]
            )
            out_dict[position]["n_tp"] += (
                prediction[position][
                    segment.index[detection_warmup] : segment.index[-1]
                ]
            ).sum()

        out_dict[position]["n_fn"] = (
            out_dict[position]["n_frames"] - out_dict[position]["n_tp"]
        )
        out_dict[position]["mean_recall"] = (
            out_dict[position]["n_tp"] / out_dict[position]["n_frames"]
        )

    # Precision / false positives
    segments = (
        (gt["correctlyseated"].shift(1) != gt["correctlyseated"]).astype(int).cumsum()
    )
    out_dict["correctlyseated"] = {"n_segments": 0, "n_frames": 0}
    for _, segment in segments.groupby(segments):
        # Positive annotations only
        if not gt["correctlyseated"][segment.index[0]]:
            continue
        # Short segments are removed
        if segment.index[-1] - segment.index[0] < detection_warmup:
            continue
        out_dict["correctlyseated"]["n_segments"] += 1
        out_dict["correctlyseated"]["n_frames"] += (
            segment.index[-1] - segment.index[detection_warmup]
        )
        # Find all segments of bad predictions for every position
        for position in labels.values():
            if position == "correctlyseated":
                continue

            pred_segments = (
                (prediction[position].shift(1) != prediction[position])
                .astype(int)
                .cumsum()
            )
            for _, pred_segment in pred_segments.groupby(pred_segments):
                # False **Positives**
                if not prediction[position].iloc[pred_segment.index[0]]:
                    continue

                # Must start earlier than gt end
                if pred_segment.index[0] >= segment.index[-1]:
                    continue
                # Must end later than gt start (we allow some warmup)
                if pred_segment.index[-1] < segment.index[detection_warmup]:
                    continue

                # Must not be the beginning/end of the correct prediction
                if (
                    gt[position]
                    .iloc[pred_segment.index[0] : pred_segment.index[-1]]
                    .any()
                ):
                    continue

                if debug:
                    from IPython.display import display

                    print(f"Predicted class: {pred_segment.name}")
                    print(
                        f"Correctly seated start: {segment.index[0]}, end: {segment.index[-1]}"
                    )
                    print(
                        f"Pred. position start: {pred_segment.index[0]}, end: {pred_segment.index[-1]}"
                    )
                    mask = prediction[position].iloc[segment.index]
                    df = pd.DataFrame()
                    df["gt"] = segment[mask]
                    df["pred"] = prediction[position].iloc[segment.index][mask]
                    display(df)
                    print("=====================")
                out_dict[position]["fp_segments"].append(pred_segment)

    return out_dict


def evaluate_sliding_window(
    gt,
    pred,
    labels,
    sliding_window=10,
    threshold=0.5,
    detection_warmup=10,
    min_pos_length=25,
    debug=False,
):
    parameter_dict = {
        "threshold": threshold,
        "sliding_window": sliding_window,
        "detection_warmup": detection_warmup,
        "min_pos_length": min_pos_length,
    }
    prediction = pred.rolling(sliding_window).mean() > threshold
    out_dict = evaluate_prediction(
        gt,
        prediction,
        labels,
        detection_warmup=detection_warmup,
        min_pos_length=min_pos_length,
        debug=debug,
    )
    for perfs in out_dict.values():
        perfs.update(parameter_dict.copy())
    return out_dict


def initialize_results_dict(labels):
    """
    Initialize a results dictionary with keys for each label.
    """
    return {
        label: {
            "n_tp": [],
            "n_fn": [],
            "fp_frames": [],
            "fp_segments": [],
            "mean_recall": [],
            "threshold": [],
        }
        for label in labels.values()
        if label != "correctlyseated"
    }


def update_results_dict(global_dict, local_dict, threshold):
    """
    Update the global results dictionary with local results for a given threshold.
    """
    for label in global_dict.keys():
        if label == "correctlyseated":
            continue
        global_dict[label]["n_tp"].append(local_dict[label]["n_tp"])
        global_dict[label]["n_fn"].append(local_dict[label]["n_fn"])
        global_dict[label]["fp_frames"].append(
            sum(len(seg) for seg in local_dict[label]["fp_segments"])
        )
        global_dict[label]["fp_segments"].append(len(local_dict[label]["fp_segments"]))
        global_dict[label]["mean_recall"].append(local_dict[label]["mean_recall"])
        global_dict[label]["threshold"].append(threshold)

    return global_dict


def evaluate_varying_threshold(
    gt,
    pred,
    labels,
    step,
    sliding_window=10,
    detection_warmup=10,
    min_pos_length=25,
):
    """Calculate recall and False positive rates on the entire dataset for linearly spaced
    thresholds with the given step.

    Returns: Dict[str, pd.DataFrame] - dataframes with metrics, which vary with threshold
    """
    # Similar function can also be made for sliding windows
    thresholds = np.arange(1.0, 0.0, -step)
    results_dict = initialize_results_dict(labels)
    # Brute force through all thresholds
    progressbar = tqdm(thresholds)
    for threshold in progressbar:
        progressbar.set_description(f"Sweeping thresholds with step {step}")
        prediction = pred.rolling(sliding_window).mean() >= threshold
        one_eval_results = evaluate_prediction(
            gt,
            prediction,
            labels,
            detection_warmup=detection_warmup,
            min_pos_length=min_pos_length,
        )
        results_dict = update_results_dict(results_dict, one_eval_results, threshold)

    # Gather total numbers of sequences and frames, independent of the threshold
    totals = pd.DataFrame(index=["n_frames", "n_segments"])
    for position, res in one_eval_results.items():
        totals[position] = pd.Series(
            {"n_frames": res["n_frames"], "n_segments": res["n_segments"]}
        )

    # Some additional metrics for plotting
    results = {
        position: pd.DataFrame(data).set_index("threshold")
        for position, data in results_dict.items()
    }
    for position, df in results.items():
        df["FPR Segments"] = (
            df["fp_segments"] / totals.loc["n_segments", "correctlyseated"]
        )
        df["FPR Frames"] = df["fp_frames"] / totals.loc["n_frames", "correctlyseated"]

    return results


def plot_varying_thresholds(results, save_dir=None):
    for position, df in results.items():
        metrics = ["FPR Frames", "FPR Segments", "Mean Recall"]
        plt.figure()
        df.rename(columns={"mean_recall": "Mean Recall"}, errors="ignore")[
            metrics
        ].plot(
            marker=".",
            ylim=(-0.05, 1.05),
            grid=True,
        )
        plt.title(position)

        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(f"{save_dir}/{position}_metrics_threshold.png")
        else:
            plt.show()


def print_table_results(perf_dict, out_file=None):
    from prettytable import PrettyTable
    
    t = PrettyTable(
        [
            "Position",
            "Mean recall",
            "TP",       # True Positives (Frames)
            "FN",       # False Negatives (Frames)
            "FP",       # False Positives (Frames)
            "TN",       # True Negatives (Frames)
            "Total Eval Frames", 
            "Total Segments",
        ]
    )
    for position, perf in perf_dict.items():
        if position == "correctlyseated":
            t.add_row(
                [position, "---", "---", "---", "---", "---", perf["n_frames"], perf["n_segments"]]
            )
        else:
            tp = int(perf["n_tp"])
            fn = int(perf["n_fn"])
            fp = sum(len(seg) for seg in perf["fp_segments"])
            tn = "---" # Non calculé par l'approche séquentielle basée sur les événements
            
            t.add_row(
                [
                    position,
                    f"{perf['mean_recall']:.2f}",
                    tp,
                    fn,
                    fp,
                    tn,
                    perf["n_frames"], # Correspond à TP + FN
                    perf["n_segments"],
                ]
            )
    print(t)
    if out_file is not None:
        with open(out_file, "w") as f:
            f.write(str(t))
def print_out_perf(perf_dict):
    for position, perf in perf_dict.items():
        if position == "correctlyseated":
            print(f"Position: {position}")
            print(f"  Total Annotated Frames: {perf['n_frames']}")
            print(f"  Total Annotated Segments: {perf['n_segments']}")

            continue
        print(f"Position: {position}")
        print(f"  Total Annotated Frames: {perf['n_frames']}")
        print(f"  Total Annotated Segments: {perf['n_segments']}")
        print(f"  Mean Recall: {perf['mean_recall']:.2f}")
        print(f"  Number of False Positive Segments: {len(perf['fp_segments'])}")
        print(
            f"  Number of frames in False Positive Segments: {sum(len(seg) for seg in perf['fp_segments'])}"
        )
        print()


KEEP_LABELS = [
    "Too_close_to_airbag",
    "Pelvis rotation",
    "Torso_dissimetry",
    "One_shoulder_in_contact",
    "Pelvis_drift",
    "Legs_on_dashboard",
    "Reaching_back",
    "Lateral_body_contact",
    "Correctly_seated",
]
root = "../data-local/position_4/"


def read_labels(input_directory, labels_file=None):
    if labels_file is None:
        print(
            "No labels file provided, looking for synsets.txt in the input directory."
        )
        labels_file = os.path.join(input_directory, "synsets.txt")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file {labels_file} does not exist.")
    with open(labels_file, "r") as fid:
        lines = fid.readlines()
    lines = [x.strip() for x in lines]
    labels = {
        i: x.lower().replace("_", "").replace(" ", "").replace("too", "")
        for i, x in enumerate(lines)
        if x in KEEP_LABELS
    }
    return labels


def get_annotations_path(annotations_directory, pred_filename):
    # TODO this logic is not very general and may break for other people

    subfolder = pred_filename[:-18]  # remove "out_positions.csv"

    filepath = os.path.join(annotations_directory, subfolder, "default.txt")
    assert os.path.exists(filepath), f"Annotations file {filepath} does not exist."
    return filepath


def read_data(input_directory, labels, annotations_directory=None):
    if annotations_directory is None or not os.path.exists(annotations_directory):
        print(
            (
                f"Annotations directory {annotations_directory} does not exist. "
                f"Looking for annotations in the inference folder."
            )
        )
        annotations_directory = os.path.join(input_directory, "labels")
        if not os.path.exists(annotations_directory):
            raise FileNotFoundError(
                f"Annotations directory {annotations_directory} not found"
            )

    test_sequences = []
    for filename in os.listdir(input_directory):
        filepath = os.path.join(input_directory, filename)
        if not filename.endswith("out_positions.csv"):
            continue

        print(f"Loading {filepath}")

        # Load predictions
        pred = pd.read_csv(filepath)
        mask = ~(pred == -1).any(axis=1)
        pred = pred[mask].reset_index(drop=True)
        assert (
            len(set(pred.columns) - set(labels.values())) == 1
        )  # Columns should be the same

        # Load Annotations
        filepath = get_annotations_path(annotations_directory, filename)
        with open(filepath, "r") as fid:
            gt_list = fid.readlines()
        gt_list = [x.strip() for x in gt_list]
        gt = [
            {
                **{"frame": i, "seqname": filename[:-18]},
                **{v: str(k) in x.split(" ") for k, v in labels.items()},
            }
            for i, x in enumerate(gt_list)
        ]
        gt = pd.DataFrame(gt)
        gt = gt[mask].reset_index(drop=True)
        test_sequences.append((pred, gt))
    return test_sequences


def main(
    input_directory,
    output_directory=None,
    labels_filepath=None,
    annotations_directory=None,
    sliding_window=10,
    threshold=0.5,
    warmup=10,
    min_seq_length=25,
    threshold_step=0.01,
):
    """Evaluate position model performance on multiple sequences in the sequence-manner:
    plot the sequence plots and pring out the table of sequence-wise metrics.

    Args:
        input_directory (str): Path to the directory with c++ inference results: csv files <seq_name>_out_positions.csv
        output_directory (str, optional): Path to the directory where results will be saved.
            Defaults to <input_directory>/metrics.
        labels_filepath (str, optional): Path to the file with labels: a synsets.txt file.
        annotations_directory (str, optional): Path to the directory with annotations. This Path
            should contain all subfolders with default.txt files in them. Defaults to <input_directory>/labels.
        sliding_window (int, optional): Size of the sliding window for aggregating predictions. Defaults to 10.
        threshold (float, optional): Threshold for the predictions. Defaults to 0.5.
        warmup (int, optional): Number of frames to skip (not count as correct or incorrect)
            at the beginning of every annotation. Defaults to 10.
        min_seq_length (int, optional): Minimum length of the annotation sequence to be considered valid.
            Defaults to 25.
        threshold_step (float, optional): step for sweeping the threshold. Defaults to 0.01
    Returns:
        None

    """
    if output_directory is None:
        output_directory = os.path.join(input_directory, "metrics")
    os.makedirs(output_directory, exist_ok=True)
    print(f"Results will be saved to {output_directory}")
    labels = read_labels(input_directory, labels_filepath)
    list_of_sequences = read_data(
        input_directory, labels, annotations_directory=annotations_directory
    )

    all_gt = pd.concat([x[1] for x in list_of_sequences], ignore_index=True)
    all_pred = pd.concat([x[0] for x in list_of_sequences], ignore_index=True)
    plot_start_end_annotated_segments(
        all_gt,
        all_pred,
        labels,
        sliding_window=sliding_window,
        threshold=threshold,
        min_pos_length=min_seq_length,
        save_dir=output_directory,
    )

    results_dict = evaluate_sliding_window(
        all_gt,
        all_pred,
        labels,
        sliding_window=sliding_window,
        threshold=threshold,
        detection_warmup=warmup,
        min_pos_length=min_seq_length,
    )
    print(f"Summary of results for threshold {threshold}")
    print_table_results(
        results_dict,
        out_file=os.path.join(output_directory, f"threshold_{threshold}_results.txt"),
    )

    varying_threshold_results = evaluate_varying_threshold(
        all_gt,
        all_pred,
        labels,
        step=threshold_step,
        sliding_window=sliding_window,
        detection_warmup=warmup,
        min_pos_length=min_seq_length,
    )

    plot_varying_thresholds(varying_threshold_results, save_dir=output_directory)


if __name__ == "__main__":
    Fire(main)
