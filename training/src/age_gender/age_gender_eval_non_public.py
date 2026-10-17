"""This script allows for a quantitative evaluation of age & gender prediction
on a non-public dataset. It requires a config file to specify the data augmentations
and model parameters. The script processes each video in the input directory.
"""

import argparse
import ast
import datetime
import logging
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

from src.age_gender.inference import InferenceEngineAgeGender
from src.utils.viewer_utils import get_face_bbox
from src.utils.config import get_config


def evaluate(input_path=None, model_path=None, output_path=None, config=None):
    """Evaluation procedure. The main steps are:
    1. Load the configuration file if provided.
    2. Load the model
    3. Create an albumentation pipeline for data augmentations if specified in the config.
    4. Loop over all files in the input directory:
        - Load the video and corresponding CSV data.
        - For each frame, extract the face crops for both occupants.    
        - Apply the albumentation pipeline if it exists.
        - Run the model on the face crop and store the predictions.
    5. Save the results in a CSV file.
    6. Plot some evaluation figures.
    Args:
        input_path (str): Path to the input directory containing videos and CSV files.
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the results and figures.
        config (dict, optional): Configuration dictionary containing data augmentations and model parameters.
    Returns:
        None
    """
    # Load config from yaml file if given
    if config is not None:
        albumentation_ops = config["input"]["data_augmentations"]["test"]["pipeline"][
            "albumentation_ops"
        ]
        p = config["input"]["data_augmentations"]["test"]["pipeline"]["p"]

        albumentation_pipeline = None
        list_data_aug = []
        for albumentation_op, params in albumentation_ops.items():
            if albumentation_op in A.__dict__:
                params2 = params.copy()
                for key in params2:
                    if isinstance(params2[key], str):
                        params2[key] = ast.literal_eval(params2[key])
                list_data_aug.append(A.__dict__[albumentation_op](**params2))

        if list_data_aug:
            albumentation_pipeline = A.Compose(list_data_aug, p=p)

        # Check if age is predicted center-reduced
        try:
            center_reduce = (config.model.heads_params.age.loss == "MSE")
        except AttributeError:
            center_reduce = False
    else:
        print("No config file given, using no data augmentations.")
        albumentation_pipeline = None
        center_reduce = False

    # Create inference engine for age and gender
    age_gender_inference = InferenceEngineAgeGender(model_path)
    logging.info("Inference Engine for Age and Gender successfully created.")

    # Get all unique files in input_path (remove duplicates by stem)
    files_to_process = os.listdir(input_path)
    files_to_process = sorted({os.path.splitext(f)[0] for f in files_to_process})

    logging.info(f"{len(files_to_process)} files to process")

    dataset_dfs = []
    # Loop over files
    with tqdm(total=len(files_to_process), desc="Processing Files") as pbar_files:
        for f in files_to_process:
            f_root = os.path.splitext(f)[0]
            logging.info(f"Processing {f_root}...")

            # Extract ground truth age and gender from filename
            gt_age = int(f_root.split("_")[2])
            gt_gender = f_root.split("_")[3]
            logging.info(f"\tFound GTs: gender {gt_gender}, age: {gt_age}")

            # Load video
            video_path = os.path.join(input_path, f_root + ".mp4")
            cap = cv2.VideoCapture(video_path)
            logging.info(
                f"\tVideo loaded, dimensions (LxHxW):"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}x"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}x"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
            )

            # Load corresponding CSV data
            csv_path = os.path.join(input_path, f_root + ".csv")
            df = pd.read_csv(csv_path)

            # Check that length of csv and video match
            if len(df) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                logging.warning(
                    f"\tLength of csv ({len(df)}) does not match video frames ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}). This video will be ignored for evaluation."
                )
            else:
                count = 0
                file_results = []

                with tqdm(
                    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    desc=f'Processing Frames of "{f_root}"',
                ) as pbar_frames:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Get crops and crop quality for both occupants
                        head_crop0, head_crop_quality0, _, bbox_0 = get_face_bbox(
                            df, occupant_id=0, frame=frame, frame_cnt=count
                        )
                        head_crop1, head_crop_quality1, _, bbox_1 = get_face_bbox(
                            df, occupant_id=1, frame=frame, frame_cnt=count
                        )

                        # Apply augmentations if pipeline exists, then infer
                        def infer_crop(crop):
                            if crop is None:
                                return None
                            if albumentation_pipeline:
                                crop = albumentation_pipeline(image=crop)["image"]
                            return age_gender_inference([crop])[0]

                        pred0 = infer_crop(head_crop0)
                        pred1 = infer_crop(head_crop1)

                        # Prepare result row (gender, score_gender, age, confidence_pred_age, head_crop_quality)
                        res = [None, None, None, None, None]

                        if pred0 is not None:
                            res = [
                                {"M": 0, "F": 1}[pred0[0]],
                                pred0[1],
                                pred0[2] if not center_reduce else pred0[2] * 120 + 60,
                                pred0[3],
                                head_crop_quality0,
                            ]

                        if pred1 is not None:
                            res = [
                                {"M": 0, "F": 1}[pred1[0]],
                                pred1[1],
                                pred1[2] if not center_reduce else pred1[2] * 120 + 60,
                                pred1[3],
                                head_crop_quality1,
                            ]

                        # Add results to list
                        file_results.append([*res, gt_gender, gt_age, f_root, count])

                        pbar_frames.update(1)
                        count += 1

                df_file_results = pd.DataFrame(
                    file_results,
                    columns=[
                        "gender",
                        "score_gender",
                        "age",
                        "confidence_pred_age",
                        "head_crop_quality",
                        "GT_gender",
                        "GT_age",
                        "file_name",
                        "frame_number",
                    ],
                )
                dataset_dfs.append(df_file_results)

            pbar_files.update(1)

    # Concatenate all results and save
    df_dataset_results = pd.concat(dataset_dfs)
    df_dataset_results.to_csv(
        os.path.join(output_path, "inference_results.csv"), index=False
    )

    # Generate plots for evaluation
    plot_figures(df_dataset_results, output_path)


def plot_age_figures(df_results, output_path):
    # Color-blind friendly colors
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442"]

    # Create directories for figures
    figures_dir = os.path.join(output_path, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    age_figures_dir = os.path.join(figures_dir, "age")
    os.makedirs(age_figures_dir, exist_ok=True)
    logging.info(f"Age figures directory created: {age_figures_dir}")

    # Plot boxplot of absolute errors and individual files plots
    absolute_error_file = []

    for n, g in df_results.groupby("file_name"):
        print(n)
        valid_df = g[~g.age.isna()]

        if len(valid_df):
            min_index = valid_df.index.values[0]
            max_index = valid_df.index[-1]

            rollings = [20]  # Only 20-frame rolling median used

            for r in rollings:
                valid_df[f"rolling_median_age_{r}"] = valid_df.age.rolling(
                    window=r, min_periods=1
                ).median()

            # Compute absolute error for each frame and rolling median
            for _, row in valid_df.iterrows():
                absolute_error_file.append([n, 0, np.abs(row.age - row.GT_age)])
                for r in rollings:
                    absolute_error_file.append(
                        [n, f"{r}", np.abs(row[f"rolling_median_age_{r}"] - row.GT_age)]
                    )

            # Plot age prediction and rolling median over time
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=valid_df.index - min_index,
                    y=valid_df.age,
                    mode="markers",
                    name="Age Prediction",
                )
            )
            for r in rollings:
                fig.add_trace(
                    go.Scatter(
                        x=valid_df.index - min_index,
                        y=valid_df[f"rolling_median_age_{r}"],
                        mode="lines",
                        name=f"{r}-median-rolling Age Prediction",
                    )
                )
            # Add cumulative mean of age
            valid_df["cumulative_mean_age"] = valid_df.age.expanding().median()
            fig.add_trace(
                go.Scatter(
                    x=valid_df.index - min_index,
                    y=valid_df.cumulative_mean_age,
                    mode="lines",
                    name="Cumulative Mean Age",
                )
            )
            # Add ground truth line
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, max_index - min_index),
                    y=[valid_df.GT_age.values[0]] * (max_index - min_index),
                    mode="lines",
                    name="Ground Truth",
                )
            )
            with open(os.path.join(age_figures_dir, f"{n}.html"), "w") as f:
                f.write(fig.to_html())
        else:
            print(f"\tNo valid predictions for {n}, skipping...")

    # Boxplot of absolute errors (per file and overall)
    df = pd.DataFrame(
        absolute_error_file, columns=["file_name", "integration", "absolute_error"]
    )
    df["set"] = "individual"
    df_global = df.copy(deep=True)
    df_global["file_name"] = "Dataset Overall"
    df_global["set"] = "Overall"
    df = pd.concat([df, df_global], axis=0)

    fig = px.box(df, x="file_name", y="absolute_error", color="integration")
    fig.update_traces(boxmean=True)
    index_global_result = len(df_results.file_name.unique()) + 0.5
    fig.add_vrect(
        x0=index_global_result - 1,
        x1=index_global_result + 0.1,
        line_width=0,
        fillcolor="grey",
        opacity=0.2,
    )
    fig.add_hline(y=8, line_width=2, line_dash="dash")
    fig.update_layout(
        title="Box plots of age prediction absolute errors with different integration windows",
        xaxis_title="File Name",
        yaxis_title="Age Absolute Error (years)",
    )
    with open(os.path.join(figures_dir, "age_boxplot.html"), "w") as f:
        f.write(fig.to_html())

    # Scatter plot: predicted age vs ground truth
    df_results_valid = df_results[~df_results.age.isna()]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_results_valid.GT_age, y=df_results_valid.age, mode="markers")
    )
    fig.update_layout(
        title="Scatter of age predictions with respect to ground truths",
        xaxis_title="Ground Truth (years)",
        yaxis_title="Frame Predictions (years)",
    )
    with open(os.path.join(figures_dir, "age_error_scatter.html"), "w") as f:
        f.write(fig.to_html())

    # Boxplot: absolute error vs ground truth age
    df_results_valid["error_age"] = np.abs(
        df_results_valid.age - df_results_valid.GT_age
    )
    fig = px.box(df_results_valid, y="error_age", x="GT_age")
    fig.update_traces(boxmean=True)
    fig.add_hline(y=8, line_width=2, line_dash="dash")
    fig.update_layout(
        title="Boxplots of absolute age error with respect to ground truth age",
        xaxis_title="Ground Truth Age (years)",
        yaxis_title="Absolute Prediction Error (years)",
    )
    with open(os.path.join(figures_dir, "age_bloxplot_by_age.html"), "w") as f:
        f.write(fig.to_html())

    # Boxplot: absolute error vs ground truth gender
    fig = px.box(df_results_valid, y="error_age", x="GT_gender", points="all")
    fig.update_traces(boxmean=True, boxpoints=False)
    fig.add_hline(y=8, line_width=2, line_dash="dash")
    fig.update_layout(
        title="Boxplots of absolute age error with respect to ground truth gender",
        xaxis_title="Ground Truth Gender",
        yaxis_title="Absolute Prediction Error (years)",
    )
    with open(os.path.join(figures_dir, "age_bloxplot_by_gender.html"), "w") as f:
        f.write(fig.to_html())

    # Boxplot: error for first N seconds of video
    data = []
    for s in np.arange(10, 350, 10):
        df_s = df_results_valid[df_results_valid.frame_number < s].copy()
        df_s["error_age"] = np.abs(df_s.age - df_s.GT_age)
        df_s["s"] = s
        data.append(df_s)
    data = pd.concat(data)
    fig = px.box(data, y="error_age", x="s")
    fig.update_traces(boxmean=True)
    fig.update_layout(
        title="Boxplots of absolute age error with respect to number of seconds from beginning of recording",
        xaxis_title="Time from beginning of recording (seconds)",
        yaxis_title="Absolute Prediction Error (years)",
    )
    with open(
        os.path.join(figures_dir, "boxplot_of_age_for_N_first_frames.html"), "w"
    ) as f:
        f.write(fig.to_html())


def plot_gender_figures(df_results, output_path):
    # Color-blind friendly colors
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442"]

    # Create directory to save gender figures
    figures_dir = os.path.join(output_path, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    valid_df = df_results[~df_results.gender.isna()].copy()
    valid_df["gender"] = valid_df["gender"].apply(lambda x: "M" if int(x) == 0 else "F")

    from sklearn.metrics import confusion_matrix

    # Compute normalized confusion matrix for gender prediction
    conf_matrix = confusion_matrix(
        valid_df["GT_gender"], valid_df["gender"], labels=["M", "F"], normalize="true"
    )
    conf_matrix = np.fliplr(conf_matrix)
    classes = ["F", "M"]
    inv_classes = ["M", "F"]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=conf_matrix,
            x=classes,  # Predicted labels
            y=inv_classes,  # True labels
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(title="Gender Confusion Matrix")
    fig.update_xaxes(title_text="Predicted Gender")
    fig.update_yaxes(title_text="True Gender")
    with open(os.path.join(figures_dir, "gender_confidence_matrix.html"), "w") as f:
        f.write(fig.to_html())


def plot_detection_figures(df_results, output_path):
    # Color-blind friendly colors
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442"]

    # Create directories for detection figures
    figures_dir = os.path.join(output_path, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    detection_figures_dir = os.path.join(figures_dir, "detections")
    os.makedirs(detection_figures_dir, exist_ok=True)
    logging.info(f"Detections figures directory created: {detection_figures_dir}")

    detection_rates = []
    filenames = []
    total_n_detections = 0

    # Loop over results dataframe by file
    for n, g in df_results.groupby("file_name"):
        print(n)
        f_n_detections = len(g[~g.gender.isna()])
        print(f"\tDetections: {f_n_detections}")

        filenames.append(n)
        total_n_detections += f_n_detections

        detection_rate = f_n_detections / len(g) * 100
        detection_rates.append(detection_rate)
        print(f"\tDetection Rate:{np.round(detection_rate, 2)} %")

        detections = np.where(g.gender.isna(), 0, 1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=g.reset_index().index,
                y=detections,
                mode="lines",
                marker=dict(color=colors[1]),
                name="Detections",
            )
        )
        fig.update_layout(
            title=f"File name: {n} | Detection Rate:{np.round(detection_rate, 2)} %",
            xaxis_title="Frame number",
            yaxis_title="Number of Detections",
        )
        with open(
            os.path.join(detection_figures_dir, f"detections_{n}.html"), "w"
        ) as f:
            f.write(fig.to_html())

    # Add data for global plot
    print(f"Global Detections: {total_n_detections}")
    detection_rates.append((total_n_detections / len(df_results)) * 100)
    filenames.append("Global")

    # Plot dataset detection rates figure
    fig = go.Figure()
    display_opacity = [0.5] * (len(filenames) - 1) + [1]
    display_colors = [colors[0]] * len(filenames)
    display_dict = dict(color=display_colors, opacity=display_opacity)
    fig.add_trace(
        go.Bar(
            x=filenames,
            y=detection_rates,
            marker=display_dict,
            name="Detection Rate",
        )
    )
    fig.update_layout(
        title="Percentage of frames with face detections",
        xaxis_title="File Name",
        yaxis_title="Detection Rate (%)",
    )
    with open(os.path.join(figures_dir, "detection_rates.html"), "w") as f:
        f.write(fig.to_html())


def plot_figures(df_results, output_path):
    # Generate all evaluation plots
    logging.info("Plotting age figures...")
    plot_age_figures(df_results, output_path)
    logging.info("Plotting detection figures...")
    plot_detection_figures(df_results, output_path)
    logging.info("Plot gender figures...")
    plot_gender_figures(df_results, output_path)


if __name__ == "__main__":
    """Main function to run the evaluation script.
    It sets up the argument parser, loads the configuration, creates an output directory,
    and calls the evaluate function.
    """
    parser = argparse.ArgumentParser("Age and gender evaluation for non-public dataset")
    parser.add_argument(
        "-c", "--config_file_path", required=False, help="Path to config file path"
    )
    args = parser.parse_args()

    logging.info("Starting non-public dataset evaluation...")

    # Load config and set config file path if provided
    config = get_config()
    config.set_config(args.config_file_path)

    # Create unique output directory for this evaluation run
    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
    output_dir = os.path.join(config.general.test_dir, date_str)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory created: {output_dir}")

    evaluate(
        input_path=config.general.non_public_video_path,
        model_path=config.model.checkpoint,
        output_path=output_dir,
        config=config,
    )
