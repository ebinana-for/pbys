"""Age/gender model evaluation script on public datasets.
Provides a series of barplots to analyze the model performance."
"""
import argparse
import os
from src.utils.config import get_config
from src.age_gender.inference import InferenceEngineAgeGender
from src.age_gender.dataset.get_dataloader import get_eval_loader
import datetime
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.metrics import confusion_matrix

from src.age_gender.dataset.data_utils import (
    cat2age,
    age2cat,
    forvia_age_categories,
)


def evaluation(config):
    """ Main evaluation function
    Args:
        config: Configuration object containing model and evaluation parameters.
    """
    logging.info("Starting evaluation...")
    
    age_gender_inference = InferenceEngineAgeGender(config.model.checkpoint)
    eval_loader = get_eval_loader(config)
    dataset_names = [val for val in eval_loader.dataloaders.keys()]
    center_reduce_age = (
        eval_loader.dataloaders[dataset_names[0]]
        .datasets[dataset_names[0]]
        .center_reduce_age
    )
    dataset_pbar = tqdm(
        eval_loader.dataloaders.items(),
        total=len(eval_loader.dataloaders.keys()),
        colour="green",
        leave=None,
    )
    # evaluate per dataset
    results = {}
    for dataset_name, dataloader in dataset_pbar:
        dataset_pbar.set_description(f"Dataset evaluation '{dataset_name}'")
        results[dataset_name] = pd.DataFrame()
        for image_tensors, gender_labels, age_labels, *other in tqdm(dataloader, desc=f"Inferring {dataset_name}", colour="cyan", leave=False):

            # --- AJOUTE CES DEUX LIGNES ICI ---
            if image_tensors.shape[0] != config.input.batch_size:
                continue
            # ----------------------------------

            if len(other) == 3:
                ethnicity_labels = other[-1]
            else:
                ethnicity_labels = [None]
            image_tensors = image_tensors.cpu().numpy()
            # NHWC to NCHW for inference engine
            preds_raw = age_gender_inference.infer(
                np.transpose(image_tensors, (0, 2, 3, 1))
            )
            preds = [preds_raw[0][:, :2], preds_raw[0][:, 2:]]
            for gender_pred, age_pred, gender_gt, age_gt, ethnicity_gt in zip(
                preds[0],
                preds[1],
                gender_labels.cpu().numpy(),
                age_labels.cpu().numpy(),
                ethnicity_labels,
            ):
                gender_pred = ["M", "F"][np.argmax(gender_pred)]
                gender_gt = ["M", "F"][np.argmax(gender_gt)]

                if dataset_name == "Fairface":
                    age_gt = cat2age(age_gt[0])
                if config.model.heads_params.age.loss == "MSE":
                    if center_reduce_age:
                        age_pred[0] = age_pred[0] * 120 + 60

                        # print("[DEBUG] Age avant correction:", age_gt[0])

                        age_gt[0] = age_gt[0] * 120 + 60

                    # print("[DEBUG] Age avant norm:", age_gt[0])
                    age_gt = int(age_gt[0])
                    # print("DEBUG Age apres norm:", age_gt)
                    age_pred = int(age_pred[0])
                else:
                    age_gt = int(np.argmax(age_gt))
                    age_pred = int(np.argmax(age_pred))
                age_diff = abs(age_gt - age_pred)
                age_cat_pred = age2cat(age_pred, forvia_age_categories)
                age_cat_gt = age2cat(age_gt, forvia_age_categories)
                age_cat_correct = 1 if age_cat_pred == age_cat_gt else 0
                gender_correct = 1 if gender_pred == gender_gt else 0
                df_row = pd.DataFrame(
                    {
                        "age_gt": age_gt,
                        "age_pred": age_pred,
                        "age_diff": age_diff,
                        "age_cat_pred": age_cat_pred,
                        "age_cat_gt": age_cat_gt,
                        "age_cat_correct": age_cat_correct,
                        "gender_pred": gender_pred,
                        "gender_gt": gender_gt,
                        "gender_correct": gender_correct,
                        "ethnicity_gt": ethnicity_gt,
                        "counter": 1,
                    },
                    index=[0],
                )
                results[dataset_name] = pd.concat(
                    [results[dataset_name], df_row], ignore_index=True
                )

    # plots
    for dataset_name, df in results.items():

        out_folder = os.path.join(config.general.test_dir, dataset_name)
        os.makedirs(out_folder, exist_ok=True)

        print(df.age_diff.std())
        # nb per age category
        fig = px.histogram(
            df,
            x="age_diff",
            y="counter",
            histfunc="sum",
            title=f"histo for {dataset_name}",
        )
        with open(out_folder + "/histo.html", "w") as f:
            f.write(fig.to_html())

        # nb per ethnicity
        fig = px.histogram(
            df,
            x="ethnicity_gt",
            y="counter",
            histfunc="sum",
            title=f"nb de {dataset_name}",
        )
        with open(out_folder + "/histo.html", "w") as f:
            f.write(fig.to_html())

        conf_mat = confusion_matrix(df.age_gt.values, df.age_pred.values)
        fig = px.imshow(
            conf_mat,
            text_auto=True,
            labels=dict(x="age prediction", y="age ground truth"),
            title=f"Age confusion matrix for {dataset_name}",
        )
        with open(out_folder + "/age_heatmap.html", "w") as f:
            f.write(fig.to_html())

        # nb per age category
        fig = px.histogram(
            df,
            x="age_cat_gt",
            y="counter",
            histfunc="sum",
            title=f"Nb samples per age category for {dataset_name}",
        )
        fig.update_xaxes(categoryorder="array", categoryarray=forvia_age_categories)
        fig.update_layout(bargap=0.2)
        with open(out_folder + "/nb_samples_per_age_category.html", "w") as f:
            f.write(fig.to_html())

        # age accuracy per age category
        fig = px.histogram(
            df,
            x="age_cat_gt",
            y="age_cat_correct",
            histfunc="avg",
            title=f"Age accuracy per age category for {dataset_name}",
        )
        fig.update_xaxes(categoryorder="array", categoryarray=forvia_age_categories)
        fig.update_layout(bargap=0.2)
        with open(out_folder + "/age_accuracy_per_age_category.html", "w") as f:
            f.write(fig.to_html())

        # age MAE per age category
        fig = px.histogram(
            df,
            x="age_cat_gt",
            y="age_diff",
            histfunc="avg",
            title=f"Age MAE per age category for {dataset_name}",
        )
        fig.update_xaxes(categoryorder="array", categoryarray=forvia_age_categories)
        # fig = px.histogram(df,
        #                    x="age_gt",
        #                    y="counter",
        #                    histfunc='sum',
        #                    nbins=12)

        fig.update_layout(bargap=0.2)
        with open(out_folder + "/age_MAE_per_age_category.html", "w") as f:
            f.write(fig.to_html())

        # age boxplot per age category
        fig = px.box(
            df,
            x="age_cat_gt",
            y="age_diff",
            title=f"Age boxplot per age category for {dataset_name}",
        )
        fig.update_xaxes(categoryorder="array", categoryarray=forvia_age_categories)
        fig.update_layout(bargap=0.2)
        with open(out_folder + "/age_boxplot_per_age_category.html", "w") as f:
            f.write(fig.to_html())

        # gender accuracy per age
        fig = px.histogram(
            df,
            x="age_cat_gt",
            y="gender_correct",
            histfunc="avg",
            title=f"Gender accuracy per age category for {dataset_name}",
        )
        fig.update_xaxes(categoryorder="array", categoryarray=forvia_age_categories)
        fig.update_layout(bargap=0.2)
        with open(out_folder + "/gender_accuracy_per_age_category.html", "w") as f:
            f.write(fig.to_html())

        # age MAE per ethnicity
        fig = px.histogram(
            df,
            x="ethnicity_gt",
            y="age_diff",
            histfunc="avg",
            title=f"Age MAE per ethnicity for {dataset_name}",
        )
        fig.update_layout(bargap=0.2)
        with open(out_folder + "/age_MAE_per_ethnicity.html", "w") as f:
            f.write(fig.to_html())

        # age boxplot per ethnicity
        fig = px.box(
            df,
            x="ethnicity_gt",
            y="age_diff",
            title=f"Age boxplot per ethnicity for {dataset_name}",
        )
        fig.update_layout(bargap=0.2)
        with open(out_folder + "/age_boxplot_per_ethnicity.html", "w") as f:
            f.write(fig.to_html())

        # gender accuracy per ethnicity
        fig = px.histogram(
            df,
            x="ethnicity_gt",
            y="gender_correct",
            histfunc="avg",
            title=f"gender accuracy per ethnicity for {dataset_name}",
        )
        fig.update_layout(bargap=0.2)
        with open(out_folder + "/gender_accuracy_per_ethnicity.html", "w") as f:
            f.write(fig.to_html())


if __name__ == "__main__":
    # Get args
    parser = argparse.ArgumentParser("parser")
    parser.add_argument("--config_file", required=True, help="Path to input config")
    args = parser.parse_args()

    # Get config
    config = get_config()
    config.set_config(args.config_file)
    date_str = ""
    if config.general.get("create_timestamp_folder", True):
        date_str = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")

    output_dir = os.path.join(config.general.test_dir, date_str)
    config.general.set_param("test_dir", output_dir)
    os.makedirs(config.general.test_dir, exist_ok=True)

    out_config_file = os.path.join(config.general.test_dir, "config.yaml")
    config.to_yaml(out_config_file)

    logging.info("Config written successfully in file config.yaml")

    evaluation(config)
