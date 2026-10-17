"""Evaluation script for age/gender estimation models"""

import logging
from tqdm import tqdm
import torch
import torchvision
import numpy as np
import cv2

__author__ = "Probayes"


def eval(eval_datamodule, model, heads_list, writer, epoch, device, config):
    """Evaluation function for posture models.
    Args:
        eval_datamodule : The datamodule containing the evaluation datasets.
        model : The posture classification model to evaluate.
        heads_list : List of heads in the model, each containing information about the loss and name.
        writer : TensorBoard writer for logging results.
        epoch : Current epoch number for logging.
        device : Device to run the evaluation on (CPU or GPU).
        config : Configuration object containing model and evaluation parameters.
    Returns:
        List of accuracy or mean absolute error for each head in the model.

    This function performs the following steps:
    1. Initializes the model in evaluation mode.
    2. Iterates over each dataset in the datamodule.
    3. For each dataset, iterates over the dataloader to get image tensors and labels.
    4. Computes predictions using the model and compares them with ground truth labels.
    """
    logging.info("Evaluating model on datasets:")
    for dataset_name in eval_datamodule.dataloaders.keys(): # Log dataset names
        logging.info(f"- {dataset_name}")
    logging.info("-" * 80)
    logging.info("Starting evaluation...")
    if not eval_datamodule.dataloaders: # Check if there are any datasets to evaluate
        logging.info("No datasets found for evaluation.")
        return []
    
    model.eval()
    sum_gender_accuracy = 0.0
    sum_age_mae = 0.0
    dataset_names = [val for val in eval_datamodule.dataloaders.keys()]
    center_reduce_age = (
        eval_datamodule.dataloaders[dataset_names[0]]
        .datasets[dataset_names[0]]
        .center_reduce_age
    )

    logging.info("-" * 80)

    dataset_pbar = tqdm(
        eval_datamodule.dataloaders.items(),
        total=len(eval_datamodule.dataloaders.keys()),
        colour="green",
        leave=None,
    )

    # evaluate per dataset
    for dataset_name, dataloader in dataset_pbar:
        dataset_pbar.set_description(f"Dataset evaluation '{dataset_name}'")

        correct_gender = 0
        age_abs_err = 0
        nb_imgs = 0
        # test loop

        for image_tensors, gender_labels, age_labels, *_ in dataloader:

            # one_hot to class
            gender_labels = torch.argmax(gender_labels, dim=1)
            gender_labels = gender_labels.numpy()
            age_labels = age_labels.numpy()
            logits = model(image_tensors.to(device))

            preds = []
            for idx, elt in enumerate(heads_list):
                if "BCE" in elt["loss"]:
                    if elt["name"] == "age":
                        # pred = torch.sigmoid(logits[idx]).cpu().numpy()
                        pred = (
                            torch.nn.functional.softmax(logits[idx], dim=1)
                            .cpu()
                            .numpy()
                        )
                        pred = np.sum(
                            pred
                            * np.arange(0, config.model.heads_params.age.out_dim, 1),
                            axis=1,
                        ) / np.sum(pred, axis=1)
                    else:
                        pred = torch.argmax(logits[idx], dim=1).cpu().numpy()
                    preds.append(pred)
                elif elt["loss"] == "MSE":
                    if center_reduce_age:
                        logits[idx] = logits[idx] * 120 + 60
                    preds.append(np.squeeze(logits[idx].cpu().numpy()))
                else:
                    logging.error("Unkown loss {}".format(elt["loss"]))

                if elt["name"] == "gender":
                    correct_gender += (preds[idx] == gender_labels).sum()
                elif elt["name"] == "age":
                    if "BCE" in elt["loss"]:
                        age_abs_err += np.abs(
                            preds[idx] - np.argmax(age_labels, axis=1)
                        ).sum()
                    else:
                        if center_reduce_age:
                            age_labels0 = age_labels * 120 + 60
                        else:
                            age_labels0 = age_labels
                        age_abs_err += np.abs(
                            preds[idx] - np.squeeze(age_labels0)
                        ).sum()

            if nb_imgs == 0:  # Display some images in tensorboard
                images_to_show = image_tensors[: config.eval.nb_imgs_in_tensorboard]
                preds_to_show = [
                    pred[: config.eval.nb_imgs_in_tensorboard] for pred in preds
                ]
                gt_to_show = []
                for elt in heads_list:
                    if elt["name"] == "gender":
                        gt_to_show.append(
                            gender_labels[: config.eval.nb_imgs_in_tensorboard]
                        )
                    elif elt["name"] == "age":
                        if "BCE" in elt["loss"]:
                            gt_to_show.append(
                                np.argmax(age_labels, axis=1)[
                                    : config.eval.nb_imgs_in_tensorboard
                                ]
                            )
                        else:
                            gt_to_show.append(
                                np.squeeze(age_labels)[
                                    : config.eval.nb_imgs_in_tensorboard
                                ]
                            )
            nb_imgs += image_tensors.size(0)

        lst_img_final = []

        for idx, image in enumerate(images_to_show):
            preds = [elt[idx] for elt in preds_to_show]
            gts = [elt[idx] for elt in gt_to_show]
            image = np.array(image.cpu()).transpose((1, 2, 0)) * 255
            image = np.ascontiguousarray(image, dtype=np.uint8)

            for elt, pred, gt in zip(heads_list, preds, gts):
                if elt["name"] == "gender":
                    color = (0, 255, 0) if gt == pred else (255, 0, 0)
                    image = cv2.putText(
                        image,
                        str(["M", "F"][int(pred)]),
                        (10, 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        color,
                        2,
                    )
                elif elt["name"] == "age":
                    image = cv2.putText(
                        image,
                        "Age pred/gt",
                        (60, 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.2,
                        (255, 255, 0),
                        1,
                    )
                    image = cv2.putText(
                        image,
                        "{0:.1f} / {1:.1f}".format(pred, gt),
                        (60, 50),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.7,
                        (255, 255, 0),
                        2,
                    )

            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
            lst_img_final.append(image)

        grid = torchvision.utils.make_grid(lst_img_final, nrow=3)
        writer.add_image(f"test_images/{dataset_name}", grid, epoch)

        for elt in heads_list:
            if elt["name"] == "gender":
                gender_accuracy = correct_gender / nb_imgs
                writer.add_scalar(
                    f"accuracy/gender_acc/{dataset_name}", gender_accuracy, epoch
                )
                sum_gender_accuracy += gender_accuracy
            elif elt["name"] == "age":
                age_mae = age_abs_err / nb_imgs
                writer.add_scalar(f"accuracy/age_mae/{dataset_name}", age_mae, epoch)
                sum_age_mae += age_mae

    out = []
    for elt in heads_list:
        if elt["name"] == "gender":
            total_gender_accuracy = sum_gender_accuracy / len(
                eval_datamodule.dataloaders
            )
            writer.add_scalar(
                "accuracy/total_gender_accuracy", total_gender_accuracy, epoch
            )
            out.append(total_gender_accuracy)
        elif elt["name"] == "age":
            total_age_mae = sum_age_mae / len(eval_datamodule.dataloaders)
            writer.add_scalar("accuracy/total_age_mae", total_age_mae, epoch)
            out.append(total_age_mae)

    return out
