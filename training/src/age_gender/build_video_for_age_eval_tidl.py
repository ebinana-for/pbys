""" Convert public age/gender datasets to a video with metadata for TIDL evaluation."""

import os
import cv2
import fire
import numpy as np
import pandas as pd
import torch
import tqdm

from src.age_gender.dataset.get_dataloader import get_eval_loader
from src.utils.config import get_config

NUM_DEBUG = 55607


def compose_video_and_metadata(config, output_folder, debug=False):
    """
    Compose video and metadata for the evaluation datasets.
    Args:
        config: Configuration object containing model and evaluation parameters.
        output_folder: Folder where the output video and metadata will be stored.
        debug: If True, only a subset of the dataset will be processed for debugging purposes.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")
    
    eval_loader = get_eval_loader(config)

    dataframe_colums = [
        "/trackingState/occupants/0/bodyKeyPoints/1/value/0",  # left eye keypoint x,
        "/trackingState/occupants/0/bodyKeyPoints/1/value/1",  # left eye keypoint y,
        "/trackingState/occupants/0/bodyKeyPoints/1/quality",
        "/trackingState/occupants/0/bodyKeyPoints/2/value/0",  # right eye keypoint x,
        "/trackingState/occupants/0/bodyKeyPoints/2/value/1",  # right eye keypoint y,
        "/trackingState/occupants/0/bodyKeyPoints/2/quality",
        "/trackingState/occupants/0/bodyKeyPoints/3/value/0",  # left ear keypoint x,
        "/trackingState/occupants/0/bodyKeyPoints/3/value/1",  # left ear keypoint y,
        "/trackingState/occupants/0/bodyKeyPoints/3/quality",
        "/trackingState/occupants/0/bodyKeyPoints/4/value/0",  # right ear keypoint x,
        "/trackingState/occupants/0/bodyKeyPoints/4/value/1",  # right ear keypoint y,
        "/trackingState/occupants/0/bodyKeyPoints/4/quality",
        "/trackingState/occupants/0/bodyKeyPoints/25/value/0",  # chin keypoint x,
        "/trackingState/occupants/0/bodyKeyPoints/25/value/1",  # chin keypoint y,
        "/trackingState/occupants/0/bodyKeyPoints/25/quality",
    ]

    # Infer the eval dataset
    for dataset_name, dataloader in eval_loader.dataloaders.items():
        # Initialize and prepare everything
        videowriter = cv2.VideoWriter(
            os.path.join(output_folder, f"{dataset_name}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (1024, 1024),  # TODO image shape
            isColor=False,
        )

        age_label_sequence = []
        gender_label_sequence = []
        path_label_sequence = []
        metadata = []
        # Iterate through the dataloader
        dataset_pbar = tqdm.tqdm(
            dataloader,
            colour="green",
            leave=None,
            total=NUM_DEBUG if debug else len(dataloader),
        )
        dataset_pbar.set_description(f"Video creation '{dataset_name}'")
        for image_tensors, gender_labels, age_labels, img_path, bbox in dataset_pbar:
            img = (image_tensors[0][0].cpu().numpy() * 255).astype(np.uint8)
            out_image = np.zeros((1024, 1024), dtype=np.uint8)
            out_image[0 : img.shape[0], 0 : img.shape[1]] = img
            videowriter.write(out_image)
            # Extract virtual face keypoint for head bbox computation
            eye_dist = (img.shape[1] - 1) / 3.2
            eye_to_chin = (img.shape[0] - 2) / 2.2 / 0.9
            l_ear = [0, (img.shape[0] - 2) // 2]
            r_ear = [img.shape[1], (img.shape[0] - 2) // 2]
            eye_y = (img.shape[0] - 2) * 0.4 / 0.9
            l_eye = [
                (img.shape[1] - 1 - eye_dist) // 2,
                eye_y,
            ]
            r_eye = [
                (img.shape[1] - 1 + eye_dist - 1) // 2,
                eye_y,
            ]
            chin = [(img.shape[1] - 1) // 2, eye_y + eye_to_chin]
            keypoint_list = [
                l_eye[0],
                l_eye[1],
                1,
                r_eye[0],
                r_eye[1],
                1,
                l_ear[0],
                l_ear[1],
                1,
                r_ear[0],
                r_ear[1],
                1,
                chin[0],
                chin[1],
                1,
            ]

            # ==================================================================
            # Copy of the calculation from C++ to make sure the crop is correct
            interEyeDistance = np.sqrt(
                (l_eye[0] - r_eye[0]) ** 2 + (l_eye[1] - r_eye[1]) ** 2
            )
            eye_to_chin_vert_distance = chin[1] - (r_eye[1] + l_eye[1]) / 2.0

            center_x = (r_ear[0] + l_ear[0]) / 2.0
            center_y = (l_eye[1] + r_eye[1]) / 2.0
            width = interEyeDistance * 3.2
            height = eye_to_chin_vert_distance * 2.2

            faceCrop = {}
            faceCrop["xmin"] = int(round(center_x - width / 2.0 + 0.5))
            faceCrop["ymin"] = int(round(center_y - 0.8 * height / 2.0 + 0.5))
            faceCrop["xmax"] = int(round(center_x + width / 2.0 + 0.5))
            faceCrop["ymax"] = int(round(center_y + height / 2.0 + 0.5))

            width = float(faceCrop["xmax"] - faceCrop["xmin"])
            height = float(faceCrop["ymax"] - faceCrop["ymin"])

            if (
                faceCrop["xmax"] > 1024
                or faceCrop["ymax"] > 1024
                or faceCrop["xmin"] < 0
                or faceCrop["ymin"] < 0
            ):
                print(
                    f"Calculated width: {width}, height: {height}, image width {img.shape[1]}, height, {img.shape[0]}"
                )
                print(
                    f'ymin: {faceCrop["ymin"]}, ymax: {faceCrop["ymax"]}, xmin {faceCrop["xmin"]}, xmax, {faceCrop["xmax"]}'
                )
            # ==================================================================

            age_label = age_labels.numpy().squeeze()
            gender_label = torch.argmax(gender_labels).numpy().item()  # 0: M ; 1: F
            age_label_sequence.append(age_label)
            gender_label_sequence.append(gender_label)
            path_label_sequence.append(img_path[0])
            metadata.append(keypoint_list)

            # Early exit for debug purposes
            if debug and len(gender_label_sequence) == NUM_DEBUG:
                break

        # Store age & gender label for the current sequence to a dataframe and export to csv
        agegender = pd.DataFrame(
            {
                "age": age_label_sequence,
                "gender": gender_label_sequence,
                "path": path_label_sequence,
            }
        )
        agegender.to_csv(
            os.path.join(output_folder, f"{dataset_name}_agegender.csv"),
            index=False,
            sep=",",
        )
        # Store meta data for the current sequence, using dataframe_colums as column names (Smarteye formatting)
        assert len(dataframe_colums) == len(
            metadata[0]
        ), f"Metadata length mismatch: {len(dataframe_colums)} != {len(metadata[0])}"
        keypoints = pd.DataFrame(metadata, columns=dataframe_colums)
        keypoints.to_csv(
            os.path.join(output_folder, f"{dataset_name}.csv"),
            index=False,
            sep=",",
        )

        # Close videowriter
        videowriter.release()

        for x in [
            dataset_pbar,
            dataloader,
            videowriter,
            keypoints,
            agegender,
            metadata,
            age_label_sequence,
            gender_label_sequence,
            path_label_sequence,
        ]:
            del x


def plot_sample(img_path, bbox, label, pred, out_dir):
    img = cv2.imread(img_path)
    if img is None:
        return
    img = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    # resize to 224x224 for lisibility, and set to 3 channels
    img_resize = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    if len(img_resize.shape) < 3:
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_GRAY2BGR)

    txt1 = "GT = {}".format(label)
    txt2 = "Pred = {}".format(pred)
    img_resize = cv2.putText(
        img_resize,
        txt1,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    img_resize = cv2.putText(
        img_resize,
        txt2,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img_resize)


def main(config_path, output_folder, debug=False):

    assert os.path.exists(config_path), f"Config file {config_path} does not exist"

    # Get config
    config = get_config()
    config.set_config(str(config_path))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Video and labels will be stored to: {output_folder}")

    compose_video_and_metadata(config, output_folder, debug=debug)


if __name__ == "__main__":
    fire.Fire(main)
