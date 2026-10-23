"""PostureDataset class for posture classification
This class is used to load and preprocess the posture classification dataset.
"""
import logging
from typing import Any
from pprint import pformat
import numpy as np
import pandas as pd
import cv2
import os
import torch
import glob
from sklearn.utils import resample
from functools import reduce
from src.utils.data_augmentations import DataAugPipeline

import pdb

np.random.seed(34)
IMG_WIDTH = 2560
IMG_HEIGHT = 1984
KPTS_LIST = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    # "left_knee",
    # "right_knee",
    # "left_ankle",
    # "right_ankle",
]


# List of lines to be drawn in images
KEYPT_ORDER = [
    ["keypoints/left_shoulder", "keypoints/right_shoulder"],
    ["keypoints/left_elbow", "keypoints/left_shoulder"],
    ["keypoints/right_elbow", "keypoints/right_shoulder"],
    ["keypoints/left_wrist", "keypoints/left_elbow"],
    ["keypoints/right_wrist", "keypoints/right_elbow"],
    ["keypoints/left_hip", "keypoints/left_shoulder"],
    ["keypoints/right_hip", "keypoints/right_shoulder"],
    ["keypoints/left_hip", "keypoints/right_hip"],
    # ["keypoints/left_knee", "keypoints/left_hip"],
    # ["keypoints/right_knee", "keypoints/right_hip"],
    # ["keypoints/left_ankle", "keypoints/left_knee"],
    # ["keypoints/right_ankle", "keypoints/right_knee"],
]


class PostureDataset(torch.utils.data.Dataset):
    """
    Custom dataset class storing the data for posture classification
        Methods :
            -__getitem__() : return the datas with index
            -__len__() : return the len of the datasets
    """

    def __init__(self,
                 config,
                 task: str = "TRAIN",
                 return_img: bool = True,
                 side=None,
                 **kwargs):
        """
        Initialize the PostureDataset class.
        Args:
            config: Configuration object containing dataset parameters.
            task: Task type, either 'TRAIN' or 'TEST'.
            return_img: Whether to return the image tensor.
            side: Side of the body to consider ('driver', 'passenger', or None).
        """
        super(PostureDataset, self).__init__()

        # self.root_dir = config.input.root_dir
        self.config = config
        self.task = task
        self.return_img = return_img
        self.side = side
        self.pos_labels_config = [elt for elt in config.input.positive_labels]
        self.neg_labels_config = [elt for elt in config.input.negative_labels]
        self.discard_labels = [elt for elt in config.input.discard_labels]
        self.classif_out_dim = sum(
            [elt.out_dim for _, elt in config.model.heads_params.items()]
        )

        # Sanity checks
        assert self.task.lower() in ["train", "test"], \
            "Task must be in [''TRAIN'', ''TEST'']"
        if side is not None:    
            assert self.side.lower() in ["driver", "passenger"], \
                "side must be in [''driver'', ''passenger'', None]"
        assert (
            self.classif_out_dim > 1
        ), "Classifier out dim must be >=2. Please check config"
        assert self.classif_out_dim == len(
            self.pos_labels_config
        ), "Classifier out dim is {}, so you must have {} postures as positive labels \
            ({} found). Please check config".format(
            self.classif_out_dim, self.classif_out_dim, len(self.pos_labels_config)
        )
        # self.test_ratio = config.input.test_ratio
        self.min_ratio = (
            config.input.minority_class_min_ratio
        )  # minimum proportion of minoritary class
        self.draw_keypoints = config.input.draw_keypoints
        self.one_hot_label = config.input.one_hot_label
        transforms = config.input.data_augmentations[self.task.lower()]
        self.data_aug_pipelines = []
        self.normalize_kpts = False
        if transforms:
            for name, parameters in transforms.items():
                self.data_aug_pipelines.append(DataAugPipeline(name, **parameters))
                if (
                    "custom_ops" in parameters
                    and "ImgKptsNormalize" in parameters["custom_ops"]
                ):
                    self.normalize_kpts = True

        logging.debug(f"Attributes: {pformat(self.__dict__)}")

        # Load annotations
        self.label_list = None
        self.keypt_list = None
        self.seq_list = None
        self.label_df = None
        self.keypt_df = None
        self.img_df = None
        self.label_desc_df = None
        self.get_valid_sequences()
        self.load_annotations()

        self.pos_labels = self.label_desc_df.loc[
            self.label_desc_df["name"].isin(self.pos_labels_config)
        ].index.values.tolist()
        self.neg_labels = self.label_desc_df.loc[
            self.label_desc_df["name"].isin(self.neg_labels_config)
        ].index.values.tolist()

        label_inter = set(self.pos_labels) & set(self.neg_labels)
        assert (
            len(label_inter) == 0
        ), "Label(s) {} are both in positive and negative labels. Check config".format(
            label_inter
        )

        # Build the final label val <-> label name map
        self.label_map = {ii: val for ii, val in enumerate(self.pos_labels_config)}
        # Balance positive and negative labels by oversampling the smallest class
        if task.lower() == "TRAIN":
            self.balance_dataset()

    def balance_dataset(self):
        """
        Balance the dataset by oversampling the minority class. 
        This method checks the distribution of positive and negative labels,
        and if the ratio of the minority class is below the specified minimum ratio,
        it resamples the minority class to match the majority class size.
        """
        class0 = self.label_df[
            self.label_df[["0", "1", "2", "3"]].isin(self.pos_labels).any(axis=1)
        ]
        class1 = self.label_df[
            self.label_df[["0", "1", "2", "3"]].isin(self.neg_labels).any(axis=1)
        ]

        if len(class0) < len(class1):
            majority_class = class1
            minority_class = class0
        else:
            majority_class = class0
            minority_class = class1

        if len(minority_class) / len(majority_class) < self.min_ratio:
            nb_samples = int(
                self.min_ratio * len(majority_class) + 0.5
            )  # Nb of samples required
            minority_class_upsampled = resample(
                minority_class,
                replace=True,  # sample with replacement
                n_samples=nb_samples,  # to match majority class
                random_state=42,
            )  # for reproducible results
            minority_class_upsampled = resample(
                minority_class, replace=True, n_samples=nb_samples, random_state=42
            )

        label_df_add = self.label_df.loc[minority_class_upsampled.index].copy()
        keypt_df_add = self.keypt_df.loc[minority_class_upsampled.index].copy()
        img_df_add = self.img_df.loc[minority_class_upsampled.index].copy()

        self.label_df = pd.concat([self.label_df, label_df_add])
        self.keypt_df = pd.concat([self.keypt_df, keypt_df_add])
        self.img_df = pd.concat([self.img_df, img_df_add])

    def get_valid_sequences(self) -> None:
        """
        Get valid sequences from the dataset configuration.
        This method retrieves the list of sequences for labels, keypoints, and images
        based on the dataset configuration provided in the config object.
        It checks that all subfolders are present for labels, images, and keypoints,
        and extracts unique sequence names. The lists are then sorted and stored in
        the instance variables `label_list`, `keypt_list`, and `seq_list`.
        Returns:
            None
        """
        
        label_list = []
        keypt_list = []
        seq_list = []
        config_dataset = (
            self.config.input.train
            if self.task.lower() == "train"
            else self.config.input.test
        )

        for dataset_name, data_src in config_dataset.items():
            # Check the dataset is found
            if not os.path.isdir(data_src.annot_path) or \
                not os.path.isdir(data_src.kpts_path) or \
                    not os.path.isdir(data_src.img_path):
                logging.error(
                    "Dataset {} not found at {}, skipping.".format(
                        dataset_name, os.path.dirname(data_src.annot_path)
                    )
                )
                raise FileNotFoundError(
                    "Dataset {} not found at {}".format(
                        dataset_name, os.path.dirname(data_src.annot_path)
                    )
                )
            # Get sequence names list
            label_list0 = sorted(glob.glob(data_src.annot_path + "/*"))
            keypt_list0 = sorted(glob.glob(data_src.kpts_path + "/*"))
            seq_list0 = sorted(glob.glob(data_src.img_path + "/*"))

            # Check that all subfolders are present for label, images and keypoints
            basenames_label0 = [os.path.basename(elt) for elt in label_list0]
            basenames_keypt0 = [
                os.path.splitext(os.path.basename(elt))[0] for elt in keypt_list0
            ]
            basenames_seq0 = [os.path.basename(elt) for elt in seq_list0]
            basenames0 = list(
                set(basenames_label0 + basenames_keypt0 + basenames_seq0)
            )  # extract unique elements
            basenames0 = sorted(basenames0)
            basenames = [
                elt
                for elt in basenames0
                if elt in basenames_label0
                and elt in basenames_keypt0
                and elt in basenames_seq0
            ]

            # Get final lists
            for name in basenames:
                label_list += [elt for elt in label_list0 if name in elt]
                keypt_list += [elt for elt in keypt_list0 if name in elt]
                seq_list += [elt for elt in seq_list0 if name in elt]

        self.label_list = label_list
        self.keypt_list = keypt_list
        self.seq_list = seq_list

    def load_annotations(self):
        """
        Load annotations from the label, keypoint, and image files.
        This method reads the label data from CSV files, processes the keypoint data,
        and constructs a DataFrame for images. It also filters out unwanted labels
        based on the configuration and checks for driver and passenger tags.
        The method ensures that the data is consistent across all sequences and
        prepares the final DataFrames for labels, keypoints, and images.
        It also handles the case where only driver or passenger side data is required.
        Returns:
            None
        """

        label_data = []
        keypt_data = []
        img_data = []
        label_desc_data = []

        for label_dir, keypt_file, seq in zip(
            self.label_list, self.keypt_list, self.seq_list
        ):
            # Load label data
            label_df = pd.read_csv(
                label_dir + "/default.txt",
                names=["file", "0", "1", "2", "3", "4"],
                delimiter=r"\s+",
            )
            label_df_keep = label_df[
                ~label_df[["0", "1", "2", "3", "4"]].isnull().all(axis=1)
            ].copy()
            label_df_keep["frame_index"] = label_df_keep["file"].apply(
                lambda x: int(x[6:12])
            )
            label_df = label_df_keep.sort_values(by="frame_index", ascending=True)

            label_desc = pd.read_csv(label_dir + "/synsets.txt", names=["name"])
            if len(label_desc_data) > 0:
                for elt in label_desc_data:
                    assert label_desc.equals(
                        elt
                    ), "Labels for sequence {seq} do not match others."
            label_desc_data.append(label_desc)

            # Discard labels, if any
            disc_label_num = [
                np.squeeze(label_desc_data[0]["name"].values).tolist().index(elt)
                for elt in self.discard_labels
            ]
            label_df_keep.drop(
                index=label_df_keep.loc[
                    label_df_keep[["0", "1", "2", "3", "4"]]
                    .isin(disc_label_num)
                    .any(axis=1)
                ].index,
                inplace=True,
            )
            label_df_keep.reset_index(inplace=True)

            # Load keypt data
            keypt_df = pd.read_csv(keypt_file)
            keypt_df = keypt_df.sort_values(by="frame_index", ascending=True)

            # Fill column '4' with driver or passenger label
            driver_num = label_desc.loc[
                label_desc["name"] == "Driver"
            ].index.values.item()
            passenger_num = label_desc.loc[
                label_desc["name"] == "Passenger"
            ].index.values.item()
            driver_idx = label_df.loc[
                (label_df[["0", "1", "2", "3", "4"]] == driver_num).any(axis=1)
            ].index.values
            passenger_idx = label_df.loc[
                (label_df[["0", "1", "2", "3", "4"]] == passenger_num).any(axis=1)
            ].index.values
            assert (
                len(driver_idx) != 0 or len(passenger_idx) != 0
            ), "No driver or passenger tag found in {}".format(os.path.basename(seq))
            if len(driver_idx) != 0:
                assert len(driver_idx) > 1, "A driver tag is missing for {}".format(
                    os.path.basename(seq)
                )
                assert (
                    len(driver_idx) < 3
                ), "More than 2 driver tags found for {}".format(os.path.basename(seq))
            if len(passenger_idx) != 0:
                assert (
                    len(passenger_idx) > 1
                ), "A passenger tag is missing for {}".format(os.path.basename(seq))
                assert (
                    len(passenger_idx) < 3
                ), "More than 2 passenger tags found for {}".format(
                    os.path.basename(seq)
                )
            if len(driver_idx) != 0 and len(passenger_idx) != 0:
                if driver_idx.min() < passenger_idx.min():
                    assert (
                        driver_idx.max() < passenger_idx.min()
                    ), "Driver and passenger tags are overlapping in {}".format(
                        os.path.basename(seq)
                    )
                if passenger_idx.min() < driver_idx.min():
                    assert (
                        passenger_idx.max() < driver_idx.min()
                    ), "Driver and passenger tags are overlapping in {}".format(
                        os.path.basename(seq)
                    )
            label_df['4'] = label_df['4'].astype('str')       
            if len(driver_idx) != 0:
                label_df.loc[driver_idx.min() : driver_idx.max(), "4"] = "driver"
            if len(passenger_idx) != 0:
                label_df.loc[passenger_idx.min() : passenger_idx.max(), "4"] = (
                    "passenger"
                )

            if self.side is not None:
                if self.side.lower() == "driver":
                    label_df = label_df.loc[
                        label_df["4"] == "driver"].reset_index(drop=True)
                elif self.side.lower() == "passenger":
                    label_df = label_df.loc[
                        label_df["4"] == "passenger"].reset_index(drop=True)
                else:
                    raise (NotImplementedError, 'Unkown side ''{}'''.format(self.side))
            
            # Load image list
            img_df = pd.DataFrame({"file": glob.glob(seq + "/*.jpg")})
            img_df["frame_index"] = img_df["file"].apply(
                lambda x: int(os.path.splitext(os.path.basename(x))[0][-6:])
            )
            img_df = img_df.sort_values(by="frame_index", ascending=True)
            frame_idx_list = [
                label_df["frame_index"].values.tolist(),
                keypt_df["frame_index"].values.tolist(),
                img_df["frame_index"].values.tolist(),
            ]
            common_frame_idx = reduce(np.intersect1d, frame_idx_list)
            label_df_inner = label_df.loc[
                label_df["frame_index"].isin(common_frame_idx)
            ].copy()
            keypt_df_inner = keypt_df.loc[
                keypt_df["frame_index"].isin(common_frame_idx)
            ].copy()
            img_df_inner = img_df.loc[
                img_df["frame_index"].isin(common_frame_idx)
            ].copy()

            # Add driver/passenger label to keypt_df_inner & mean boxy bbox column
            keypt_df_inner = pd.merge(
                keypt_df_inner,
                label_df_inner[["frame_index", "4"]],
                on="frame_index",
                how="inner",
            )
            keypt_df_inner["bbox_mean_col"] = (
                keypt_df_inner["box/right"] + keypt_df_inner["box/left"]
            ) / 2
            keypt_df_inner["bbox_area"] = abs(
                (keypt_df_inner["box/right"] - keypt_df_inner["box/left"])
                * (keypt_df_inner["box/bottom"] - keypt_df_inner["box/top"])
            ) / (IMG_WIDTH * IMG_HEIGHT)

            # Process keypoint frame index duplicates
            keypt_df_inner["duplicated"] = keypt_df_inner.duplicated(
                subset=["frame_index"], keep=False
            )
            if keypt_df_inner["duplicated"].any():
                box_conf_ko = ~keypt_df_inner.apply(
                    lambda x: (
                        x["bbox_mean_col"] < (IMG_WIDTH / 2) and (x["4"] == "driver")
                    )
                    or (
                        x["bbox_mean_col"] > (IMG_WIDTH / 2) and (x["4"] == "passenger")
                    )
                    or (x["bbox_area"] < 0.1),
                    axis=1,
                )
                keypt_df_inner.drop(
                    keypt_df_inner[box_conf_ko & keypt_df_inner["duplicated"]].index,
                    axis=0,
                    inplace=True,
                )

            label_df_inner.drop_duplicates(
                subset=["frame_index"], inplace=True, ignore_index=True
            )
            keypt_df_inner.drop_duplicates(
                subset=["frame_index"], inplace=True, ignore_index=True
            )
            img_df_inner.drop_duplicates(
                subset=["frame_index"], inplace=True, ignore_index=True
            )

            # DEBUG ####
            if (len(keypt_df_inner) != len(label_df_inner)) or (
                len(keypt_df_inner) != len(img_df_inner)
            ):
                pdb.set_trace()
            ###############

            label_df_inner.reset_index(inplace=True)
            keypt_df_inner.reset_index(inplace=True)
            img_df_inner.reset_index(inplace=True)
            label_data.append(label_df_inner)
            keypt_data.append(keypt_df_inner)
            img_data.append(img_df_inner)

        self.label_desc_df = label_desc_data[0]
        self.label_df = pd.concat(label_data, axis=0, ignore_index=True)
        self.keypt_df = pd.concat(keypt_data, axis=0, ignore_index=True)
        self.img_df = pd.concat(img_data, axis=0, ignore_index=True)

        # Drop frame if the body bbox does not correspond to driver or passenger side
        bbox_mean_col_driver = self.keypt_df["bbox_mean_col"].copy()
        bbox_mean_col_passenger = self.keypt_df["bbox_mean_col"].copy()
        bbox_mean_col_driver.loc[self.keypt_df["4"] == "passenger"] = 0
        bbox_mean_col_passenger.loc[self.keypt_df["4"] == "driver"] = IMG_WIDTH
        box_conf_ok = (
            ((self.keypt_df["box/right"] - self.keypt_df["box/left"]) > 1.0)
            & ((self.keypt_df["box/bottom"] - self.keypt_df["box/top"]) > 1.0)
            & (bbox_mean_col_driver < (IMG_WIDTH / 2))
            & (bbox_mean_col_passenger > (IMG_WIDTH / 2))
        )
        self.label_df = self.label_df[box_conf_ok]
        self.keypt_df = self.keypt_df[box_conf_ok]
        self.img_df = self.img_df[box_conf_ok]

        # Clean dataframes
        self.label_df.drop("index", axis=1, inplace=True)
        self.keypt_df.drop(["index", "Unnamed: 0", "duplicated"], axis=1, inplace=True)
        self.img_df.drop("index", axis=1, inplace=True)
        self.label_df.reset_index(inplace=True)
        self.keypt_df.reset_index(inplace=True)
        self.img_df.reset_index(inplace=True)

        # DEBUG ####
        if (len(self.keypt_df) != len(self.label_df)) or (
            len(self.keypt_df) != len(self.img_df)
        ):
            pdb.set_trace()
        ###############

    def draw_kpts(self, img, keypts, width=10, dropout=0.0, img_idx=None):
        """
        Draw keypoints on the image.
        Args:
            img: Input image on which to draw keypoints.
            keypts: Dictionary containing keypoints data.
            width: Width of the lines to draw.
            dropout: Probability of dropping a keypoint.
            img_idx: Index of the image for logging purposes.
        Returns:
            img: Image with keypoints drawn on it.
        """
        if self.normalize_kpts:
            scale_y, scale_x = img.shape
        else:
            scale_y, scale_x = 1.0, 1.0

        draw_lines = []
        for kpt0, kpt1 in KEYPT_ORDER:
            x0, y0, conf0 = (
                keypts[kpt0 + "/x"],
                keypts[kpt0 + "/y"],
                keypts[kpt0 + "/conf"],
            )
            x1, y1, conf1 = (
                keypts[kpt1 + "/x"],
                keypts[kpt1 + "/y"],
                keypts[kpt1 + "/conf"],
            )
            x0 = int(x0 * scale_x + 0.5)
            y0 = int(y0 * scale_y + 0.5)
            x1 = int(x1 * scale_x + 0.5)
            y1 = int(y1 * scale_y + 0.5)
            conf0 *= int(np.random.rand() > dropout)
            conf1 *= int(np.random.rand() > dropout)
            if all([x0 > 1.0, x1 > 1.0, y0 > 1.0, y1 > 1.0]):  # no 0 or 1 value
                draw_lines.append(np.array([[x0, y0], [x1, y1]], dtype=np.int32))
        img = cv2.polylines(img, draw_lines, True, 255, width)

        # Draw face keypoints
        for kpt in [
            "keypoints/nose",
            "keypoints/left_eye",
            "keypoints/right_eye",
            "keypoints/left_ear",
            "keypoints/right_ear",
        ]:
            x0, y0, conf0 = (
                int(keypts[kpt + "/x"] + 0.5),
                int(keypts[kpt + "/y"] + 0.5),
                int(keypts[kpt + "/conf"] + 0.5),
            )
            x0 = int(x0 * scale_x + 0.5)
            y0 = int(y0 * scale_y + 0.5)
            conf0 *= int(np.random.rand() > dropout)
            if all([x0, y0]):
                img = cv2.circle(img, (x0, y0), (width + 1) // 2, 255, -1)

        return img

    def multihot_encode(self, label_list, num_classes=3):
        multi_hot_label = [1 if ii in label_list else 0 for ii in range(num_classes)]
        return torch.tensor(multi_hot_label)

    def __getitem__(self, index: int) -> Any:

        img_row = self.img_df.iloc[index]
        img_path = img_row["file"]
        seq_name = os.path.basename(os.path.dirname(img_path))
        frame_index = img_row["frame_index"]
        if self.return_img:
            keypt_row0 = self.keypt_df.iloc[index]

            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Get keypoints and body bbox
            xmin = max(0, int(keypt_row0["box/left"]))
            ymin = max(0, int(keypt_row0["box/top"]))
            xmax = min(img.shape[1], int(keypt_row0["box/right"]))
            ymax = min(img.shape[0], int(keypt_row0["box/bottom"]))

            #Add of the 10% margin

            box_w = xmax - xmin
            w_margin = int(box_w * 0.2 / 2.0)
            xmin = int(np.clip(xmin - w_margin, 0, img.shape[1]))
            xmax = int(np.clip(xmax + w_margin, 0, img.shape[1]))
            box = [xmin, ymin, xmax, ymax]

            keypt_row = {
                key: val for key, val in keypt_row0.items() if "keypoints" in key
            }

            img_height = 256
            img_width = 192

            # --- CORRECTION : VÉRIFICATION DU MODE AVANT LA PIPELINE ---
            if os.getenv("USE_CPP_LETTERBOXING") == "1":
                # 1. Découpage manuel avec la bounding box
                img_crop = img[ymin:ymax, xmin:xmax]
                roi_height, roi_width = img_crop.shape[:2]
                
                # 2. Application du Letterboxing strict C++
                scalex = img_width / float(roi_width) if roi_width > 0 else 1.0
                scaley = img_height / float(roi_height) if roi_height > 0 else 1.0
                scale = min(scalex, scaley)
                
                dst_cols = int(round(roi_width * scale))
                dst_rows = int(round(roi_height * scale))
                
                resized_img = cv2.resize(img_crop, (dst_cols, dst_rows), interpolation=cv2.INTER_LINEAR)
                
                top = int((img_height - dst_rows) / 2)
                bottom = img_height - top - dst_rows
                right = int((img_width - dst_cols) / 2)
                left = img_width - right - dst_cols
                
                img_crop = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                
                # --- NOUVEAU : Projection mathématique des points clés ---
                keypt_tr = keypt_row.copy()
                for key in keypt_tr.keys():
                    if key.endswith("/x"):
                        # Soustraire xmin (crop), appliquer le scale, ajouter la bordure gauche
                        keypt_tr[key] = (keypt_tr[key] - xmin) * scale + left
                    elif key.endswith("/y"):
                        # Soustraire ymin (crop), appliquer le scale, ajouter la bordure haute
                        keypt_tr[key] = (keypt_tr[key] - ymin) * scale + top
            else:
                # Comportement PyTorch normal (Distorsion via la pipeline d'entraînement)
                try:
                    img_crop = img
                    keypt_tr = keypt_row.copy()
                    for data_aug_pipeline in self.data_aug_pipelines:
                        result = data_aug_pipeline(img_crop, box, keypoints=keypt_tr)
                        img_crop, keypt_tr = result
                except Exception as ex:
                    logging.warning(f"failed data_aug {img_row.file}. Exception trace: {ex}")

                img_crop = cv2.resize(img_crop, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

            # Draw keypoints if required
            if self.draw_keypoints:
                img_crop = self.draw_kpts(img_crop, keypt_tr, width=3, img_idx=index)
               

            img_crop = img_crop[None, ...]
            img_crop = img_crop / 255.0  # Normalization

            input_tensor = torch.from_numpy(img_crop.astype(np.float32))

            # --- EXTRACTION DES POINTS CLÉS (À CONSERVER IMPÉRATIVEMENT) ---
            posture_kpts = []
            for kpt_name in KPTS_LIST:
                key_x = [key for key in keypt_tr.keys() if kpt_name + "/x" in key]
                key_y = [key for key in keypt_tr.keys() if kpt_name + "/y" in key]
                if keypt_tr[key_x[0]] > 1.0 and keypt_tr[key_y[0]] > 1.0:
                    posture_kpts.append([keypt_tr[key_x[0]], keypt_tr[key_y[0]]])
                else:
                    posture_kpts.append([0, 0])
            posture_kpts = np.array(posture_kpts, dtype=np.float32)
            posture_kpts = posture_kpts[5:]  # remove head keypoints
            posture_kpts = posture_kpts.reshape(-1)[None, ...]  # dim 1xN
        else:
            input_tensor = torch.zeros((1, 1))
            posture_kpts = np.zeros((1, 1))

        raw_label = (
            self.label_df.iloc[index][["0", "1", "2", "3"]]
            .dropna()
            .values.astype(np.int64)
            .tolist()
        )
        assert len(raw_label) > 0, "No label found for {}".format(
            self.label_df.iloc[index]["file"]
        )

        # Change label indexes so that it meets the positive label 
        # order defined in the config
        label_name = [self.label_desc_df.loc[elt]["name"] for elt in raw_label]
        label = [
            self.pos_labels_config.index(elt) + 1
            for elt in label_name
            if elt in self.pos_labels_config
        ]

        # If at least one positive label is found, discard any negative label 
        # and consider the sample as a positive one
        if len(label) == 0:  # the image has only negative label(s)
            label = [0]

        if self.one_hot_label:
            posture_label = self.multihot_encode(
                label, num_classes=self.classif_out_dim + 1
            )
        else:
            posture_label = torch.tensor(label)

        return (
            input_tensor,
            posture_label,
            torch.from_numpy(posture_kpts),
            seq_name,
            frame_index,
        )

    def __len__(self) -> int:
        return len(self.img_df)


if __name__ == "__main__":
    from src.utils.config import get_config
    from src import PROJECT_DIR

    config = get_config()
    config_file = PROJECT_DIR / "config/train_config_posture_multi_class.yaml"
    config.set_config(str(config_file))
    dataset0 = PostureDataset(config, task="TRAIN")
    dataset1 = PostureDataset(config, task="TEST")

    ii = 0
    img0, label0, kpts0, seq_name0, frame_index0 = dataset0[ii]
    img1, label1, kpts1, seq_name1, frame_index1 = dataset1[ii]
    print(f"Train dataset: {len(dataset0)} samples")
    print(f"Test dataset: {len(dataset1)} samples")
    print(f"Train sample {ii}: {img0.shape}, label: {label0}, kpts: {kpts0.shape}")
    print(f"Test sample {ii}: {img1.shape}, label: {label1}, kpts: {kpts1.shape}")