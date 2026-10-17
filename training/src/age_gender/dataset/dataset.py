"""Pytorch dataset building for age/gender model training and evaluation"""

from __future__ import annotations
import logging
import random
from pprint import pformat
from typing import Any, Optional
import numpy as np
import pandas as pd
import cv2
import os
import torch
import zlib

from src.utils.data_augmentations import DataAugPipeline

import pdb


class AgeGenderDataset(torch.utils.data.Dataset):
    """
    Custom dataset class storing the data for age & gender estimation
        Arguments :
            :img_path: str -> the path to the image directory
            :img_path: str -> the path to the annot csv file
            :split: str -> The type of image used (in ['TEST','TRAIN'])
            :datasets: list[str] -> list of all datasets to filter
        Methods :row.age
            -__getitem__() : return the datas with index
            -__len__() : return the len of the datasets
    """

    def __init__(
        self,
        dataset_name: str,
        img_path: str,
        annot_path: str,
        transforms: dict,
        split: Optional[str] = None,
        dataset_size: Optional[int] = None,
        one_hot_gender=True,
        one_hot_age=False,
        center_reduce_age=False,
        age_ranges=None,
        return_ethnicity=False,
        ethnicity_csv=None,
        seed: int = 42,
        **kwargs,
    ) -> None:

        self.dataset_name = dataset_name
        self.img_path = img_path
        self.annot_path = annot_path
        self.split = split
        self.dataset_size = dataset_size
        self.one_hot_gender = one_hot_gender
        self.one_hot_age = one_hot_age
        self.center_reduce_age = center_reduce_age
        self.age_ranges = age_ranges
        self.return_ethnicity = return_ethnicity
        self.seed = int(seed)
        self.epoch = 0
        self.df: pd.DataFrame = pd.DataFrame()
        self.gender_label_map = {"M": 0, "F": 1}
        self.data_aug_pipelines = []

        if transforms:
            for name, parameters in transforms.items():
                self.data_aug_pipelines.append(DataAugPipeline(name, **parameters))

        # Load annotations
        logging.debug(f"Attributes: {pformat(self.__dict__)}")
        self.load_annotations()

        if ethnicity_csv is not None:
            self.df_ethnicity = pd.read_csv(ethnicity_csv)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _seed_sample_augmentations(self, index: int) -> None:
        seed_material = f"{self.seed}:{self.epoch}:{self.dataset_name}:{index}"
        sample_seed = zlib.crc32(seed_material.encode("utf-8"))
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        for pipeline in self.data_aug_pipelines:
            compose = getattr(pipeline, "albumentation_pipeline", None)
            if compose is not None:
                compose.set_random_seed(sample_seed)

    def load_annotations(
        self,
    ) -> None:
        df = pd.read_csv(self.annot_path)

        # drop all rows that contain -1
        df = df.drop(df[df["age"] == -1].index)
        df = df.drop(df[df["gender"] == -1].index)
        df = df.drop(df[df["age"] == "-1"].index)
        df = df.drop(df[df["gender"] == "-1"].index)

        if "sanity" in df.columns:
            df = df.drop(df[df["sanity"] == 0].index)
            df = df.drop(df[df["sanity"] == "0"].index)

        if "age_pred" in df.columns:
            df["age_err"] = (df["age_pred"] - df["age"]).abs()
            df["age_err_rel"] = ((df["age_pred"] - df["age"]) / df["age"]).abs()

            df = df.drop(df[(df["age_err"] > 25) & (df["age_err_rel"] > 0.5)].index)
            df = df.drop(
                df[(df["age_err_rel"] > 4)].index
            )  # used for small age values filtering
        # else:
        #     pdb.set_trace()
        if self.dataset_size:
            df = df.head(self.dataset_size)
        self.df = df
        logging.info(f"Loaded {len(self.df)} annotations")

    def store_image(self, row, out_dir):
        age = int(row.age)
        age_pred = int(row.age_pred)
        # face box coords
        xmin = int(row.face_x0)
        ymin = int(row.face_y0)
        xmax = int(row.face_x1)
        ymax = int(row.face_y1)
        img = cv2.imread(os.path.join(self.img_path, row.img_name))
        img_crop = img[ymin:ymax, xmin:xmax]
        img_crop = cv2.resize(img_crop, (224, 224), interpolation=cv2.INTER_AREA)
        txt1 = "Age GT: {}".format(age)
        txt2 = "Age pred: {}".format(age_pred)
        img_crop = cv2.putText(
            img_crop,
            txt1,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        img_crop = cv2.putText(
            img_crop,
            txt2,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, os.path.basename(row.img_name))
        cv2.imwrite(out_path, img_crop)

    def __getitem__(self, index: int) -> Any:
        self._seed_sample_augmentations(index)
        row = self.df.iloc[index]
        gender_annot = self.gender_label_map[row.gender]
        # if self.age_ranges is not None:
        #     age_num_classes = len(self.age_ranges)
        # else:
        age_num_classes = 120

        try:
            if self.dataset_name == "FairFace":
                age0 = row.age.split(";")
                age = int(np.mean([int(elt) for elt in age0]))
            else:
                age = int(row.age)
            age_annot = np.clip(age, 0, 119)  # Ensure labels are in [0,119]

            # Convert to age range label if requested

            # if self.age_ranges is not None:
            #     age_annot = np.digitize(age_annot, self.age_ranges)

        except Exception as ex:
            logging.warning(f"problem with age annotation for {row.img_name}. \
                            Trace: {ex}")

        img_path = os.path.join(self.img_path, row.img_name)
        img = cv2.imread(img_path)

        # face box coords
        xmin = int(row.face_x0)
        ymin = int(row.face_y0)
        xmax = int(row.face_x1)
        ymax = int(row.face_y1)
        box = [xmin, ymin, xmax, ymax]

        try:
            # data aug on boxes + face crop + data aug on face_crop
            face_crop = img
            for data_aug_pipeline in self.data_aug_pipelines:
                face_crop, _ = data_aug_pipeline(face_crop, box)
            # Convert to gray levels, one channel
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)[..., None]
        except Exception as ex:
            logging.warning(f"failed data_aug {row.img_name}. Trace: {ex}")
            face_crop = np.ones((224, 192, 1))
            gender_annot = 0
            age_annot = 0

        face_crop = face_crop.transpose((2, 0, 1))
        # face_crop = face_crop.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        face_crop = np.ascontiguousarray(face_crop)
        face_crop = face_crop / 255.0  # normalization
        input_tensor = torch.from_numpy(face_crop.astype(np.float32))

        if self.one_hot_gender:
            gender_labels = torch.nn.functional.one_hot(
                torch.tensor(gender_annot), num_classes=2
            ).float()
        else:
            gender_labels = torch.tensor(gender_annot)

        if self.one_hot_age:
            age_labels = torch.nn.functional.one_hot(
                torch.tensor(age_annot), num_classes=age_num_classes
            ).float()
        else:
            if self.center_reduce_age:
                age_annot = (age_annot - 60.0) / 120.0
            age_labels = torch.tensor([age_annot])
            age_labels = torch.tensor([age_annot])
        if self.return_ethnicity:
            if "UTK-Face" in self.dataset_name:
                # [race] is an integer from 0 to 4, denoting White, Black, Asian,
                # Indian, and Others (like Hispanic, Latino, Middle Eastern).
                ethnicity = row.img_name.split("_")[
                    2
                ]
                try:
                    ethnicity = ["White", "Black", "Asian", "Indian", "Others"][
                        int(ethnicity)
                    ]
                except:
                    ethnicity = "Others"

            elif self.dataset_name == "FairFace": 
                # Race class among ["East Asian", "Indian", "Black", "White",
                # "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
                ethnicity = (
                    self.df_ethnicity[self.df_ethnicity.file == row.img_name]
                    .iloc[0]
                    .race
                )
            else:
                ethnicity = "Unknown"
            return input_tensor, gender_labels, age_labels, img_path, box, ethnicity
        else:
            return input_tensor, gender_labels, age_labels, img_path, box

    def __len__(self) -> int:
        return len(self.df)


if __name__ == "__main__":
    from src.utils.config import get_config
    from src import PROJECT_DIR

    config = get_config()
    config_file = PROJECT_DIR / "config/train_config_multi_heads_jd.yaml"
    config.set_config(str(config_file))

    subdataset0 = AgeGenderDataset(
        dataset_name="UTK-Face",
        transforms=config.input.data_augmentations["train"],
        one_hot_gender=True,
        one_hot_age=config.model.heads_params.age.loss == "BCE",
        **config.input.train["UTK-Face"],
    )
    subdataset1 = AgeGenderDataset(
        dataset_name="LAGENDA",
        transforms=config.input.data_augmentations["train"],
        one_hot_gender=True,
        one_hot_age=config.model.heads_params.age.loss == "BCE",
        **config.input.train["LAGENDA"],
    )
    subdataset2 = AgeGenderDataset(
        dataset_name="IMDB",
        transforms=config.input.data_augmentations["train"],
        one_hot_gender=True,
        one_hot_age=config.model.heads_params.age.loss == "BCE",
        **config.input.train["IMDB"],
    )

    # for ii in range(20):
    #     face, gender, age, img_path, _  = subdataset0[ii]
    #     img_name = os.path.basename(img_path)
    #     age, gender = img_name.split('_')[:2]
    #     face = face.cpu().numpy() * 255
    #     out_file = "{0:}_{1:}_face_crop_{2:03d}.png".format(age, gender, ii)
    #     cv2.imwrite(out_file, face[0])
    face, gender, age, img_path, _ = subdataset0[0]
    pdb.set_trace()
