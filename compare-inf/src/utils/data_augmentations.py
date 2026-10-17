#!/usr/bin/env python3
"""Data augmentation pipeline for object"""
import logging
import cv2
import numpy as np
import random
import albumentations as A
from . import box_aug
from . import custom_aug
import ast


class DataAugPipeline:
    """Data augmentation pipeline for object detection tasks.
    This class combines box augmentation, albumentation transformations,
    and custom transformations into a single pipeline.
    Args:
        name (str): Name of the augmentation pipeline.
        box_ops (dict): Dictionary of box augmentation operations and their parameters.
        albumentation_ops (dict): Dictionary of albumentation operations and their parameters.
        custom_ops (dict): Dictionary of custom augmentation operations and their parameters.
        p (float): Probability of applying the augmentation.
    """

    def __init__(self, name, box_ops, albumentation_ops, custom_ops, p, seed=42):
        
        self.name = name
        self.box_ops = box_ops
        self.albumentation_ops = albumentation_ops
        self.custom_ops = custom_ops
        self.p = p

        # box part
        self.box_pipeline = []
        if self.box_ops:
            for box_op, params in self.box_ops.items():
                self.box_pipeline.append(box_aug.__dict__[box_op](**params))

        # Albumentation part
        self.albumentation_pipeline = None
        list_data_aug = []
        for albumentation_op, params in self.albumentation_ops.items():
            if albumentation_op in A.__dict__.keys():
                params2 = params.copy()
                for key in params2.keys():
                    if type(params2[key]) == str:
                        params2[key] = ast.literal_eval(params2[key])
                list_data_aug.append(A.__dict__[albumentation_op](**params2))

        if len(list_data_aug) > 0:
            self.albumentation_pipeline = A.Compose(list_data_aug, p=self.p, seed=seed)

        # custom part
        self.custom_pipeline = []
        if self.custom_ops:
            for custom_op, params in self.custom_ops.items():
                self.custom_pipeline.append(custom_aug.__dict__[custom_op](**params))
        logging.debug(f"Data augmentation params: {self.__dict__}")

    def resize_kpts(self, keypoints, init_shape, resize_shape, mode="simple"):
        x_factor = resize_shape[1] / init_shape[1]
        y_factor = resize_shape[0] / init_shape[0]
        if mode == "simple":  # simple resize, without preserving aspect ratio
            x_scale = x_factor
            x_bias = 0
            y_scale = y_factor
            y_bias = 0
        elif mode == "pad":  # resize with horizontal or vertical padding
            if x_factor < y_factor:  # y padding
                x_scale = x_factor
                x_bias = 0
                y_scale = x_factor
                y_bias = (resize_shape[0] - init_shape[0] * x_factor) / 2
            else:  # x padding
                x_scale = y_factor
                x_bias = (resize_shape[1] - init_shape[1] * y_factor) / 2
                y_scale = y_factor
                y_bias = 0
        else:
            raise (NotImplementedError)

        new_kpts = keypoints.copy()
        for key in new_kpts.keys():
            if "/x" in key:
                new_kpts[key] = (
                    int(new_kpts[key] * x_scale + x_bias + 0.5)
                    if new_kpts[key] > 1.0
                    else 0
                )
            if "/y" in key:
                new_kpts[key] = (
                    int(new_kpts[key] * y_scale + y_bias + 0.5)
                    if new_kpts[key] > 1.0
                    else 0
                )

        return new_kpts

    def __call__(self, im: np.ndarray, box, keypoints=None):
        if random.uniform(0, 1) > self.p:
            return im

        # perfom box aug
        for box_op in self.box_pipeline:
            box, im = box_op(box, im)

        # perfom img crop
        im = im[box[1] : box[3], box[0] : box[2]]
        if keypoints is not None:
            for key in keypoints.keys():
                if "/x" in key:
                    keypoints[key] -= box[0]
                    keypoints[key] = np.clip(keypoints[key], 0, im.shape[1])
                if "/y" in key:
                    keypoints[key] -= box[1]
                    keypoints[key] = np.clip(keypoints[key], 0, im.shape[0])

        # albumentations is using rgb colorspace
        if len(im.shape) > 2:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        transformed = self.albumentation_pipeline(image=im)

        # Update kpts
        if keypoints is not None:
            mode = "pad" if "PadIfNeeded" in self.albumentation_ops.keys() else "simple"
            keypoints = self.resize_kpts(
                keypoints, im.shape, transformed["image"].shape, mode=mode
            )

        im = transformed["image"]

        if len(im.shape) > 2:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        for custom_op in self.custom_pipeline:
            im, keypoints = custom_op(image=im, keypoints=keypoints)

        return im, keypoints
