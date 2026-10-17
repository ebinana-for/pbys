"""Box augmentation utilities"""
from src.utils.train_utils import isless
import random
import numpy as np
import cv2

class ScaleBox:
    """
    Scale the bounding box by a random factor within a specified range.
    The scaling is applied uniformly to both width and height of the box.
    Args:
        p (float): Probability of applying the scaling.
        scale_min (float): Minimum scaling factor.
        scale_max (float): Maximum scaling factor.
    """
    def __init__(self, p, scale_min, scale_max):
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max
        return

    def __call__(self, box, img):
        img_h, img_w = img.shape[:2]
        xmin, ymin, xmax, ymax = box
        if isless(self.p):
            box_w = xmax - xmin
            box_h = ymax - ymin
            x_shift = (random.uniform(self.scale_min, self.scale_max) * box_w) / 2
            y_shift = (random.uniform(self.scale_min, self.scale_max) * box_h) / 2
            xmin = int(np.clip(xmin - x_shift, 0, img_w))
            xmax = int(np.clip(xmax + x_shift, 0, img_w))
            ymin = int(np.clip(ymin - y_shift, 0, img_h))
            ymax = int(np.clip(ymax + y_shift, 0, img_h))
        return [xmin, ymin, xmax, ymax], img


class ShiftBox:
    """
    Shift the bounding box by a random amount within a specified range.
    The shift is applied uniformly to both width and height of the box.
    Args:
        p (float): Probability of applying the shift.
        shift (float): Maximum shift as a fraction of the box width/height.
    """
    def __init__(
        self,
        p,
        shift,
    ):
        self.p = p
        self.shift = shift
        return

    def __call__(self, box, img):
        img_h, img_w = img.shape[:2]
        xmin, ymin, xmax, ymax = box
        if isless(self.p):
            box_w = xmax - xmin
            box_h = ymax - ymin
            x_shift = random.uniform(-self.shift, self.shift) * box_w
            y_shift = random.uniform(-self.shift, self.shift) * box_h
            xmin = int(np.clip(xmin + x_shift, 0, img_w))
            xmax = int(np.clip(xmax + x_shift, 0, img_w))
            ymin = int(np.clip(ymin + y_shift, 0, img_h))
            ymax = int(np.clip(ymax + y_shift, 0, img_h))
        return [xmin, ymin, xmax, ymax], img


class RotateBox:
    """
    Rotate the bounding box by a random angle within a specified range.
    Args:
        min_angle (float): Minimum rotation angle in degrees.
        max_angle (float): Maximum rotation angle in degrees.
        p (float): Probability of applying the rotation.
    """
    def __init__(self, min_angle, max_angle, p):
        self.p = p
        self.min_angle = min_angle
        self.max_angle = max_angle
        return

    def __call__(self, box, img):
        xmin, ymin, xmax, ymax = box
        if isless(self.p):
            rot_angle = random.uniform(self.min_angle, self.max_angle)
            img_h, img_w = img.shape[:2]
            cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
            Mrot = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
            img = cv2.warpAffine(img, Mrot, (img_w, img_h))
        return [xmin, ymin, xmax, ymax], img
