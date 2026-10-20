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

class SmartEyeJitterBox:
    """
    Simulate 'SmartEye' erratic bounding boxes by independently jittering 
    each of the 4 edges (xmin, ymin, xmax, ymax) of the YOLO ground truth.
    This creates asymmetric crops, extreme aspect ratio changes, and heavy truncations.
    
    Args:
        p (float): Probability of applying the jitter.
        edge_noise (float): Maximum noise applied to each edge, as a fraction of the box width/height.
                            e.g., 0.3 means an edge can move inward or outward by up to 30%.
    """
    def __init__(self, p, edge_noise=0.3):
        self.p = p
        self.edge_noise = edge_noise

    def __call__(self, box, img):
        if not isless(self.p):
            return box, img

        img_h, img_w = img.shape[:2]
        xmin, ymin, xmax, ymax = box
        
        box_w = xmax - xmin
        box_h = ymax - ymin

        # Jitter indépendant pour chaque bord (positif = rogne l'image, négatif = ajoute du fond)
        # On biaise légèrement vers l'intérieur (rogne) car SmartEye a tendance à couper trop serré
        dx_min = random.uniform(-self.edge_noise * 0.5, self.edge_noise) * box_w
        dx_max = random.uniform(-self.edge_noise, self.edge_noise * 0.5) * box_w
        dy_min = random.uniform(-self.edge_noise * 0.5, self.edge_noise) * box_h
        dy_max = random.uniform(-self.edge_noise, self.edge_noise * 0.5) * box_h

        # Application du jitter
        new_xmin = xmin + dx_min
        new_xmax = xmax + dx_max
        new_ymin = ymin + dy_min
        new_ymax = ymax + dy_max

        # Sécurité : on s'assure que la boîte fait au moins 10% de sa taille d'origine (pas de boîte inversée ou vide)
        if new_xmax - new_xmin < 0.1 * box_w:
            center_x = (xmin + xmax) / 2
            new_xmin, new_xmax = center_x - 0.05 * box_w, center_x + 0.05 * box_w
            
        if new_ymax - new_ymin < 0.1 * box_h:
            center_y = (ymin + ymax) / 2
            new_ymin, new_ymax = center_y - 0.05 * box_h, center_y + 0.05 * box_h

        # Clip final pour ne pas sortir de l'image
        new_xmin = int(np.clip(new_xmin, 0, img_w - 1))
        new_xmax = int(np.clip(new_xmax, 0, img_w - 1))
        new_ymin = int(np.clip(new_ymin, 0, img_h - 1))
        new_ymax = int(np.clip(new_ymax, 0, img_h - 1))

        return [new_xmin, new_ymin, new_xmax, new_ymax], img