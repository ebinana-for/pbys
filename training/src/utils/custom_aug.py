"""Custom augmentations for image and keypoints processing."""
from src.utils.train_utils import isless
import cv2


class ExtractRChannel:
    """
    Extract the R channel from the image and set G and B channels to R.
    This is used for public datasets where only the R channel is used.
    """
    def __init__(self, p):
        self.p = p
        return

    def __call__(self, image, keypoints=None):
        if isless(self.p):
            # R channel only todo put it in custom data aug
            image[:, :, 0] = image[:, :, 2]
            image[:, :, 1] = image[:, :, 2]
        return image, keypoints


class ImgKptsHorizontalFlip:
    """Flip the image and keypoints horizontally."""
    def __init__(self, p):
        self.p = p
        return

    def __call__(self, image, keypoints=None):
        keypoints_flip = keypoints.copy()
        if isless(self.p):
            image_flip = cv2.flip(image, 1)

            for key in keypoints.keys():
                if "right" in key:
                    flip_key = key.replace("right", "left")
                elif "left" in key:
                    flip_key = key.replace("left", "right")
                else:
                    flip_key = key
                if "/x" in key:
                    keypoints_flip[flip_key] = image_flip.shape[1] - keypoints[key]
                else:
                    keypoints_flip[flip_key] = keypoints[key]
        else:
            image_flip = image

        return image_flip, keypoints_flip


class ImgKptsNormalize:
    """Normalize keypoints coordinates by image dimensions."""
    def __init__(self, p=1):
        self.p = p
        return

    def __call__(self, image, keypoints):
        keypoints_norm = keypoints.copy()
        norm_y, norm_x = image.shape
        for key in keypoints_norm.keys():
            if "/x" in key:
                keypoints_norm[key] /= norm_x
            if "/y" in key:
                keypoints_norm[key] /= norm_y

        return image, keypoints_norm



class ImgHorizontalFlip:
    """Flip the image horizontally."""
    def __init__(self, p):
        self.p = p
        return

    def __call__(self, image, keypoints=None):
        from src.utils.train_utils import isless
        import cv2

        if isless(self.p):
            image = cv2.flip(image, 1)
        return image, keypoints

class ImgColorJitter:
    """Color Jitter implementation using OpenCV and standard random """
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.7, hue=0.5, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image, keypoints=None):
        from src.utils.train_utils import isless
        import cv2
        import numpy as np
        import random

        if not isless(self.p):
            return image, keypoints

        # Brightness & Contrast
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = random.uniform(-self.brightness, self.brightness) * 255
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Saturation & Hue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        sat_mult = 1.0 + random.uniform(-self.saturation, self.saturation)
        hsv[:, :, 1] *= sat_mult
        
        hue_shift = random.uniform(-self.hue, self.hue) * 180
        hsv[:, :, 0] += hue_shift
        
        # Range clipping
        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image, keypoints


class ImgOpticalDistortion:
    """Optical Distortion using OpenCV remap"""
    def __init__(self, distort_limit=0.3, p=0.5):
        self.distort_limit = distort_limit
        self.p = p

    def __call__(self, image, keypoints=None):
        from src.utils.train_utils import isless
        import cv2
        import numpy as np
        import random

        if not isless(self.p):
            return image, keypoints

        h, w = image.shape[:2]
        k = random.uniform(-self.distort_limit, self.distort_limit)
        
        # Camera matrix
        fx, fy = w, h
        cx, cy = w * 0.5, h * 0.5
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        # Distortion coeffs
        dist_coeffs = np.array([k, 0, 0, 0, 0], dtype=np.float32)
        
        # Remap
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1)
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        return image, keypoints