"""Custom augmentations for image and keypoints processing."""
from .train_utils import isless
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
