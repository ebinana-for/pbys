"""Custom data augmentation functions for age/gender datasets"""

from src.utils.train_utils import isless


class ExtractRChannel:
    """
    Extract the R channel from the image and set G and B channels to R.
    This is used for public datasets where only the R channel is used.
    """
    def __init__(self, p):
        self.p = p
        return

    def __call__(self, image):
        if isless(self.p):
            # R channel only todo put it in custom data aug
            image[:, :, 0] = image[:, :, 2]
            image[:, :, 1] = image[:, :, 2]
        return image
