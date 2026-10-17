"""Age/gender inference engine using OpenCV and NumPy for preprocessing and postprocessing."""

import cv2
import logging

logging.basicConfig(level=logging.INFO)
import argparse
import numpy as np
from src.utils.inference_engine import InferenceEngine


class InferenceEngineAgeGender(InferenceEngine):
    """Class handling image pre and post processing."""

    def __init__(
        self,
        model_path: str,
    ):
        """Initialize the inference engine with model path."""
        super().__init__(model_path)

    def preprocess(self, img_list):
        input_tensor = []
        for img in img_list:
            img = cv2.resize(img, (self.width, self.height))
            # R channel only
            if len(img.shape) > 2:
                img[:, :, 0] = img[:, :, 2]
                img[:, :, 1] = img[:, :, 2]
            if self.channels == 1:
                img = img[:, :, 0][..., None]
            x = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
            input_tensor.append(x)
        input_tensor = np.concatenate(input_tensor, axis=0, out=None)
        return input_tensor

    def postprocess(self, outputs):
        out_list = []
        
        outputs = np.array(outputs)


        for gender_array, age_array in zip(outputs[0, :, :2], [outputs[0, :, 2]]):         
            e_x = np.exp(gender_array - np.max(gender_array))
            e_x /= e_x.sum()
            gender = ["M", "F"][np.argmax(gender_array)]
            gender_conf = float(e_x[np.argmax(gender_array)])
            if len(age_array) == 1:  # regression model
                age_mean = float(age_array[0])
                age_std = 0
            else:  # classification model
                e_x = np.exp(age_array - np.max(age_array))
                e_x /= e_x.sum()
                age_mean = float(np.sum(e_x * np.arange(0, len(e_x))))
                age_var = np.sum((e_x * (np.arange(0, len(e_x)) - age_mean) ** 2))
                age_std = float(np.sqrt(age_var))

            out_list.append((gender, gender_conf, age_mean, age_std))
        return out_list


def main():
    """"""
    parser = argparse.ArgumentParser("Age and gender inference")
    parser.add_argument("--model_path", required=True, help="Path to input model")
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Path to input model"
    )

    args = parser.parse_args()
    age_gender_inference = InferenceEngineAgeGender(args.model_path)
    img = cv2.imread("/workspace/ressources/nir1.png")
    preds = age_gender_inference.predict([img] * args.batch_size)
    print(preds)


if __name__ == "__main__":
    main()
