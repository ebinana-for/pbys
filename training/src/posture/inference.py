"""Posture inference engine using OpenCV and NumPy for preprocessing and postprocessing."""
import cv2
import logging
import argparse
import numpy as np
from src.utils.inference_engine import InferenceEngine

logging.basicConfig(level=logging.INFO)


class InferenceEnginePosture(InferenceEngine):
    """Class handling image pre and post processing for posture estimation."""
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
        for logits in outputs[0]:
            softmax = np.exp(logits) / sum(np.exp(logits))
            print(softmax)
            out_list.append(softmax)
        return out_list


def main():
    parser = argparse.ArgumentParser("Age and gender inference")
    parser.add_argument("--model_path", required=True, help="Path to input model")
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Path to input model"
    )

    args = parser.parse_args()
    age_gender_inference = InferenceEnginePosture(args.model_path)
    img = cv2.imread("/mnt/data/pbys/training/ressources/nir1.png")
    preds = age_gender_inference.predict([img] * args.batch_size)
    print(preds)


if __name__ == "__main__":
    main()
