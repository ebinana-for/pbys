import os
from collections import defaultdict

import cv2
import fire
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from src.age_gender.dataset.get_dataloader import get_train_loader
from src.posture.dataset.posture_dataset import PostureDataset
from src.utils.config import get_config

AGE_BINS = [(0, 10), (10, 20), (20, 50), (50, 80), (80, 120)]
SAMPLES_PER_BIN = [2, 2, 2, 2, 12]


def create_images_agegender(config, output_folder, max_iterations):
    train_loader = get_train_loader(config)
    (
        samples,
        bin_counts,
        total_processed,
    ) = sample_with_age_distribution(iter(train_loader), max_iterations=max_iterations)

    # -----------------------------------------------------------
    print(f"Total samples processed: {total_processed}")
    print(f"Num saved samples: {len(samples)}")
    for bin_idx, count in bin_counts.items():
        print(f"Bin {bin_idx} ({AGE_BINS[bin_idx]}): {count} samples")
    # -----------------------------------------------------------

    for i, image in enumerate(samples):
        cv2.imwrite(os.path.join(output_folder, f"agegender_crop_{i}.png"), image)


MAX_IMAGES_PER_CLASS = 5


def create_images_posture(config, output_folder, max_iterations):
    train_dataset = PostureDataset(config, task="TRAIN")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # train_loader = iter(get_train_loader(config))
    counter = [0 for _ in range(train_dataset.classif_out_dim + 1)]
    for ii in range(train_dataset.classif_out_dim + 1):
        os.makedirs(os.path.join(output_folder, str(ii)), exist_ok=True)
    for iteration in tqdm.tqdm(range(max_iterations), desc="Creating posture images"):
        try:
            image_tensor, label, _, _, _ = next(iter(train_loader))
            label = torch.argmax(label, dim=1).item()
        except StopIteration:
            break

        if image_tensor is None or len(image_tensor) == 0:
            continue
        if counter[label] < MAX_IMAGES_PER_CLASS:
            counter[label] += 1
            image = (image_tensor[0][0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(
                    output_folder, str(label), f"posture_crop_{iteration}.png"
                ),
                image,
            )
    print(f"Total images saved for all classes: {counter}")


def sample_with_age_distribution(
    iterator, age_bins=AGE_BINS, samples_per_bin=SAMPLES_PER_BIN, max_iterations=None
):
    """
    Samples from an iterator to achieve a controlled age distribution.

    Args:
        iterator: Iterator that yields tuples of (image, metadata), where metadata
                  contains 'age' and optionally 'gender' fields.
        age_bins: List of tuples representing age bin boundaries, e.g., [(0, 10), [10, 20), ...]
        samples_per_bin: Number of samples required for each age bin
        max_iterations: Optional safety limit on total samples to process

    Returns:
        A tuple with:
            - 'samples': List of selected samples
            - 'bin_counts': Final count of samples in each bin
            - 'total_processed': Total number of samples processed
    """
    # Initialize
    samples = []
    bin_counts = defaultdict(int)
    bin_ranges = [range(bin[0], bin[1]) for bin in age_bins]
    total_processed = 0
    targets_met = False

    # Convert samples_per_bin to dict if it's not already
    if isinstance(samples_per_bin, (list, tuple, np.ndarray)):
        samples_per_bin = {i: count for i, count in enumerate(samples_per_bin)}

    # Process iterator until all bins are filled or max_iterations reached
    while not targets_met and (
        max_iterations is None or total_processed < max_iterations
    ):
        try:
            image_tensors, _, age_labels, _, _ = next(iterator)
        except StopIteration:
            break

        total_processed += 1
        age = age_labels

        # Find which bin this age falls into
        for bin_idx, bin_range in enumerate(bin_ranges):
            if age in bin_range:
                if bin_counts[bin_idx] < samples_per_bin[bin_idx]:
                    samples.append(
                        (image_tensors[0][0].cpu().numpy() * 255).astype(np.uint8)
                    )
                    bin_counts[bin_idx] += 1
                break

        # Check if all bins have enough samples
        targets_met = all(
            bin_counts.get(bin_idx, 0) >= count
            for bin_idx, count in samples_per_bin.items()
        )

    return (
        samples,
        dict(bin_counts),
        total_processed,
    )


def main(config_path, output_folder, task, max_iterations=1000):
    """Read the train dataset, load training images, create model input images
    and save them for further recalibration  (post-train quantization) of the model.
    Args:
        config_path (str): Path to the configuration file. It should contain the description of the train dataset.
        output_folder (str): Path to the folder where the images will be saved.
        task (str): Task type, either "posture" or "agegender".
        max_iterations (int): Maximum number of data loading iterations to process. Default is 1000.
    """
    assert os.path.exists(config_path), f"Config file {config_path} does not exist"

    # Get config
    config = get_config()
    config.set_config(str(config_path))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Output images are save to: {output_folder}")

    if task == "posture":
        create_images_posture(config, output_folder, max_iterations)
    elif task == "agegender":
        create_images_agegender(config, output_folder, max_iterations)
    else:
        raise ValueError(f"Unknown task: {task}. Use 'posture' or 'agegender'.")


if __name__ == "__main__":
    fire.Fire(main)
