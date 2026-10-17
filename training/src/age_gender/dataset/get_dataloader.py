"""Dataloader utility function for age/gender training and evaluation"""

import logging
from src.age_gender.dataset.dataset import AgeGenderDataset
from src.age_gender.dataset.custom_dataloader import CustomDataloader, DataModule
import ast

import torch
import random
import numpy as np

from torch.utils.data import ConcatDataset, Subset, get_worker_info

def _seed_augmentations(dataset, seed, visited=None):
    if visited is None:
        visited = set()
    if id(dataset) in visited:
        return 0
    visited.add(id(dataset))
    count = 0
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            count += _seed_augmentations(child, seed, visited)
    elif isinstance(dataset, Subset):
        count += _seed_augmentations(dataset.dataset, seed, visited)
    for pipeline in getattr(dataset, "data_aug_pipelines", []):
        compose = getattr(pipeline, "albumentation_pipeline", None)
        if compose is not None:
            compose.set_random_seed(int(seed))
            count += 1
    return count

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    n = 0
    info = get_worker_info()
    if info is not None:
        n = _seed_augmentations(info.dataset, worker_seed)
    # print(f"[worker {worker_id}] seed={worker_seed} pipelines_seeded={n} "
    #       f"dataset_type={type(info.dataset).__name__}", flush=True)


def get_train_loader(config) -> CustomDataloader:
    """
    Return the custom train dataloader
    Args:
        config: Configuration object containing model and training parameters.
    Returns:
        CustomDataloader: A dataloader for training datasets.
    This function initializes the AgeGenderDataset for each training dataset specified in the configuration
    """
    logging.info("Configuring 'Train set'")
    train_datasets: dict[str, AgeGenderDataset] = {}
    weights = dict()

    one_hot_gender = config
    one_hot_age = False
    if "center_reduce" in config.model.heads_params.age:
        center_reduce_age = config.model.heads_params.age.center_reduce
    else:
        center_reduce_age = False
    seed_value = getattr(config.general, "seed", 42) if hasattr(config, "general") else 42
    for key, val in config.model.heads_params.items():
        if key == "age":
            one_hot_age = True if "BCE" in val.loss else False
            center_reduce_age = True if "MSE" in val.loss else False
            if "age_ranges" in val and "BCE" in val.loss and val.out_dim != 120:
                age_ranges = [elt for elt in ast.literal_eval(val.age_ranges)]
                age_ranges = sorted(age_ranges)
            else:
                age_ranges = None
    for name_of_dataset, dataset_params in config.input.train.items():
        subdataset = AgeGenderDataset(
            dataset_name=name_of_dataset,
            transforms=config.input.data_augmentations["train"],
            one_hot_gender=one_hot_gender,
            one_hot_age=one_hot_age,
            center_reduce_age=center_reduce_age,
            age_ranges=age_ranges,
            seed=seed_value,
            **dataset_params
        )
        _seed_augmentations(subdataset, seed_value)
        train_datasets[name_of_dataset] = subdataset
        weights[name_of_dataset] = dataset_params.get("weight", 1)

    sampler_generator = torch.Generator().manual_seed(seed_value)
    worker_generator = torch.Generator().manual_seed(seed_value + 1)

    train_dataloader = CustomDataloader(
        train_datasets,
        weights,
        split="train",
        weight_dataset_policy=config.input.weight_datasets_policy,
        batch_size=config.input.batch_size,
        num_workers=config.input.num_workers,
        worker_init_fn=seed_worker,
        generator=worker_generator,
        sampler_generator=sampler_generator,
    )

    return train_dataloader


def get_eval_loader(config) -> DataModule:
    """
    Return the custom evaluation dataloader
    Args:
        config: Configuration object containing model and evaluation parameters.
    Returns:
        DataModule: A dataloader for evaluation datasets.
    This function initializes the AgeGenderDataset for each evaluation dataset specified in the configuration.
    """
    logging.info("Configuring 'Test set'")
    test_datasets: dict[str, AgeGenderDataset] = {}

    one_hot_gender = True
    one_hot_age = False
    if "center_reduce" in config.model.heads_params.age:
        center_reduce_age = config.model.heads_params.age.center_reduce
    else:
        center_reduce_age = False
    seed_value = getattr(config.general, "seed", 42) if hasattr(config, "general") else 42
    for key, val in config.model.heads_params.items():
        if key == "age":
            one_hot_age = True if "BCE" in val.loss else False
            center_reduce_age = True if "MSE" in val.loss else False
            if "age_ranges" in val and "BCE" in val.loss and val.out_dim != 120:
                age_ranges = [elt for elt in ast.literal_eval(val.age_ranges)]
                age_ranges = sorted(age_ranges)
            else:
                age_ranges = None

    for name_of_dataset, dataset_params in config.input.test.items():
        subdataset = AgeGenderDataset(
            dataset_name=name_of_dataset,
            transforms=config.input.data_augmentations["test"],
            one_hot_gender=one_hot_gender,
            one_hot_age=one_hot_age,
            center_reduce_age=center_reduce_age,
            age_ranges=age_ranges,
            seed=seed_value,
            **dataset_params
        )
        _seed_augmentations(subdataset, seed_value)
        test_datasets[name_of_dataset] = subdataset

    g = torch.Generator().manual_seed(seed_value + 1)

    test_dataloader = DataModule(
        test_datasets,
        split="test",
        batch_size=config.input.batch_size,
        num_workers=config.input.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return test_dataloader
