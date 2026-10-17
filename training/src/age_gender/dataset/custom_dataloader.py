"""Provide customized Pytorch dataloaders for age/gender training and evaluation on multple datasets.
"""
from __future__ import annotations
import logging
from pprint import pformat
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    SequentialSampler,
    WeightedRandomSampler,
)


def uniform_weights(datasets: dict[str, Dataset]) -> list[float]:
    global_len = 0
    weights = []
    for d in datasets.values():
        global_len += len(d)
    for k, d in datasets.items():
        tmp_weight = float(global_len) / max(len(d), 1)
        logging.info(f"Applying weight for dataset {k}: {tmp_weight}")
        weights += [tmp_weight] * len(d)
    return weights


def dataset_weights(
    datasets: dict[str, Dataset], probability: dict[str, float]
) -> list[float]:
    """_summary_

    Args:
        datasets (dict[str, Dataset]): _description_
        weights (dict[str, float]): _description_

    Returns:
        list[float]: _description_
    """
    # Normalizing weights -> sum to one
    sum_proba = sum([v for k, v in probability.items()])
    normalized_proabilities = {k: v / sum_proba for k, v in probability.items()}
    global_len = sum([len(datasets[key]) for key in datasets])
    weight_dataset: list[float] = []
    for k, d in datasets.items():
        tmp_weight = (
            float(normalized_proabilities[k]) * float(global_len) / max(len(d), 1)
        )
        logging.info(f"Applying weight for dataset {k}: {tmp_weight}")
        weight_dataset += [tmp_weight] * len(d)
    return weight_dataset


def image_weights(
    datasets: dict[str, Dataset], weights: dict[str, float]
) -> list[float]:
    """_summary_

    Args:
        datasets (dict[str, Dataset]): _description_
        weights (dict[str, float]): _description_

    Raises:
        NotImplementedError: _description_
        ValueError: _description_

    Returns:
        list[float]: _description_
    """
    weight_dataset: list[float] = []
    for type_dataset in datasets:
        logging.info(
            f"Applying weight for dataset {type_dataset}: {weights[type_dataset]}"
        )
        weight_dataset += [weights[type_dataset]] * len(datasets[type_dataset])
    return weight_dataset


class CustomDataloader(DataLoader):
    """
    Custom Dataloader for handling multiple datasets with different sampling strategies.
    This class allows for uniform, dataset-specific, or image-specific weights for sampling.
    It supports both training and testing splits with appropriate samplers.
    """
    _log: logging.Logger

    def __init__(
        self,
        datasets: dict[str, Dataset],
        weights: dict[str, float] = {},
        split: str = "train",
        weight_dataset_policy: str = "uniform_weight",
        **kwargs,
    ) -> None:
        """
        Initialize the CustomDataloader with multiple datasets and a specified split.
        Args:
            datasets (dict[str, Dataset]): Dictionary of datasets to be fused.
            weights (dict[str, float]): Weights for each dataset, used for sampling.
            split (str): The split of the dataset, either 'train' or 'test'.
            weight_dataset_policy (str): Policy for applying weights during sampling.
                Options are 'uniform_weight', 'dataset_weight', or 'image_weight'.
        Raises:
            NotImplementedError: If an unsupported weight policy is provided.
            ValueError: If an unsupported split is provided.
        """

        logging.info("Initializing CustomDataloader")
        logging.info(f"Datasets: {pformat(list(datasets.keys()))}")

        assert split.lower() in ["train", "test"], f"Unsupported split: {split}"

        # Building dataset
        self.split = split
        self.datasets: dict[str, Dataset] = datasets
        logging.info(f"Building fused datasets: {pformat(list(self.datasets.keys()))}")
        self.n_samples = 0
        for dataset_name, dataset in self.datasets.items():
            self.n_samples += len(dataset)
        global_dataset: Dataset = ConcatDataset(datasets.values())

        sampler_generator = kwargs.pop("sampler_generator", kwargs.get("generator"))

        # Test split
        if split == "test":
            logging.info(
                f"Using Sequential sampler for datasets: \
                    {pformat(list(self.datasets.keys()))}"
            )
            sampler = SequentialSampler(global_dataset)

        # Train split
        elif split == "train":
            logging.info(
                f"Applying '{weight_dataset_policy}' \
                    weights policy for sampling '{split} set'"
            )
            # Weight policy
            if weight_dataset_policy == "uniform_weight":
                weight_dataset = uniform_weights(datasets)
            elif weight_dataset_policy == "dataset_weight":
                weight_dataset = dataset_weights(datasets, weights)
            elif weight_dataset_policy == "image_weight":
                weight_dataset = image_weights(datasets, weights)
            else:
                raise NotImplementedError(
                    f"Does not support other type of weights policy than \
                        'uniform_weight', 'dataset_weight', 'image_weight'. \
                            You passed {weight_dataset_policy}."
                )
            
            assert weight_dataset is not None
            sampler = WeightedRandomSampler(
                weight_dataset, len(global_dataset), generator=sampler_generator
            )

        else:
            raise ValueError(f"Not supported {split} for get sampler")

        super().__init__(
            dataset=global_dataset,
            sampler=sampler,
            **kwargs,
        )
        logging.info(f"{self.display()}")

    def display(
        self,
    ) -> str:
        output = f"'{self.split}' dataset summary: batch_size = {self.batch_size} \
            n_iter = {len(self)} n_samples = {self.n_samples}\n"
        output += "Subdatasets summaries:\n"
        for dataset_name, dataset in self.datasets.items():
            output += f"\t'{self.split}' {dataset_name}: n_samples = {len(dataset)}\n"
        return output


class DataModule:
    """
    DataModule for handling multiple datasets with their respective dataloaders.
    This class aggregates multiple CustomDataloaders for different datasets,
    allowing for easy access and management of training and testing splits.
    """
    _log: logging.Logger

    def __init__(
        self,
        datasets: dict[str, Dataset],
        split: str = "train",
        **kwargs,
    ) -> None:
        """
        Initialize the DataModule with multiple datasets and a specified split.
        Args:
            datasets (dict[str, Dataset]): Dictionary of datasets to be managed.
            split (str): The split of the dataset, either 'train' or 'test'.
            **kwargs: Additional keyword arguments for CustomDataloader.
        Raises:
            ValueError: If an unsupported split is provided.
        """

        self.split = split
        self.dataloaders: dict = {}
        self.n_samples = 0
        for dataset_name, dataset in datasets.items():
            self.dataloaders[dataset_name] = CustomDataloader(
                {dataset_name: dataset}, split=split, **kwargs
            )
            self.n_samples += self.dataloaders[dataset_name].n_samples

        logging.info(self)

    def __str__(
        self,
    ) -> str:
        total_samples = 0
        total_iter = 0
        detailed_output = "Dataloaders summaries:\n"
        for data_loader_name, value in self.dataloaders.items():
            total_samples += value.n_samples
            total_iter += len(value)
            detailed_output += f"\t'{value.split}' {data_loader_name}: batch_size = \
                    {value.batch_size} n_iter = {len(value)} \
                    n_samples = {value.n_samples}\n"
        output = f"'{self.split}' data module summary: '{self.split}': batch_size = \
            {value.batch_size} n_iter = {total_iter} n_samples = {total_samples}\n"
        output += detailed_output
        return output

    def get_datasets_name(self) -> list[str]:
        return list(self.dataloaders.keys())
