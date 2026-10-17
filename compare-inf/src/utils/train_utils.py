#!/usr/bin/env python3
"""Utility functions for training in PyTorch.
This module provides functions for averaging losses, loading model checkpoints,
and checking probabilities."""
from __future__ import annotations
import numpy as np
import torch
import logging


def isless(prob: float = 0.5) -> bool:
    return np.random.uniform(0, 1) < prob


class Averager:
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v: torch.Tensor):
        count = v.data.numel()
        v = v.data.sum()  # .item()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0.0

    def val(self) -> float:
        res = 0.0
        if self.n_count:
            res = self.sum / float(self.n_count)
        return res


def load_ckpt(
    model: torch.nn.Module, ckpt: dict[str, torch.Tensor], strict: bool
) -> torch.nn.Module:
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logging.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logging.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=strict)
    return model


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return float(param_group["lr"])