#!/usr/bin/env python3
"""MobileNetV2 based posture estimation models with different heads.

- MobileNetV2Posture: Standard MobileNetV2 for posture estimation.

"""
from typing import Callable, List, Optional
import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNetV2
from copy import deepcopy


class MobileNetV2Posture(MobileNetV2):
    """ MobileNetV2 model for posture estimation with a single head."""
    def __init__(
        self,
        head_param,
        in_chans=3,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the MobileNetV2Posture model.
        Args:
            head_param: dict with the following structure
            {"name": num_classes (int)}
            in_chans (int): Number of input channels (default: 3).
            width_mult (float): Width multiplier for the model (default: 1.0).
            inverted_residual_setting (List[List[int]]): Configuration for inverted residual blocks.
            round_nearest (int): Round nearest value for channel width (default: 8).
            block (Callable[..., nn.Module]): Block type for the model (default: None).
            norm_layer (Callable[..., nn.Module]): Normalization layer type (default: None).
            dropout (float): Dropout rate for the model (default: 0.2).
        Returns:
            None
        """
        super().__init__(
            num_classes=2,  # not used
            width_mult=width_mult,
            inverted_residual_setting=inverted_residual_setting,
            round_nearest=round_nearest,
            block=block,
            norm_layer=norm_layer,
            dropout=dropout,
        )
        self.dropout = dropout
        self.n_inputs = 1  # 2 input in forward method

        # Change the number of channels in the first layer, if needed
        if in_chans != 3:
            from torchvision.models._utils import _make_divisible
            from torchvision.ops.misc import Conv2dNormActivation

            input_channel = _make_divisible(32 * width_mult, round_nearest)
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            keep_layers = list(self.features.children())[1:]
            first_layer = Conv2dNormActivation(
                in_chans,
                input_channel,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
            self.features = torch.nn.Sequential(first_layer, *keep_layers)

        self.head_param = head_param
        # key, param  = next(iter(head_param[0].items()))

        del self.classifier
        head = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.last_channel, head_param[0]["out_dim"]),
        )
        self.head = head

    def _forward_impl(self, x: torch.Tensor):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x_out = self.head(x)
        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

