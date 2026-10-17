#!/usr/bin/env python3
"""MobilenetV2 implementation for age/gender estimation with multiple heads."""

from typing import Callable, List, Optional
import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNetV2

# Valid heads for the model
valid_heads = ["gender", "age"]


class MobileNetV2MultiHeads(MobileNetV2):

    def __init__(
        self,
        heads_list,
        in_chans=3,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:

        super().__init__(
            num_classes=2,  # not used
            width_mult=width_mult,
            inverted_residual_setting=inverted_residual_setting,
            round_nearest=round_nearest,
            block=block,
            norm_layer=norm_layer,
            dropout=dropout,
        )

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

        self.dropout = dropout
        self.heads_list = heads_list
        del self.classifier

        # Create heads
        self.heads = nn.ModuleList()
        for elt in self.heads_list:
            assert (
                elt["name"] in valid_heads
            ), "Invalid head name. Valid names are {}".format(valid_heads)
            assert (
                "out_dim" in elt.keys()
            ), "You must specify out_dim for head {}".format(elt["name"])
            assert "loss" in elt.keys(), "You must specify loss for head {}".format(
                elt["loss"]
            )
            head = nn.Sequential(
                nn.Dropout(p=self.dropout), nn.Linear(self.last_channel, elt["out_dim"])
            )

            self.heads.append(head)

    def _forward_impl(self, x: torch.Tensor):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        out = []
        for head in self.heads:
            x_out = head(x)
            out.append(x_out)

        return out


def main():
    import torchinfo

    heads_list = [
        {"name": "gender", "loss": "BCE", "out_dim": 2},
        {"name": "age", "loss": "MSE", "out_dim": 1},
    ]
    net = MobileNetV2MultiHeads(heads_list=heads_list)
    torchinfo.summary(net)


if __name__ == "__main__":
    main()
