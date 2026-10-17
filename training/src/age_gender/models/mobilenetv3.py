#!/usr/bin/env python3
"""MobilenetV3 implementation for age/gender estimation with multiple heads."""
from typing import Callable, Optional
import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf

# Valid heads for the model
valid_heads = ["gender", "age"]


class MobileNetV3MultiHeads(MobileNetV3):

    def __init__(
        self,
        heads_list,
        in_chans=3,
        width_mult: float = 1.0,
        arch="mobilenet_v3_small",
        reduce_tail=True,
        dilated=False,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:

        setting_args = {
            "_width_mult": width_mult,
            "_reduced_tail": reduce_tail,
            "_dilated": dilated,
        }

        inverted_residual_setting, last_channel = _mobilenet_v3_conf(
            arch, **setting_args
        )
        super().__init__(
            inverted_residual_setting,
            last_channel,
            num_classes=2,  # not used
            block=block,
            norm_layer=norm_layer,
        )

        # Change the number of channels in the first layer, if needed
        if in_chans != 3:
            from torchvision.ops.misc import Conv2dNormActivation

            firstconv_output_channels = inverted_residual_setting[0].input_channels
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            keep_layers = list(self.features.children())[1:]
            first_layer = Conv2dNormActivation(
                in_chans,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
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
                nn.Linear(6 * inverted_residual_setting[-1].out_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(last_channel, elt["out_dim"]),
            )

            self.heads.append(head)

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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
    net = MobileNetV3MultiHeads(heads_list=heads_list)
    torchinfo.summary(net)


if __name__ == "__main__":
    main()
