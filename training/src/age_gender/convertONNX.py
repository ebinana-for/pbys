"""Pytorch to onnx conversion script for the age/gender model"""

import torch
import logging
import argparse

from src.utils.config import get_config
from src.age_gender.models.mobilenetv3 import MobileNetV3MultiHeads
from src.age_gender.models.mobilenetv2 import MobileNetV2MultiHeads
from src.age_gender.models.edgevit.edgevit import (
    edgevit_xxs_multi_heads,
    edgevit_xs_multi_heads,
    edgevit_s_multi_heads
)


class ModelOneOutput(torch.nn.Module):
    """Model with one output, used to fuse the two heads of the model."""

    def __init__(self, model, model_name):
        """
        Initialize the model with one output.
        Args:
            model (torch.nn.Module): The original model with two heads.
            model_name (str): The name of the model, used to determine the architecture.
        """
        super().__init__()
        self.model = model.module
        self.model_name = model_name
        # pdb.set_trace()
        if "mobilenet" in self.model_name.lower():
            channels1 = self.model.heads[0][1].bias.data.shape[0]
            channels2 = self.model.heads[1][1].bias.data.shape[0]
            in_channels = self.model.heads[0][1].weight.data.shape[1]
        elif "edgevit" in self.model_name.lower():
            channels1 = self.model.heads[0].bias.data.shape[0]
            channels2 = self.model.heads[1].bias.data.shape[0]
            in_channels = self.model.heads[0].weight.data.shape[1]
        else:
            raise (NotImplementedError)

        self.linear_out = torch.nn.Linear(in_channels, channels1 + channels2)

        if "mobilenet" in self.model_name.lower():
            weight_c = torch.cat(
                (
                    self.model.heads[0][1].weight.data,
                    self.model.heads[1][1].weight.data,
                ),
                dim=0,
            )
            bias_c = torch.cat(
                (self.model.heads[0][1].bias.data, self.model.heads[1][1].bias.data),
                dim=0,
            )
        elif "edgevit" in self.model_name.lower():
            weight_c = torch.cat(
                (self.model.heads[0].weight.data, self.model.heads[1].weight.data),
                dim=0,
            )
            bias_c = torch.cat(
                (self.model.heads[0].bias.data, self.model.heads[1].bias.data), dim=0
            )
        else:
            raise (NotImplementedError)

        self.linear_out.weight.data = weight_c
        self.linear_out.bias.data = bias_c

    def forward(self, x):
        if "mobilenet" in self.model_name.lower():
            x = self.model.features(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        elif "edgevit" in self.model_name.lower():
            x = self.model.forward_features(x)
            x = x.flatten(2).mean(-1)
        x = self.linear_out(x)
        return x


def make_parser():
    parser = argparse.ArgumentParser("ONNX convert parser")
    parser.add_argument("--config_file", required=True, help="Path to input config")
    parser.add_argument("--model_path", required=True, help="Path to input model")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch_size")
    parser.add_argument(
        "--dynamic", action="store_true", required=False, help="Dynamic model flag"
    )
    parser.add_argument(
        "--fuse", action="store_true", required=False, help="fuse the two heads"
    )
    return parser


def convertONNX(config, model_path, batch_size, dynamic, fuse):
    """
    Convert the model to ONNX format.
    Args:
        config (Config): The configuration object.
        model_path (str): The path to the model file.
        batch_size (int): The batch size for the model.
        dynamic (bool): Whether to use dynamic axes.
        fuse (bool): Whether to fuse the two heads of the model.
    """
    # Check if the model path exists
    if not model_path.endswith(".pth"):
        logging.error("Model path must end with .pth")
        return
    # Model configuration
    heads_param = []
    for key, val in config.model.heads_params.items():
        head = val.copy()
        head["name"] = key
        heads_param.append(head)
    if config.model.name == "mobilenet_v2_multi_heads":
        model = MobileNetV2MultiHeads(
            heads_param,
            in_chans=config.model.input_channels,
            width_mult=config.model.width_multiplier,
        )
    elif config.model.name == "mobilenet_v3_multi_heads":
        model = MobileNetV3MultiHeads(
            heads_param,
            in_chans=config.model.input_channels,
            width_mult=config.model.width_multiplier,
        )
    elif config.model.name == "edgevit_xxs_multi_heads":
        model = edgevit_xxs_multi_heads(
            heads_param, img_size, in_chans=1, pretrained=False
        )
    elif config.model.name == "edgevit_xs_multi_heads":
        model = edgevit_xs_multi_heads(
            heads_param, img_size, in_chans=1, pretrained=False
        )
    elif config.model.name == "edgevit_s_multi_heads":
        img_size = (config.input.input_height, config.input.input_width)
        model = edgevit_s_multi_heads(
            heads_param, img_size, in_chans=1, pretrained=False
        )
    else:
        logging.error(f"Unkown model {config.model}")

    model = torch.nn.DataParallel(model).to(torch.device("cpu"))
    model.eval()
    model.load_state_dict(torch.load(model_path))
    if fuse:
        model = ModelOneOutput(model, config.model.name)
    else:
        model = model.module
    dummy_input = torch.randn(
        batch_size,
        config.model.input_channels,
        config.input.input_height,
        config.input.input_width,
    )
    dynamic_axes_dict = {}
    if dynamic:
        dynamic_axes_dict = {
            "images": {0: "bs"},
            "output": {0: "bs"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        model_path.replace(".pth", ".onnx"),
        export_params=True,
        input_names=["images"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes_dict,
    )
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    # Get args
    args = make_parser().parse_args()

    # Get config
    config = get_config()
    config.set_config(args.config_file)

    print(
        f"Converting with batch_size {args.batch_size}, dynamic {args.dynamic}, fuse {args.fuse}"
    )
    convertONNX(config, args.model_path, args.batch_size, args.dynamic, args.fuse)
