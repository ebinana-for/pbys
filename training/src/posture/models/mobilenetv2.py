#!/usr/bin/env python3
"""MobileNetV2 based posture estimation models with different heads.
This module implements various MobileNetV2-based models for posture estimation,
including:
- MobileNetV2TrPosture: Uses a Transformer encoder for keypoint processing.
- MobileNetV2MlpPosture: Uses a Multi-Layer Perceptron (MLP) for keypoint processing.
- MobileNetV2Posture: Standard MobileNetV2 for posture estimation.
- MobileNetV2PostureMultiHeads: Supports multiple heads for different outputs.
Each model can be customized with different parameters such as input channels,
width multiplier, dropout rate, and keypoint processing dimensions.
"""
from typing import Callable, List, Optional
import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNetV2
from copy import deepcopy
import math
import copy
import pdb


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in the "Attention Is All You Need" paper.
    """
    def __init__(self, d_model, num_heads):
        """
        Initialize the MultiHeadAttention module.
        Args:
            d_model (int): Dimension of the model (input and output).
            num_heads (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = (
            d_model // num_heads
        )  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided 
        # (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class EncoderLayer(torch.nn.Module):
    """
    Implement the Transformer’s Encoder layer
    Based on th paper called "Attention Is All You Need"
    """

    def __init__(
        self, embed_dim: int, n_heads: int, dim_feedforward: int = 128, dropout=0.1
    ):
        """
        Initialize the EncoderLayer.
        Args:
            embed_dim (int): Dimension of the input embeddings.
            n_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.mha = MultiHeadAttention(embed_dim, num_heads=n_heads)
        self.layer_norm1 = torch.nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(normalized_shape=embed_dim)

        # section 5.4
        # apply dropout to output of each sublayer before 
        # it is added to sublayer's input
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)

        # section 3.3 in paper
        self.position_wise_ff = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=embed_dim, out_features=dim_feedforward, bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=dim_feedforward, out_features=embed_dim, bias=True
            ),
        )

    def forward(self, x, src_mask=None):
        # x.shape = (batch_size, seq_len, embed_dim)
        # src_key_padding_mask = (bs, seq_len), 
        # True value indicates it should not attend
        # src_mask.shape = (bs, seq_len, seq_len) of dtype torch.bool, 
        # True value indicates it shouldn't attend
        x1 = x.detach().clone()
        x2 = x.detach().clone()
        attn_output = self.mha(x, x1, x2, mask=src_mask)
        # dropout and residual connection
        x = x + self.dropout1(attn_output)
        x = self.layer_norm1(x)

        projection = self.position_wise_ff(x)
        # dropout and residual connection
        x = x + self.dropout2(projection)
        # layer norm
        x = self.layer_norm2(x)
        return x


class Encoder(torch.nn.Module):
    """
    Implement the Transformer’s Encoder
    Based on the paper called "Attention Is All You Need"
    """   
    def __init__(self, encoder_layer, num_layers: int):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(deepcopy(encoder_layer))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, src_mask=None):
        # x.shape = (batch_size, channels, height, width)
        # kpts.shape = (batch_size, 1, embed_dim)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x


class MobileNetV2TrPosture(MobileNetV2):
    """
    MobileNetV2 model with a Transformer encoder for keypoint processing.
    """
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
        kpt_d_in=24,  # keypoint input dim
        kpt_d_hid=24,  # dim of the feedforward network model in nn.TransformerEncoder
        kpt_nlayers=1,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        kpt_nhead=3,  # number of heads in nn.MultiheadAttention
        kpt_dropout=0.05,
    ) -> None:
        """
        Initialize the MobileNetV2TrPosture model.
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
            kpt_d_in (int): Dimension of the keypoint input (default: 24).
            kpt_d_hid (int): Dimension of the hidden layer in the MLP (default: 24).
            kpt_nlayers (int): Number of layers in the Transformer encoder (default: 1).
            kpt_nhead (int): Number of heads in the Multi-Head Attention (default: 3).
            kpt_dropout (float): Dropout rate for the keypoint processing (default: 0.05).
        Returns:
            None
        """
        # Initialize the MobileNetV2 base class

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
        self.n_inputs = 2  # 2 input in forward method

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

        # Create keypoints backbone
        encoder_layers = EncoderLayer(
            kpt_d_in, kpt_nhead, dim_feedforward=kpt_d_hid, dropout=kpt_dropout
        )
        self.transformer_encoder = Encoder(encoder_layers, kpt_nlayers)

        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        # self.kpt_mask = torch.triu(
        # torch.ones(kpt_d_in, kpt_d_in) * float('-inf'), diagonal=1)
        head = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.last_channel + kpt_d_in, head_param[0]["out_dim"]),
        )
        self.head = head

    def _forward_impl(self, x: torch.Tensor, kpts: torch.Tensor):
        # x.shape = (batch_size, channels, height, width)
        # kpts.shape = (batch_size, 1, embed_dim)
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        kpts_output = self.transformer_encoder(kpts)
        x_in = torch.cat((x, kpts_output[:, 0, :]), dim=1)
        x_out = self.head(x_in)
        return x_out

    def forward(self, x: torch.Tensor, kpts: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, channels, height, width)
        # kpts.shape = (batch_size, 1, embed_dim)
        return self._forward_impl(x, kpts)


class MobileNetV2MlpPosture(MobileNetV2):
    """ MobileNetV2 model with a Multi-Layer Perceptron (MLP) for keypoint processing."""
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
        kpt_d_in=24,  # keypoint input dim
        kpt_d_hid=24,  # dimension of the MLP hidden layer
        kpt_dropout=0.05,
    ) -> None:
        """
        Initialize the MobileNetV2MlpPosture model.
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
            kpt_d_in (int): Dimension of the keypoint input (default: 24).
            kpt_d_hid (int): Dimension of the hidden layer in the MLP (default: 24).
            kpt_dropout (float): Dropout rate for the keypoint processing (default: 0.05).
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
        self.n_inputs = 2  # 2 input in forward method

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

        # Create keypoints backbone
        self.kpt_mlp = nn.Sequential(
            nn.Dropout(kpt_dropout),
            nn.Linear(kpt_d_in, kpt_d_hid),
            nn.Linear(kpt_d_hid, kpt_d_in),
        )

        head = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.last_channel + kpt_d_in, head_param[0]["out_dim"]),
        )

        self.head = head

    def _forward_impl(self, x: torch.Tensor, kpts: torch.Tensor):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        kpts_output = self.kpt_mlp(kpts[:, 0, :])
        x_in = torch.cat((x, kpts_output), dim=1)
        x_out = self.head(x_in)
        return x_out

    def forward(self, x: torch.Tensor, kpts: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, channels, height, width)
        # kpts.shape = (batch_size, 1, embed_dim)
        return self._forward_impl(x, kpts)


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


class MobileNetV2PostureMultiHeads(MobileNetV2):
    """ MobileNetV2 model for posture estimation with multiple heads."""
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
        """Initialize the MobileNetV2PostureMultiHeads model.
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

        from torchvision.models._utils import _make_divisible
        from torchvision.ops.misc import Conv2dNormActivation

        self.dropout = dropout
        self.n_inputs = 1  # 2 input in forward method

        # Change the number of channels in the first layer, if needed
        if in_chans != 3:
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

        # Create heads
        del self.classifier
        self.heads = nn.ModuleList()
        if len(self.head_param) > 1:
            common_layers = list(self.features.children())[:-2]
            for elt in self.head_param:
                # key, elt = next(iter(elt0.items()))
                assert (
                    "out_dim" in elt.keys()
                ), "You must specify out_dim for head {}".format(elt["name"])
                assert "loss" in elt.keys(), "You must specify loss for head {}".format(
                    elt["name"]
                )
                assert (
                    "loss_weight" in elt.keys()
                ), "You must specify loss_weight for head {}".format(elt["name"])
                head_layer = copy.deepcopy(list(self.features.children())[-2:])
                head = nn.Sequential(
                    *head_layer,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(start_dim=1),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(self.last_channel, elt["out_dim"])
                )
                self.heads.append(head)
            self.features = torch.nn.Sequential(*common_layers)
        else:
            self.heads.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(start_dim=1),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(
                        self.last_channel,
                        [elt["out_dim"] for _, elt in self.head_param.items()][0],
                    ),
                )
            )

    def _forward_impl(self, x: torch.Tensor):
        x = self.features(x)
        out = []
        for head in self.heads:
            x_out = head(x)
            out.append(x_out)
        return torch.concat(out, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def main():
    from src.utils.config import get_config
    from src import PROJECT_DIR

    config_path = PROJECT_DIR / "config/posture_multiclass_2heads_jd.yaml"
    config = get_config()
    config.set_config(str(config_path))
    head_param = {key: val for key, val in config.model.head_params.items()}

    net0 = MobileNetV2TrPosture(head_param, in_chans=1)
    # net1 = MobileNetV2MlpPosture(head_param, in_chans=1)
    # net2 = MobileNetV2Posture(head_param, in_chans=1)
    net3 = MobileNetV2PostureMultiHeads(head_param, in_chans=1)

    img = torch.rand(8, 1, 320, 192)
    kpts = torch.rand(8, 1, 24)
    net0(img, kpts)
    # net1(img, kpts)
    # net2(img)
    net3(img)
    pdb.set_trace()


if __name__ == "__main__":
    main()
