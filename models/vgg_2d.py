"""
Modified from:
    - torchvision implementation of VGG (linked below)
    - https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
    - VGG paper: https://arxiv.org/abs/1409.1556
"""

from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import program_conv_filters
from .activation import get_activation_class


vgg_layer_cfgs: Dict[str, List[Dict[str, int]]] = {
    "2D-VGG-11": [
        {"layers": 1, "channel_mul": 1},
        {"layers": 1, "channel_mul": 2},
        {"layers": 2, "channel_mul": 4},
        {"layers": 2, "channel_mul": 8},
        {"layers": 2, "channel_mul": 8},
    ],
    "2D-VGG-13": [
        {"layers": 2, "channel_mul": 1},
        {"layers": 2, "channel_mul": 2},
        {"layers": 2, "channel_mul": 4},
        {"layers": 2, "channel_mul": 8},
        {"layers": 2, "channel_mul": 8},
    ],
    "2D-VGG-16": [
        {"layers": 2, "channel_mul": 1},
        {"layers": 2, "channel_mul": 2},
        {"layers": 3, "channel_mul": 4},
        {"layers": 3, "channel_mul": 8},
        {"layers": 3, "channel_mul": 8},
    ],
    "2D-VGG-19": [
        {"layers": 2, "channel_mul": 1},
        {"layers": 2, "channel_mul": 2},
        {"layers": 4, "channel_mul": 4},
        {"layers": 4, "channel_mul": 8},
        {"layers": 4, "channel_mul": 8},
    ],
}


class VGG2D(nn.Module):
    def __init__(
        self,
        model: str,
        in_channels: int,
        out_dims: int,
        seq_len_2d: Tuple[int],
        use_age: str,
        base_channels: int = 64,
        dropout: float = 0.5,
        batch_norm: bool = False,
        fc_stages: int = 2,
        base_pool: str = "max",
        final_pool: str = "average",
        activation: str = "relu",
        multiscale: Tuple[int, ...] = (1,),
        attn_pool: bool = False,
        attn_heads: int = 4,
        film_age: bool = False,
        film_hidden: int = 64,
        **kwargs,
    ):
        super().__init__()

        if model not in vgg_layer_cfgs.keys():
            raise ValueError(
                f"{self.__class__.__name__}.__init__(model) " f"receives one of [{vgg_layer_cfgs.keys()}]."
            )

        if use_age not in ["fc", "conv", "no"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) " f"receives one of ['fc', 'conv', 'no'].")

        if final_pool not in ["average", "max"] or base_pool not in ["average", "max"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(final_pool, base_pool) both "
                f"receives one of ['average', 'max']."
            )

        if fc_stages < 1:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(fc_stages) receives " f"an integer equal to ore more than 1."
            )

        self.use_age = use_age
        if self.use_age == "conv":
            in_channels += 1
        self.multiscale = tuple(sorted(set(multiscale))) if multiscale else (1,)
        if any(s < 1 for s in self.multiscale):
            raise ValueError(f"{self.__class__.__name__}.__init__(multiscale) receives positive integers.")
        self.fc_stages = fc_stages

        self.batch_norm = batch_norm
        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)

        if base_pool == "average":
            self.base_pool = nn.AvgPool2d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool2d

        layer_cfgs = vgg_layer_cfgs[model]
        conv_filter_list = []
        for i in layer_cfgs:  # to prevent shallow copying
            conv_filter_list.append({"kernel_size": 3})
        self.seq_len_2d = seq_len_2d
        self.output_length = program_conv_filters(
            sequence_length=min(seq_len_2d),
            conv_filter_list=conv_filter_list,
            output_lower_bound=4,
            output_upper_bound=8,
            stride_to_pool_ratio=1.5,
            class_name=self.__class__.__name__,
        )

        # optional multiscale stem
        self.stem_scales = None
        stem_out_channels = in_channels
        if len(self.multiscale) > 1 or self.multiscale[0] != 1:
            stems: List[nn.Module] = []
            for s in self.multiscale:
                layers: List[nn.Module] = []
                if s > 1:
                    layers.append(nn.AvgPool2d(kernel_size=s, stride=s))
                if self.batch_norm:
                    layers += [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(in_channels),
                        self.nn_act(),
                    ]
                else:
                    layers += [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                        self.nn_act(),
                    ]
                stems.append(nn.Sequential(*layers))
            self.stem_scales = nn.ModuleList(stems)
            stem_out_channels = in_channels * len(self.multiscale)

        # convolution stage
        self.current_channels = stem_out_channels
        self.conv_stage1 = self._make_conv_stage(conv_filter_list[0], layer_cfgs[0], base_channels)
        self.conv_stage2 = self._make_conv_stage(conv_filter_list[1], layer_cfgs[1], base_channels)
        self.conv_stage3 = self._make_conv_stage(conv_filter_list[2], layer_cfgs[2], base_channels)
        self.conv_stage4 = self._make_conv_stage(conv_filter_list[3], layer_cfgs[3], base_channels)
        self.conv_stage5 = self._make_conv_stage(conv_filter_list[4], layer_cfgs[4], base_channels)
        self.conv_out_channels = self.current_channels

        # pooling right before fully-connection
        self.use_attn_pool = attn_pool
        if self.use_attn_pool:
            if self.conv_out_channels % attn_heads != 0:
                raise ValueError(
                    f"{self.__class__.__name__}.__init__(attn_heads) requires conv_out_channels % attn_heads == 0."
                )
            self.attn_query = nn.Parameter(torch.zeros(1, 1, self.conv_out_channels))
            self.attn_pool = nn.MultiheadAttention(
                embed_dim=self.conv_out_channels, num_heads=attn_heads, batch_first=False
            )
            self.attn_norm = nn.LayerNorm(self.conv_out_channels)
            nn.init.normal_(self.attn_query, mean=0.0, std=0.02)
        else:
            if final_pool == "average":
                self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
            elif final_pool == "max":
                self.final_pool = nn.AdaptiveMaxPool2d((1, 1))

        # optional FiLM conditioning on age for conv features
        self.film_age = film_age
        if self.film_age:
            self.film_mlp = nn.Sequential(
                nn.Linear(1, film_hidden),
                self.nn_act(),
                nn.Linear(film_hidden, 2 * self.conv_out_channels),
            )

        # fully-connected stage
        fc_stage: List[nn.Module] = []
        self.current_channels = self.conv_out_channels
        if self.use_age == "fc":
            self.current_channels = self.current_channels + 1

        for i in range(fc_stages - 1):
            if self.batch_norm:
                layer = nn.Sequential(
                    nn.Linear(self.current_channels, self.current_channels // 2, bias=False),
                    nn.Dropout(p=dropout),
                    nn.BatchNorm1d(self.current_channels // 2),
                    self.nn_act(),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(self.current_channels, self.current_channels // 2, bias=True),
                    nn.Dropout(p=dropout),
                    self.nn_act(),
                )
            self.current_channels = self.current_channels // 2
            fc_stage.append(layer)

        fc_stage.append(nn.Linear(self.current_channels, out_dims, bias=True))
        self.fc_stage = nn.Sequential(*fc_stage)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear,)):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _make_conv_stage(self, conv_filter, cfg, base_channels):
        conv_layers: List[nn.Module] = []

        if conv_filter["pool"] > 1:
            conv_layers += [self.base_pool(conv_filter["pool"])]

        for k in range(cfg["layers"]):
            if k == 0:
                stride = conv_filter["stride"]
            else:
                stride = 1

            if self.batch_norm:
                conv_layers += [
                    nn.Conv2d(
                        in_channels=self.current_channels,
                        out_channels=cfg["channel_mul"] * base_channels,
                        kernel_size=conv_filter["kernel_size"],
                        padding=conv_filter["kernel_size"] // 2,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(cfg["channel_mul"] * base_channels),
                    self.nn_act(),
                ]
            else:
                conv_layers += [
                    nn.Conv2d(
                        in_channels=self.current_channels,
                        out_channels=cfg["channel_mul"] * base_channels,
                        kernel_size=conv_filter["kernel_size"],
                        padding=conv_filter["kernel_size"] // 2,
                        stride=stride,
                        bias=True,
                    ),
                    self.nn_act(),
                ]

            self.current_channels = cfg["channel_mul"] * base_channels
        return nn.Sequential(*conv_layers)

    def get_output_length(self):
        return self.output_length

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        # keep the original scalar age for FiLM/FC use
        age_scalar = age

        if self.use_age == "conv":
            N, _, H, W = x.size()
            age_conv = age_scalar.reshape((N, 1, 1, 1)).expand(N, 1, H, W)
            x = torch.cat((x, age_conv), dim=1)

        if self.stem_scales is not None:
            stem_feats = []
            for stem, scale in zip(self.stem_scales, self.multiscale):
                xs = stem(x)
                if scale > 1:
                    xs = F.interpolate(xs, size=(x.size(2), x.size(3)), mode="bilinear")
                stem_feats.append(xs)
            x = torch.cat(stem_feats, dim=1)

        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)
        x = self.conv_stage5(x)

        if self.use_attn_pool:
            B, C, H, W = x.shape
            tokens = x.flatten(2).permute(2, 0, 1)  # (HW, B, C)
            tokens = self.attn_norm(tokens)  # stabilize attention weights
            q = self.attn_query.expand(1, B, C)
            pooled, _ = self.attn_pool(q, tokens, tokens, need_weights=False)
            x = pooled.squeeze(0)
        else:
            x = self.final_pool(x)
            x = torch.flatten(x, 1)

        if self.film_age:
            if x.dim() != 2 or x.shape[0] != age_scalar.shape[0] or x.shape[1] != self.conv_out_channels:
                raise ValueError(
                    f"{self.__class__.__name__}: FiLM expects pooled features of shape "
                    f"(batch={age_scalar.shape[0]}, channels={self.conv_out_channels}); "
                    f"got x={tuple(x.shape)}, age={tuple(age_scalar.shape)}. "
                    f"Check multiscale/attn_pool settings or disable film_age."
                )
            age_feat = self.film_mlp(age_scalar.reshape(-1, 1))
            gamma, beta = torch.chunk(age_feat, 2, dim=1)
            x = gamma * x + beta

        if self.use_age == "fc":
            x = torch.cat((x, age_scalar.reshape(-1, 1)), dim=1)

        if target_from_last == 0:
            x = self.fc_stage(x)
        else:
            if target_from_last > self.fc_stages:
                raise ValueError(
                    f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                    f"an integer equal to or smaller than fc_stages={self.fc_stages}."
                )

            for l in range(self.fc_stages - target_from_last):
                x = self.fc_stage[l](x)
        return x

    def forward(self, x, age):
        x = self.compute_feature_embedding(x, age)
        # return F.log_softmax(x, dim=1)
        return x
