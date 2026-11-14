# models/xeeegnet.py
# xEEGNet-like compact CNN for EEG classification
# Input:  x (B, C, T)
# Output: logits (B, n_classes)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSamePad(nn.Module):
    """Minimal SAME padding for Conv2d on time dimension.
    We use (1, k) kernels so only time axis matters. This keeps T roughly stable.
    """
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T) where we convolve along T with kernel (1, k)
        kH, kW = self.conv.kernel_size
        sH, sW = self.conv.stride
        dH, dW = self.conv.dilation

        # only pad width (time)
        eff_kW = (kW - 1) * dW + 1
        outW = (x.size(-1) + sW - 1) // sW
        padW = max(0, (outW - 1) * sW + eff_kW - x.size(-1))
        pad_left = padW // 2
        pad_right = padW - pad_left

        if pad_left or pad_right:
            x = F.pad(x, (pad_left, pad_right, 0, 0))
        return self.conv(x)


class XEEGNet(nn.Module):
    """
    A practical xEEGNet-style architecture (EEGNet family):
      - Temporal conv on each channel (implemented as Conv2d with kernel (1, K))
      - Depthwise spatial filtering across channels (kernel (C, 1))
      - Separable conv (depthwise temporal + pointwise)
      - Global pooling + classifier

    Notes:
      - This is designed to be a strong compact baseline and plug into CEEDNet.
      - If you later obtain the authors' official implementation, you can swap it
        behind the same class name / signature.
    """

    def __init__(
        self,
        in_channels: int,
        seq_length: int,
        n_classes: int=2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        pool_size_1: int = 4,
        pool_size_2: int = 8,
        dropout: float = 0.25,
        use_bn: bool = True,
        return_feature: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert in_channels > 0 and seq_length > 0 and n_classes > 0

        self.in_channels = in_channels
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.return_feature = return_feature
        self.output_length = 1

        bn = (lambda c: nn.BatchNorm2d(c, eps=1e-5, momentum=0.1)) if use_bn else (lambda c: nn.Identity())

        # Input reshape: (B,C,T) -> (B,1,C,T)
        self.temporal = nn.Sequential(
            Conv2dSamePad(nn.Conv2d(1, F1, kernel_size=(1, kernel_length), stride=(1, 1), bias=False)),
            bn(F1),
        )

        # Depthwise spatial conv across channels: (C,1) with groups=F1
        self.spatial = nn.Sequential(
            nn.Conv2d(
                F1,
                F1 * D,
                kernel_size=(in_channels, 1),
                groups=F1,
                bias=False,
            ),
            bn(F1 * D),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool_size_1), stride=(1, pool_size_1)),
            nn.Dropout(p=dropout),
        )

        # Separable conv: depthwise temporal then pointwise
        self.separable = nn.Sequential(
            # depthwise temporal
            Conv2dSamePad(
                nn.Conv2d(
                    F1 * D,
                    F1 * D,
                    kernel_size=(1, 16),
                    groups=F1 * D,
                    bias=False,
                )
            ),
            # pointwise
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            bn(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool_size_2), stride=(1, pool_size_2)),
            nn.Dropout(p=dropout),
        )

        # infer feature dim
        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, seq_length)
            feat = self._forward_features(dummy)  # (B, F)
            feat_dim = feat.shape[-1]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        x = x.unsqueeze(1)  # (B,1,C,T)
        x = self.temporal(x)   # (B,F1,C,T)
        x = self.spatial(x)    # (B,F1*D,1,T')
        x = self.separable(x)  # (B,F2,1,T'')
        x = x.squeeze(2)       # (B,F2,T'')
        x = x.mean(dim=-1)     # global avg pool over time -> (B,F2)
        return x

    def forward(self, x: torch.Tensor, age: torch.Tensor | None = None):
        feat = self._forward_features(x)
        logits = self.classifier(feat)
        if self.return_feature:
            return logits, feat
        return logits

    def get_output_length(self) -> int:
        return int(self.n_classes)