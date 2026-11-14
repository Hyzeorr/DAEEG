# models/eegconformer.py
# CEEDNet-compatible EEGConformer-style model
# Input:  x (B, C, T)
# Output: logits (B, n_classes)
#
# Notes:
# - This is a practical EEGConformer-inspired architecture:
#   CNN stem (temporal+spatial) -> tokenization -> Transformer encoder -> pooling -> head
# - It is designed to plug into your CEEDNet training loop:
#   forward(x, age=None), get_output_length(), and optional compute_feature_embedding().
#
# If you later decide to import an official implementation (braindecode, etc.),
# you can keep this wrapper interface and swap the internals.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGConformer(nn.Module):
    """
    EEGConformer-style hybrid:
      1) CNN stem:
         - temporal conv over time
         - depthwise spatial filtering across channels
      2) Tokenization:
         - project to d_model and chunk into tokens (via conv stride)
      3) Transformer encoder (global context)
      4) Pool + classifier

    CEEDNet compatibility:
      - forward(x, age=None) -> logits
      - get_output_length() -> n_classes
      - compute_feature_embedding(x, age=None) -> (B, D)
    """

    def __init__(
        self,
        in_channels: int,
        seq_length: int,
        n_classes: int = 3,
        # ----- CNN stem -----
        stem_out: int = 32,          # channels after temporal conv
        temporal_kernel: int = 64,   # 32~128 sweep (depends on fs)
        spatial_mult: int = 2,       # depthwise spatial expansion factor
        dropout: float = 0.25,
        use_bn: bool = True,
        # ----- token / transformer -----
        d_model: int = 128,
        token_stride: int = 8,       # downsample factor along time for tokens
        n_heads: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        # ----- pooling/head -----
        pool: str = "mean",          # "mean" or "cls"
        return_feature: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert in_channels > 0 and seq_length > 0 and n_classes > 0
        assert pool in ("mean", "cls")

        self.in_channels = int(in_channels)
        self.seq_length = int(seq_length)
        self.n_classes = int(n_classes)
        self.return_feature = bool(return_feature)
        self.pool = pool

        BN2 = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())
        BN1 = (lambda c: nn.BatchNorm1d(c)) if use_bn else (lambda c: nn.Identity())

        # Input: (B, C, T) -> (B, 1, C, T)
        # Temporal conv: (1, K) over time
        self.temporal = nn.Sequential(
            nn.Conv2d(1, stem_out, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            BN2(stem_out),
            nn.ELU(inplace=True),
        )

        # Spatial filtering across channels (depthwise over channel dimension)
        # (B, stem_out, C, T) -> (B, stem_out*spatial_mult, 1, T)
        self.spatial = nn.Sequential(
            nn.Conv2d(
                stem_out,
                stem_out * spatial_mult,
                kernel_size=(in_channels, 1),
                groups=stem_out,
                bias=False,
            ),
            BN2(stem_out * spatial_mult),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

        # Tokenization: map to d_model and stride along time to create tokens
        # (B, stem_out*mult, 1, T) -> (B, d_model, T_tokens)
        self.to_tokens = nn.Sequential(
            nn.Conv2d(
                stem_out * spatial_mult,
                d_model,
                kernel_size=(1, 1),
                bias=False,
            ),
            # squeeze channel-dim later
        )

        # Optional CLS token
        if self.pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        # Positional embedding (learned) for max token length inferred
        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, seq_length)
            tok = self._extract_tokens(dummy)  # (B, Ttok, d_model)
            max_toks = tok.shape[1] + (1 if self.pool == "cls" else 0)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_toks, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=ff_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.attn_drop = nn.Dropout(attn_dropout)

        # Head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

        # Keep for CEEDNet-style access (some code may inspect output_length)
        self.output_length = n_classes

    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: tokens (B, Ttok, d_model)
        """
        x = x.unsqueeze(1)                 # (B,1,C,T)
        x = self.temporal(x)               # (B,stem_out,C,T)
        x = self.spatial(x)                # (B,stem_out*mult,1,T)
        x = self.to_tokens(x).squeeze(2)   # (B,d_model,T)
        # stride along time to form fewer tokens
        x = x[:, :, :: max(1, int(self._token_stride()))]  # (B,d_model,Ttok)
        x = x.transpose(1, 2).contiguous() # (B,Ttok,d_model)
        return x

    def _token_stride(self) -> int:
        # token_stride can be overridden by kwargs; keep safe
        # We store it as module attr if set; else fallback
        return getattr(self, "token_stride", 8)

    def compute_feature_embedding(self, x: torch.Tensor, age: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return (B, d_model) feature used before final classifier.
        age is accepted for CEEDNet compatibility but not used.
        """
        tokens = self._extract_tokens(x)  # (B,Ttok,D)

        if self.pool == "cls":
            B = tokens.size(0)
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            tokens = torch.cat([cls, tokens], dim=1)

        # Add positional embedding (crop if token len differs due to stride/seq_length changes)
        pos = self.pos_embed[:, : tokens.size(1), :]
        tokens = tokens + pos
        tokens = self.attn_drop(tokens)

        h = self.encoder(tokens)  # (B,T,D)
        h = self.norm(h)

        if self.pool == "cls":
            feat = h[:, 0, :]
        else:
            feat = h.mean(dim=1)

        return feat

    def forward(self, x: torch.Tensor, age: Optional[torch.Tensor] = None):
        feat = self.compute_feature_embedding(x, age=age)
        logits = self.classifier(feat)
        if self.return_feature:
            return logits, feat
        return logits

    def get_output_length(self) -> int:
        return int(self.n_classes)
