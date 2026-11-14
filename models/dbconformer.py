from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1d(nn.Module):
    def __init__(self, conv: nn.Conv1d, use_bn: bool = True, activation: nn.Module | None = None):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm1d(conv.out_channels) if use_bn else nn.Identity()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TemporalStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        patch_size: int,
        radix: int = 1,
        dropout: float = 0.5,
        trim_last: bool = False,
    ):
        super().__init__()
        if in_channels % radix != 0:
            raise ValueError("TemporalStem requires in_channels to be divisible by radix.")

        self.out_channels = out_channels
        self.radix = radix
        self.trim_last = trim_last

        self.channel_mix = ConvBlock1d(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels * radix,
                kernel_size=1,
                groups=radix,
                bias=False,
            ),
            use_bn=True,
            activation=None,
        )

        branch_kernel = int(kernel_size)
        self.temporal_branches = nn.ModuleList()
        for _ in range(radix):
            self.temporal_branches.append(
                ConvBlock1d(
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=max(3, branch_kernel | 1),
                        stride=1,
                        padding=max(3, branch_kernel | 1) // 2,
                        groups=out_channels,
                        bias=False,
                    ),
                    use_bn=True,
                    activation=None,
                )
            )
            branch_kernel = max(3, branch_kernel // 2)

        self.pool = nn.AvgPool1d(kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_mix(x)
        branches = torch.split(x, self.out_channels, dim=1)
        x = sum(branch(branch_input) for branch, branch_input in zip(self.temporal_branches, branches))
        x = F.gelu(x)

        if self.trim_last and x.size(-1) > 1:
            x = x[..., :-1]

        x = self.pool(x)
        x = self.dropout(x)
        return x


class TemporalPatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_size: int,
        kernel_size: int,
        patch_size: int,
        radix: int = 1,
        dropout: float = 0.5,
        trim_last: bool = False,
    ):
        super().__init__()
        self.stem = TemporalStem(
            in_channels=in_channels,
            out_channels=emb_size,
            kernel_size=kernel_size,
            patch_size=patch_size,
            radix=radix,
            dropout=dropout,
            trim_last=trim_last,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return x.transpose(1, 2).contiguous()


class SpatialPatchEmbedding(nn.Module):
    def __init__(self, spatial_dim: int, emb_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spatial_dim, kernel_size=25, stride=5, padding=12),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spatial_dim, emb_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, seq_length = x.shape
        x = x.reshape(batch_size * channels, 1, seq_length)
        x = self.encoder(x)
        return x.view(batch_size, channels, -1)


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, out_dims: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(in_features, 64), nn.ELU(inplace=True), nn.Dropout(dropout)),
                nn.Sequential(nn.Linear(64, 32), nn.ELU(inplace=True), nn.Dropout(max(0.3, dropout * 0.6))),
                nn.Linear(32, out_dims),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_until(self, x: torch.Tensor, target_from_last: int) -> torch.Tensor:
        if target_from_last < 0 or target_from_last > len(self.layers):
            raise ValueError(
                f"ClassificationHead.forward_until(target_from_last={target_from_last}) must be in "
                f"[0, {len(self.layers)}]."
            )

        if target_from_last == 0:
            return self.forward(x)

        stop_index = len(self.layers) - target_from_last
        for idx in range(stop_index):
            x = self.layers[idx](x)
        return x


class DBConformer(nn.Module):
    """
    DBConformer adapted to this repository's training interface.

    Input: (B, C, T)
    Output: logits (B, out_dims)
    """

    def __init__(
        self,
        in_channels: int,
        seq_length: int,
        out_dims: int,
        emb_size: int = 40,
        tem_depth: int = 5,
        chn_depth: int = 5,
        n_heads: int = 10,
        temporal_kernel: int = 63,
        patch_size: int = 125,
        temporal_radix: int = 1,
        spatial_dim: int = 16,
        dropout: float = 0.5,
        branch: str = "all",
        gate_flag: bool = False,
        posemb_flag: bool = True,
        chn_atten_flag: bool = True,
        trim_last: bool = False,
        return_feature: bool = False,
        **kwargs,
    ):
        super().__init__()

        if branch not in {"all", "temporal", "spatial"}:
            raise ValueError("branch must be one of {'all', 'temporal', 'spatial'}.")
        if emb_size % n_heads != 0:
            raise ValueError("emb_size must be divisible by n_heads.")

        self.in_channels = int(in_channels)
        self.seq_length = int(seq_length)
        self.out_dims = int(out_dims)
        self.branch = branch
        self.gate_flag = bool(gate_flag)
        self.posemb_flag = bool(posemb_flag)
        self.chn_atten_flag = bool(chn_atten_flag)
        self.return_feature = bool(return_feature)

        self.temporal_embedding = TemporalPatchEmbedding(
            in_channels=self.in_channels,
            emb_size=emb_size,
            kernel_size=temporal_kernel,
            patch_size=patch_size,
            radix=temporal_radix,
            dropout=dropout,
            trim_last=trim_last,
        )
        self.spatial_embedding = SpatialPatchEmbedding(spatial_dim=spatial_dim, emb_size=emb_size)

        temporal_tokens = self.temporal_embedding(torch.zeros(1, self.in_channels, self.seq_length)).size(1)
        spatial_tokens = self.in_channels

        if self.posemb_flag:
            self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, temporal_tokens, emb_size))
            self.spatial_pos_embedding = nn.Parameter(torch.zeros(1, spatial_tokens, emb_size))
        else:
            self.register_parameter("temporal_pos_embedding", None)
            self.register_parameter("spatial_pos_embedding", None)

        encoder_kwargs = {
            "d_model": emb_size,
            "nhead": n_heads,
            "dim_feedforward": emb_size * 4,
            "dropout": dropout,
            "activation": "gelu",
            "batch_first": True,
            "norm_first": True,
        }
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**encoder_kwargs),
            num_layers=tem_depth,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**encoder_kwargs),
            num_layers=chn_depth,
        )

        if self.gate_flag or self.branch in {"temporal", "spatial"}:
            classifier_in_features = emb_size
            self.gate_fc = nn.Linear(emb_size * 2, emb_size)
        else:
            classifier_in_features = emb_size * 2
            self.gate_fc = None

        self.classifier = ClassificationHead(classifier_in_features, self.out_dims, dropout=dropout)

        if self.chn_atten_flag:
            self.spatial_attn_pool = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.Tanh(),
                nn.Linear(emb_size, 1),
            )
        else:
            self.spatial_attn_pool = None

        self.output_length = self.out_dims
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.temporal_pos_embedding is not None:
            nn.init.trunc_normal_(self.temporal_pos_embedding, std=0.02)
        if self.spatial_pos_embedding is not None:
            nn.init.trunc_normal_(self.spatial_pos_embedding, std=0.02)

    def _encode_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_temporal = self.temporal_embedding(x)
        x_spatial = self.spatial_embedding(x)

        if self.posemb_flag:
            x_temporal = x_temporal + self.temporal_pos_embedding[:, : x_temporal.size(1), :]
            x_spatial = x_spatial + self.spatial_pos_embedding[:, : x_spatial.size(1), :]

        x_temporal = self.temporal_transformer(x_temporal)
        x_spatial = self.spatial_transformer(x_spatial)
        return x_temporal, x_spatial

    def _fuse_features(self, x_temporal: torch.Tensor, x_spatial: torch.Tensor) -> torch.Tensor:
        temporal_feature = x_temporal.mean(dim=1)

        if self.branch == "temporal":
            return temporal_feature

        if self.chn_atten_flag and self.spatial_attn_pool is not None:
            attn_scores = self.spatial_attn_pool(x_spatial)
            attn_weights = torch.softmax(attn_scores, dim=1)
            spatial_feature = torch.sum(attn_weights * x_spatial, dim=1)
        else:
            spatial_feature = x_spatial.mean(dim=1)

        if self.branch == "spatial":
            return spatial_feature

        if self.gate_flag and self.gate_fc is not None:
            gate = torch.sigmoid(self.gate_fc(torch.cat([temporal_feature, spatial_feature], dim=-1)))
            return gate * spatial_feature + (1.0 - gate) * temporal_feature

        return torch.cat([temporal_feature, spatial_feature], dim=-1)

    def compute_feature_embedding(
        self,
        x: torch.Tensor,
        age: Optional[torch.Tensor] = None,
        target_from_last: int = 1,
    ) -> torch.Tensor:
        x_temporal, x_spatial = self._encode_branches(x)
        fused = self._fuse_features(x_temporal, x_spatial)

        if target_from_last == 0:
            return self.classifier(fused)

        if target_from_last > len(self.classifier.layers):
            raise ValueError(
                f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives an integer "
                f"equal to or smaller than {len(self.classifier.layers)}."
            )

        return self.classifier.forward_until(fused, target_from_last=target_from_last)

    def forward(self, x: torch.Tensor, age: Optional[torch.Tensor] = None):
        x_temporal, x_spatial = self._encode_branches(x)
        fused = self._fuse_features(x_temporal, x_spatial)
        logits = self.classifier(fused)

        if self.return_feature:
            return logits, fused
        return logits

    def get_output_length(self) -> int:
        return self.output_length
