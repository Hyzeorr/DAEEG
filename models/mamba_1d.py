#모든 방법론 적용한 것
import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

# Optional dependency
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


# -------------------------------------------------
# Squeeze-and-Excitation
# -------------------------------------------------
class SqueezeExcitation(nn.Module):
    """Channel attention over (B, C, L) tensors."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)      # (B, C)
        weights = self.fc(weights).unsqueeze(-1)  # (B, C, 1)
        return x * weights


# -------------------------------------------------
# LoRA residual branch
# -------------------------------------------------
class LoRAResidual(nn.Module):
    """Lightweight residual path used for parameter-efficient tuning."""

    def __init__(self, dim: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.rank = rank

        if rank <= 0:
            self.lora_down = None
            self.lora_up = None
        else:
            self.lora_down = nn.Linear(dim, rank, bias=False)
            self.lora_up = nn.Linear(rank, dim, bias=False)
            self.scaling = alpha / rank
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank <= 0:
            return torch.zeros_like(x)

        update = self.lora_down(x)
        update = self.dropout(update)
        update = self.lora_up(update)
        return update * self.scaling


# -------------------------------------------------
# Adapter
# -------------------------------------------------
class Adapter(nn.Module):
    """Bottleneck adapter inserted after each Mamba block."""

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------
# Multi-scale convolution stem
# -------------------------------------------------
class MultiScaleStem(nn.Module):
    """Parallel 1D convolutions to capture local EEG patterns."""

    def __init__(self, in_channels: int, out_channels: int, kernels: Sequence[int]):
        super().__init__()

        if not isinstance(kernels, Iterable):
            raise ValueError("kernels must be an iterable of kernel sizes.")

        kernels = list(kernels)
        num_branches = len(kernels)
        if num_branches < 1:
            raise ValueError("At least one kernel size is required.")

        base = out_channels // num_branches
        remainder = out_channels - base * num_branches

        branch_dims: List[int] = []
        for i in range(num_branches):
            branch_dims.append(base + (1 if i < remainder else 0))

        self.branches = nn.ModuleList()
        for dim, k in zip(branch_dims, kernels):
            if k % 2 == 0:
                raise ValueError(f"Kernel size {k} must be odd.")
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, dim, kernel_size=k, padding=padding),
                    nn.BatchNorm1d(dim),
                    nn.GELU(),
                )
            )

        self.fuse = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        return self.fuse(x)


# -------------------------------------------------
# Mamba block
# -------------------------------------------------
class MambaBlock(nn.Module):
    """Single Mamba block with residual + LayerNorm + optional Adapter/LoRA."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        adapter_hidden: int = 0,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError("Install mamba-ssm first.")

        self.block = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.adapter = Adapter(d_model, adapter_hidden, dropout) if adapter_hidden > 0 else None
        self.lora = LoRAResidual(d_model, lora_rank, lora_alpha, lora_dropout) if lora_rank > 0 else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block(x)
        x = self.dropout(x)
        x = x + residual

        if self.adapter is not None:
            x = x + self.adapter(x)

        if self.lora is not None:
            x = x + self.lora(x)

        return self.norm(x)


# -------------------------------------------------
# Mamba1D encoder
# -------------------------------------------------
class Mamba1D(nn.Module):
    """Mamba-based encoder for 1D EEG signals (B, C, L)."""

    def __init__(
        self,
        in_channels: int,
        out_dims: int,
        seq_length: int,
        use_age: str,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fc_stages: int = 2,
        dropout: float = 0.1,
        multi_scale_conv: bool = True,
        multi_scale_kernels: Sequence[int] = (3, 5, 7),
        channel_attention: str = "se",
        se_reduction: int = 8,
        adapter_hidden: int = 0,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        if use_age not in ["fc", "conv", "no"]:
            raise ValueError("use_age must be one of ['fc', 'conv', 'no'].")

        self.use_age = use_age
        self.sequence_length = seq_length
        self.output_length = 1
        self.fc_stages = fc_stages

        in_channels_proj = in_channels + (1 if use_age == "conv" else 0)

        if multi_scale_conv:
            self.input_proj = MultiScaleStem(
                in_channels_proj, d_model, multi_scale_kernels
            )
        else:
            self.input_proj = nn.Conv1d(in_channels_proj, d_model, kernel_size=1)

        if channel_attention == "se":
            self.channel_attention = SqueezeExcitation(d_model, se_reduction)
        else:
            self.channel_attention = None

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    adapter_hidden=adapter_hidden,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        hidden_dim = d_model + (1 if use_age == "fc" else 0)
        fc_layers = []
        self.fc_stage = nn.ModuleList()

        for _ in range(fc_stages - 1):
            stage = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fc_stage.append(stage)
            fc_layers.append(stage)
            hidden_dim //= 2

        final_layer = nn.Linear(hidden_dim, out_dims)
        self.fc_stage.append(final_layer)
        fc_layers.append(final_layer)
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        x = self.compute_feature_embedding(x, age, target_from_last=0)
        return x

    def compute_feature_embedding(
        self, x: torch.Tensor, age: torch.Tensor, target_from_last: int = 0
    ) -> torch.Tensor:
        if self.use_age == "conv":
            age_ch = age.unsqueeze(-1).expand(-1, 1, x.size(-1))
            x = torch.cat([x, age_ch], dim=1)

        x = self.input_proj(x)

        if self.channel_attention is not None:
            x = self.channel_attention(x)

        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)

        if self.use_age == "fc":
            x = torch.cat([x, age.view(-1, 1)], dim=1)

        if target_from_last == 0:
            return self.fc(x)

        if target_from_last > self.fc_stages:
            raise ValueError(
                f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                f"an integer equal to or smaller than fc_stages={self.fc_stages}."
            )

        for l in range(self.fc_stages - target_from_last):
            x = self.fc_stage[l](x)
        return x

    def get_output_length(self) -> int:
        return self.output_length
'''
# adapter만 적용한거
import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

# Optional dependency
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


# =================================================
# Squeeze-and-Excitation
# =================================================
class SqueezeExcitation(nn.Module):
    """Channel attention over (B, C, L) tensors."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)          # (B, C)
        weights = self.fc(weights).unsqueeze(-1)    # (B, C, 1)
        return x * weights


# =================================================
# LoRA residual branch (ONLY adaptation module)
# =================================================
class LoRAResidual(nn.Module):
    """Low-rank residual adaptation module."""

    def __init__(self, dim: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.rank = rank

        if rank <= 0:
            raise ValueError("LoRA rank must be > 0 for LoRA-only ablation.")

        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.lora_down(x)
        update = self.dropout(update)
        update = self.lora_up(update)
        return update * self.scaling


# =================================================
# Multi-scale convolution stem
# =================================================
class MultiScaleStem(nn.Module):
    """Parallel 1D convolutions for EEG signal encoding."""

    def __init__(self, in_channels: int, out_channels: int, kernels: Sequence[int]):
        super().__init__()

        if not isinstance(kernels, Iterable):
            raise ValueError("kernels must be iterable.")

        kernels = list(kernels)
        num_branches = len(kernels)
        base = out_channels // num_branches
        remainder = out_channels - base * num_branches

        branch_dims: List[int] = [
            base + (1 if i < remainder else 0) for i in range(num_branches)
        ]

        self.branches = nn.ModuleList()
        for dim, k in zip(branch_dims, kernels):
            if k % 2 == 0:
                raise ValueError(f"Kernel size {k} must be odd.")
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, dim, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(dim),
                    nn.GELU(),
                )
            )

        self.fuse = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        return self.fuse(x)


# =================================================
# Mamba block (LoRA-only)
# =================================================
class MambaBlockLoRA(nn.Module):
    """Mamba block with residual + LoRA (NO Adapter)."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError("Install mamba-ssm first.")

        self.block = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.dropout = nn.Dropout(dropout)
        self.lora = LoRAResidual(
            dim=d_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block(x)
        x = self.dropout(x)
        x = x + residual            # base residual
        x = x + self.lora(x)        # LoRA residual ONLY
        return self.norm(x)


# =================================================
# Mamba1D encoder (LoRA-only ablation)
# =================================================
class Mamba1D(nn.Module):
    """LoRA-only Mamba encoder for EEG classification."""

    def __init__(
        self,
        in_channels: int,
        out_dims: int,
        seq_length: int,
        use_age: str,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fc_stages: int = 2,
        dropout: float = 0.1,
        multi_scale_conv: bool = True,
        multi_scale_kernels: Sequence[int] = (3, 5, 7),
        channel_attention: str = "se",
        se_reduction: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__()

        if use_age not in ["fc", "conv", "no"]:
            raise ValueError("use_age must be one of ['fc', 'conv', 'no'].")

        self.use_age = use_age
        self.sequence_length = seq_length
        self.output_length = 1

        in_ch = in_channels + (1 if use_age == "conv" else 0)

        # Input projection
        if multi_scale_conv:
            self.input_proj = MultiScaleStem(
                in_ch, d_model, multi_scale_kernels
            )
        else:
            self.input_proj = nn.Conv1d(in_ch, d_model, kernel_size=1)

        # Channel attention
        self.channel_attention = (
            SqueezeExcitation(d_model, se_reduction)
            if channel_attention == "se"
            else None
        )

        # Mamba blocks (LoRA-only)
        self.blocks = nn.ModuleList(
            [
                MambaBlockLoRA(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        hidden_dim = d_model + (1 if use_age == "fc" else 0)
        fc_layers = []

        for _ in range(fc_stages - 1):
            fc_layers += [
                nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            hidden_dim //= 2

        fc_layers.append(nn.Linear(hidden_dim, out_dims))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        if self.use_age == "conv":
            age_ch = age.unsqueeze(-1).expand(-1, 1, x.size(-1))
            x = torch.cat([x, age_ch], dim=1)

        x = self.input_proj(x)

        if self.channel_attention is not None:
            x = self.channel_attention(x)

        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)

        if self.use_age == "fc":
            x = torch.cat([x, age.view(-1, 1)], dim=1)

        return self.fc(x)

    def get_output_length(self) -> int:
        return self.output_length
'''
