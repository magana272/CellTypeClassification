import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MLP_SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + SE."""

    def __init__(self, in_ch, out_ch, kernel=5, dropout=0.2):
        super().__init__()
        pad = kernel // 2
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.drop = nn.Dropout(dropout)
        self.se = MLP_SEBlock(out_ch)
        self.act = nn.ReLU()
        self.proj = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.conv1(self.act(self.bn1(x)))
        out = self.drop(out)
        out = self.conv2(self.act(self.bn2(out)))
        out = self.se(out)
        return out + identity


class CellTypeCNN(nn.Module):
    """Residual 1D-CNN with variable depth, widening channels, SE attention, and dual-pool head."""

    def __init__(self, seq_len: int, n_classes: int, dropout: float = 0.1,
                 n_stages: int = 3, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # Channel schedule: double every other stage, cap at 256
        channels = [min(32 * 2 ** (i // 2), 256) for i in range(n_stages)]
        # Kernel schedule: start at 7, decrease to 3
        kernels = [max(7 - 2 * i, 3) for i in range(n_stages)]

        stages = []
        in_ch = 32  # stem output
        for ch, k in zip(channels, kernels):
            stages.append(nn.Sequential(
                ResBlock(in_ch, ch, kernel=k, dropout=dropout),
                nn.MaxPool1d(3),
            ))
            in_ch = ch
        self.stages = nn.ModuleList(stages)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        head_dim = 2 * channels[-1]  # avg + max pool concat
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(head_dim),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(head_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            if self.use_checkpointing and self.training:
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(x)
