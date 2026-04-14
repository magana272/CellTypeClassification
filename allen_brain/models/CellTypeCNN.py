import torch
import torch.nn as nn


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
    """Residual 1D-CNN with widening channels, SE attention, and dual-pool head."""

    def __init__(self, seq_len: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ResBlock(64, 96, kernel=7, dropout=dropout),
            nn.MaxPool1d(3),
        )
        self.stage2 = nn.Sequential(
            ResBlock(96, 160, kernel=5, dropout=dropout),
            ResBlock(160, 160, kernel=5, dropout=dropout),
            nn.MaxPool1d(3),
        )
        self.stage3 = nn.Sequential(
            ResBlock(160, 256, kernel=3, dropout=dropout),
            ResBlock(256, 256, kernel=3, dropout=dropout),
            nn.MaxPool1d(3),
        )
        self.stage4 = ResBlock(256, 256, kernel=3, dropout=dropout)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(x)