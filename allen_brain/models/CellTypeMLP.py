
from torch import nn


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


class MLPBlock(nn.Module):
    """Simple MLP block: Linear -> ReLU -> Dropout."""

    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(x)


class MLP_Model(nn.Module):
    """MLP with variable depth, SE attention on first layer."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.3,
                 n_layers: int = 2, hidden_dim: int = 512):
        super().__init__()
        # Build layer dimensions: halve each layer, min 64
        dims = [max(hidden_dim // (2 ** i), 64) for i in range(n_layers)]

        self.first = MLPBlock(input_dim, dims[0], dropout=dropout)
        self.se = MLP_SEBlock(dims[0])

        layers = []
        for i in range(1, n_layers):
            layers.append(MLPBlock(dims[i - 1], dims[i], dropout=dropout))
        self.hidden = nn.ModuleList(layers)

        self.classifier = nn.Linear(dims[-1], n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.first(x)
        x = self.se(x.unsqueeze(-1)).squeeze(-1)
        for layer in self.hidden:
            x = layer(x)
        return self.classifier(x)
