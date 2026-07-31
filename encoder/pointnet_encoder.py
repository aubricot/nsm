"""Permutation-invariant PointNet encoder: N x 3 surface points -> DeepSDF latent."""

import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    """1x1 convolutions applied identically to every point."""

    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Conv1d(dims[i], dims[i + 1], 1),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PointNetEncoder(nn.Module):
    def __init__(self, latent_size=512, point_feat_dims=(3, 64, 128, 256, 512),
                 head_dims=(512, 512, 512), dropout=0.1):
        super().__init__()
        self.point_mlp = SharedMLP(point_feat_dims)
        global_dim = point_feat_dims[-1]

        head = []
        prev = global_dim
        for h in head_dims:
            head += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        head += [nn.Linear(prev, latent_size)]
        self.head = nn.Sequential(*head)

    def forward(self, pts):
        """pts: (B, N, 3) -> (B, latent_size)."""
        x = pts.transpose(1, 2)
        x = self.point_mlp(x)
        x = torch.max(x, dim=2).values   # symmetric pool over points
        return self.head(x)
