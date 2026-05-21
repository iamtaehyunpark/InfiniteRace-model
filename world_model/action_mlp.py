"""
Action MLP — encodes the 3D action vector into AdaLN scale/shift
parameters for each transformer layer.

The final linear layer is **zero-initialized** so that action
conditioning has no effect at the start of training, preventing
the action signal from destabilising early reconstruction learning.
"""
import torch.nn as nn
from torch import Tensor


class ActionMLP(nn.Module):
    """
    Maps normalised action vector to per-layer AdaLN (scale, shift) pairs.

    Input:  (B, 3) float32 [-1, 1]  — [speed_norm, delta_heading_norm, steer]
    Output: list of (scale, shift) tuples, one per transformer layer.
            Each scale/shift: (B, d_model) float32
    """

    def __init__(self, d_model: int = 512, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        out_dim = n_layers * 2 * d_model   # 2 per layer: scale + shift

        self.mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, out_dim),
        )

        # Zero-init final layer → no action effect at start of training
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, action: Tensor) -> list[tuple[Tensor, Tensor]]:
        """
        Args:
            action: (B, 3)

        Returns:
            List of ``n_layers`` (scale, shift) tuples.
            Each tensor: (B, d_model).
        """
        x = self.mlp(action)                             # (B, n_layers * 2 * d_model)
        x = x.view(-1, self.n_layers, 2, self.d_model)   # (B, L, 2, D)

        params: list[tuple[Tensor, Tensor]] = []
        for i in range(self.n_layers):
            scale = x[:, i, 0, :]                         # (B, D)
            shift = x[:, i, 1, :]                         # (B, D)
            params.append((scale, shift))

        return params
