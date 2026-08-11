from __future__ import annotations

import torch
import torch.nn as nn


class BIAMWeightingNetwork(nn.Module):
    """将逐样本损失映射到 (0, 1) 内的训练权重。"""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = torch.device(device or config.device)
        self.input_dim = 1
        self.hidden_dim = config.hidden_dim_weighting
        self.output_dim = 1
        self.network = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        self.to(self.device)

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        if losses.ndim == 1:
            losses = losses[:, None]
        return self.network(losses)

    def get_weight_statistics(self, losses: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            weights = self(losses)
            return {
                "mean_weight": float(weights.mean()),
                "std_weight": float(weights.std(unbiased=False)),
                "min_weight": float(weights.min()),
                "max_weight": float(weights.max()),
            }
