from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.biam_additive_model import BIAMAdditiveModel
from models.biam_weighting_network import BIAMWeightingNetwork


class BIAMModel(nn.Module):
    """BIAM 推理模型；权重网络只在训练阶段使用。"""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = torch.device(device or config.device)
        self.task = config.task
        self.additive_model = BIAMAdditiveModel(config, self.device)
        self.weighting_network = BIAMWeightingNetwork(config, self.device)
        self.to(self.device)

    def forward(
        self,
        X: np.ndarray | torch.Tensor,
        return_weights: bool = False,
        targets: torch.Tensor | None = None,
    ):
        predictions = self.additive_model(X)
        if not return_weights:
            return predictions
        if targets is None:
            raise ValueError("返回样本权重时必须提供 targets")
        if self.task == "regression":
            losses = F.mse_loss(predictions.squeeze(1), targets.float(), reduction="none")
        else:
            losses = F.cross_entropy(predictions, targets.long(), reduction="none")
        return predictions, self.weighting_network(losses.detach())

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self.task != "classification":
            raise ValueError("回归模型没有类别概率")
        return torch.softmax(self(X), dim=1)

    def get_feature_importance(self) -> np.ndarray:
        return self.additive_model.get_feature_importance()

    def get_missing_indicators(self) -> np.ndarray:
        return self.additive_model.get_missing_indicators()
