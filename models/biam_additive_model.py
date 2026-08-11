from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class BIAMGroup:
    name: str
    kind: str
    source: int
    target: int | None
    start: int
    stop: int


class BIAMFeatureBuilder:
    """仅用训练集拟合 Hinge 基、中心化常数和候选交互。"""

    def __init__(self, hinge_bins: int, min_support: int):
        self.hinge_bins = hinge_bins
        self.min_support = min_support
        self.fitted = False

    def fit(self, X_train: np.ndarray) -> "BIAMFeatureBuilder":
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X_train 必须是二维数组")
        self.input_dim = X.shape[1]
        missing = np.isnan(X)
        observed = ~missing

        self.medians = np.nanmedian(X, axis=0)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.medians[~np.isfinite(self.medians)] = 0.0
        self.means[~np.isfinite(self.means)] = self.medians[~np.isfinite(self.means)]
        self.stds[~np.isfinite(self.stds) | (self.stds < 1e-12)] = 1.0
        self.missing_rates = missing.mean(axis=0)

        closed = np.where(missing, self.medians, X)
        standardized = (closed - self.means) / self.stds
        probabilities = np.arange(1, self.hinge_bins + 1) / (self.hinge_bins + 1)
        self.knots = np.zeros((self.input_dim, self.hinge_bins), dtype=np.float64)
        for feature in range(self.input_dim):
            values = standardized[observed[:, feature], feature]
            self.knots[feature] = np.quantile(values, probabilities) if len(values) else 0.0

        basis = self._hinge_basis(standardized)
        self.main_centers = np.zeros((self.input_dim, 2 * self.hinge_bins), dtype=np.float64)
        for feature in range(self.input_dim):
            support = observed[:, feature]
            if support.any():
                self.main_centers[feature] = basis[support, feature].mean(axis=0)

        pairs = []
        centers = []
        supports = []
        for missing_feature in range(self.input_dim):
            for observed_feature in range(self.input_dim):
                if missing_feature == observed_feature:
                    continue
                support = missing[:, missing_feature] & observed[:, observed_feature]
                count = int(support.sum())
                if count >= self.min_support:
                    pairs.append((missing_feature, observed_feature))
                    centers.append(basis[support, observed_feature].mean(axis=0))
                    supports.append(count)
        self.interactions = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self.interaction_centers = np.asarray(centers, dtype=np.float64).reshape(
            -1, 2 * self.hinge_bins
        )
        self.interaction_supports = np.asarray(supports, dtype=np.int64)
        self.groups = self._make_groups()
        self.design_dim = self.groups[-1].stop if self.groups else 1
        self.fitted = True
        return self

    def _make_groups(self) -> list[BIAMGroup]:
        groups = []
        cursor = 1
        width = 2 * self.hinge_bins
        for feature in range(self.input_dim):
            groups.append(BIAMGroup(f"O_{feature}", "obs", feature, None, cursor, cursor + width))
            cursor += width
        for feature in range(self.input_dim):
            groups.append(BIAMGroup(f"M_{feature}", "miss", feature, None, cursor, cursor + 1))
            cursor += 1
        for missing_feature, observed_feature in self.interactions:
            groups.append(
                BIAMGroup(
                    f"I_{missing_feature}_{observed_feature}",
                    "int",
                    int(missing_feature),
                    int(observed_feature),
                    cursor,
                    cursor + width,
                )
            )
            cursor += width
        return groups

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("必须先使用训练集拟合特征构造器")
        X = np.asarray(X, dtype=np.float64)
        missing = np.isnan(X)
        observed = ~missing
        closed = np.where(missing, self.medians, X)
        standardized = (closed - self.means) / self.stds
        basis = self._hinge_basis(standardized)
        blocks = [np.ones((len(X), 1), dtype=np.float64)]
        for feature in range(self.input_dim):
            centered = basis[:, feature] - self.main_centers[feature]
            blocks.append(observed[:, feature, None] * centered)
        blocks.append(missing.astype(np.float64) - self.missing_rates)
        for position, (missing_feature, observed_feature) in enumerate(self.interactions):
            support = missing[:, missing_feature] & observed[:, observed_feature]
            centered = basis[:, observed_feature] - self.interaction_centers[position]
            blocks.append(support[:, None] * centered)
        return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    def _hinge_basis(self, standardized: np.ndarray) -> np.ndarray:
        delta = standardized[:, :, None] - self.knots[None, :, :]
        return np.concatenate([np.maximum(delta, 0.0), np.maximum(-delta, 0.0)], axis=2)

    def active_columns(self, gates: np.ndarray | torch.Tensor) -> np.ndarray:
        values = gates.detach().cpu().numpy() if isinstance(gates, torch.Tensor) else np.asarray(gates)
        columns = [np.array([0], dtype=np.int64)]
        for active, group in zip(values.astype(bool), self.groups):
            if active:
                columns.append(np.arange(group.start, group.stop, dtype=np.int64))
        return np.concatenate(columns)

    def metadata(self) -> dict:
        return {
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "missing_rates": self.missing_rates,
            "knots": self.knots,
            "main_centers": self.main_centers,
            "interactions": self.interactions,
            "interaction_centers": self.interaction_centers,
            "interaction_supports": self.interaction_supports,
        }


class BIAMAdditiveModel(nn.Module):
    """观测主效应、缺失主效应和有向缺失交互的可加模型。"""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = torch.device(device or config.device)
        self.task = config.task
        self.input_dim = config.input_dim
        self.output_dim = config.get_output_dim()
        self.spline_dim = config.hinge_bins
        self.builder = BIAMFeatureBuilder(config.hinge_bins, config.min_interaction_support)
        self.register_buffer("full_coefficients", torch.empty(0, self.output_dim))
        self.register_buffer("hard_gates", torch.empty(0, dtype=torch.bool))
        self.to(self.device)

    def fit_preprocessor(self, X_train: np.ndarray) -> None:
        self.builder.fit(X_train)
        self.input_dim = self.builder.input_dim
        self.full_coefficients = torch.zeros(
            self.builder.design_dim, self.output_dim, device=self.device
        )
        self.hard_gates = torch.ones(len(self.builder.groups), dtype=torch.bool, device=self.device)

    def design(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            values = X.detach().cpu().numpy()
        else:
            values = X
        return torch.as_tensor(self.builder.transform(values), device=self.device)

    def active_columns(self, gates: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.builder.active_columns(gates), dtype=torch.long, device=self.device)

    def linear_from_design(
        self, design: torch.Tensor, theta: torch.Tensor, active_columns: torch.Tensor
    ) -> torch.Tensor:
        return design.index_select(1, active_columns) @ theta

    def set_final_state(self, gates: torch.Tensor, theta: torch.Tensor) -> None:
        columns = self.active_columns(gates)
        coefficients = torch.zeros(
            self.builder.design_dim, self.output_dim, device=self.device, dtype=theta.dtype
        )
        coefficients.index_copy_(0, columns, theta.detach())
        self.full_coefficients = coefficients
        self.hard_gates = gates.detach().to(self.device, dtype=torch.bool)

    def linear_predictor(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        return self.design(X) @ self.full_coefficients

    def forward(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        eta = self.linear_predictor(X)
        if self.task == "regression":
            return eta
        if self.config.num_classes == 2:
            return torch.cat([torch.zeros_like(eta), eta], dim=1)
        reference = torch.zeros((len(eta), 1), device=eta.device, dtype=eta.dtype)
        return torch.cat([eta, reference], dim=1)

    def get_feature_importance(self) -> np.ndarray:
        importance = np.zeros(self.input_dim, dtype=np.float64)
        for group, active in zip(self.builder.groups, self.hard_gates.detach().cpu().numpy()):
            if group.kind == "obs" and active:
                importance[group.source] = torch.linalg.vector_norm(
                    self.full_coefficients[group.start : group.stop]
                ).item()
        return importance

    def get_missing_indicators(self) -> np.ndarray:
        values = np.zeros((self.input_dim, self.output_dim), dtype=np.float32)
        for group, active in zip(self.builder.groups, self.hard_gates.detach().cpu().numpy()):
            if group.kind == "miss" and active:
                values[group.source] = self.full_coefficients[group.start].detach().cpu().numpy()
        return values.squeeze()

    def selected_groups(self) -> list[str]:
        gates = self.hard_gates.detach().cpu().numpy()
        return [group.name for group, active in zip(self.builder.groups, gates) if active]

    def regularization(self, theta: torch.Tensor, active_columns: torch.Tensor) -> torch.Tensor:
        penalized = active_columns != 0
        return theta[penalized].square().sum()

    def group_probabilities(self, logits: torch.Tensor) -> dict[str, float]:
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        return {group.name: float(value) for group, value in zip(self.builder.groups, probabilities)}
