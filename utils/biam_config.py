from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class BIAMConfig:
    """BIAM 论文复现实验配置。"""

    def __init__(self, args: Any = None, **overrides: Any):
        self.task = "regression"
        self.dataset = "synthetic"
        self.data_path = None
        self.target_key = "y"
        self.input_dim = 50
        self.num_classes = 2
        self.n_samples = 1000

        self.missing_mechanism = "MAR"
        self.missing_ratio = 0.3
        self.noise_ratio = 0.3
        self.noise_type = "gaussian"
        self.noise_scale = 0.3
        self.imbalance_ratio = 0.1
        self.correlation = 0.5
        self.standardize_target = False

        self.hinge_bins = 8
        self.structure_samples = 8
        self.inner_steps = 5
        self.min_interaction_support = 20
        self.lambda_l0 = 5e-4
        self.lambda_l2 = 1e-4
        self.lambda_kl = 1e-4
        self.prior_probability = 0.1
        self.initial_gate_probability = 0.5
        self.gate_threshold = 0.5
        self.hidden_dim_weighting = 10

        self.lower_lr = 1e-2
        self.structure_lr = 1e-2
        self.weight_lr = 1e-3
        self.batch_size = 64
        self.meta_batch_size = 64
        self.epochs = 200
        self.patience = 20
        self.eval_interval = 1
        self.tune_refit_steps = 10
        self.final_refit_steps = 100
        self.gradient_clip = 10.0

        self.seeds = [11, 22, 33, 44, 55]
        self.regression_outer_folds = 5
        self.regression_strata = 10
        self.output_dir = "results/biam"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.deterministic = True
        self.strict_environment = True

        if args is not None:
            values = vars(args) if hasattr(args, "__dict__") else dict(args)
            self._apply(values)
        self._apply(overrides)
        self.validate()

    @classmethod
    def from_yaml(cls, path: str | Path, args: Any = None) -> "BIAMConfig":
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        config = cls(**values)
        if args is not None:
            incoming = vars(args) if hasattr(args, "__dict__") else dict(args)
            config._apply(incoming)
            config.validate()
        return config

    def _apply(self, values: dict[str, Any]) -> None:
        aliases = {
            "upper_lr": "structure_lr",
            "penalty_coef": "lambda_l2",
            "spline_dim_regression": "hinge_bins",
            "spline_dim_classification": "hinge_bins",
        }
        for key, value in values.items():
            if value is None:
                continue
            target = aliases.get(key, key)
            if hasattr(self, target):
                if target == "device" and not isinstance(value, torch.device):
                    value = torch.device(value)
                setattr(self, target, value)

    def update(self, **kwargs: Any) -> None:
        unknown = [key for key in kwargs if not hasattr(self, key)]
        if unknown:
            raise ValueError(f"未知配置项: {', '.join(unknown)}")
        self._apply(kwargs)
        self.validate()

    @property
    def upper_lr(self) -> float:
        return self.structure_lr

    @upper_lr.setter
    def upper_lr(self, value: float) -> None:
        self.structure_lr = value

    @property
    def penalty_coef(self) -> float:
        return self.lambda_l2

    @penalty_coef.setter
    def penalty_coef(self, value: float) -> None:
        self.lambda_l2 = value

    def get_spline_dim(self) -> int:
        return self.hinge_bins

    def get_output_dim(self) -> int:
        if self.task == "regression" or self.num_classes == 2:
            return 1
        return self.num_classes - 1

    def validate(self) -> None:
        if self.task not in {"regression", "classification"}:
            raise ValueError("task 必须是 regression 或 classification")
        if self.dataset not in {"synthetic", "npz"}:
            raise ValueError("dataset 必须是 synthetic 或 npz")
        if self.dataset == "npz" and not self.data_path:
            raise ValueError("dataset=npz 时必须提供 data_path")
        if self.missing_mechanism.upper() not in {"MCAR", "MAR", "MNAR", "NONE"}:
            raise ValueError("missing_mechanism 必须是 NONE、MCAR、MAR 或 MNAR")
        if not 0 <= self.missing_ratio < 1:
            raise ValueError("missing_ratio 必须位于 [0, 1)")
        if not 0 <= self.noise_ratio < 1:
            raise ValueError("noise_ratio 必须位于 [0, 1)")
        if not 0 < self.imbalance_ratio <= 1:
            raise ValueError("imbalance_ratio 必须位于 (0, 1]")
        if self.structure_samples < 4 or self.structure_samples % 2:
            raise ValueError("structure_samples 必须是不小于 4 的偶数")
        if self.inner_steps < 1 or self.hinge_bins < 1:
            raise ValueError("inner_steps 和 hinge_bins 必须为正整数")
        if not 0 < self.prior_probability < 0.5:
            raise ValueError("prior_probability 必须位于 (0, 0.5)")
        if not 0 < self.gate_threshold < 1:
            raise ValueError("gate_threshold 必须位于 (0, 1)")
        if not self.seeds:
            raise ValueError("至少需要一个随机种子")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.device):
                result[key] = str(value)
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"BIAMConfig({self.to_dict()})"
