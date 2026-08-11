from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from utils.biam_reproducibility import child_seed


@dataclass
class BIAMDataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_meta: np.ndarray
    y_meta: np.ndarray
    X_tune: np.ndarray
    y_tune: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    indices: dict[str, np.ndarray]
    y_train_clean: np.ndarray
    artifacts: dict[str, Any] = field(default_factory=dict)
    fold: int = 0


class BIAMDataGenerator:
    """按论文协议生成四路数据及可复用索引。"""

    def __init__(self, config):
        self.config = config
        self.task = config.task
        self.dataset = config.dataset
        self.missing_ratio = config.missing_ratio
        self.noise_ratio = config.noise_ratio
        self.imbalance_ratio = config.imbalance_ratio
        self.batch_size = config.batch_size

    def generate_runs(self, seed: int) -> list[BIAMDataBundle]:
        if self.dataset == "synthetic":
            return [self._generate_synthetic(seed)]
        X, y = self._load_npz()
        if self.task == "regression":
            return self._generate_regression_folds(X, y, seed)
        indices = self._four_way_split(y, seed, "classification")
        return [self._build_bundle(X, y, indices, seed, fold=0)]

    def generate_data(self, seed: int | None = None) -> BIAMDataBundle:
        actual_seed = self.config.seeds[0] if seed is None else seed
        return self.generate_runs(actual_seed)[0]

    def _load_npz(self) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(Path(self.config.data_path), allow_pickle=False)
        if "X" not in data or self.config.target_key not in data:
            raise ValueError("NPZ 文件必须包含 X 和目标数组")
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data[self.config.target_key])
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X 必须为二维数组，目标必须为等长一维数组")
        if self.task == "classification":
            _, y = np.unique(y, return_inverse=True)
            self.config.num_classes = int(np.max(y)) + 1
        return X, y

    def _generate_synthetic(self, seed: int) -> BIAMDataBundle:
        p = self.config.input_dim
        if p < 6:
            raise ValueError("论文仿真公式至少需要 6 个特征")
        rng = np.random.default_rng(seed)
        grid = np.arange(p)
        covariance = self.config.correlation ** np.abs(grid[:, None] - grid[None, :])
        X = rng.multivariate_normal(np.zeros(p), covariance, size=self.config.n_samples)
        provisional = self._clean_response(X, np.zeros_like(X, dtype=bool))
        indices = self._four_way_split(provisional, seed, self.task)
        return self._build_bundle(X, provisional, indices, seed, fold=0, synthetic=True)

    def _generate_regression_folds(
        self, X: np.ndarray, y: np.ndarray, seed: int
    ) -> list[BIAMDataBundle]:
        splitter = KFold(
            n_splits=self.config.regression_outer_folds,
            shuffle=True,
            random_state=seed,
        )
        bundles = []
        for fold, (non_test, test) in enumerate(splitter.split(X)):
            strata = self._regression_strata(y[non_test])
            train, remainder = train_test_split(
                non_test,
                train_size=7 / 9,
                random_state=child_seed(seed, fold, 1),
                stratify=strata,
            )
            remainder_strata = self._regression_strata(y[remainder])
            meta, tune = train_test_split(
                remainder,
                test_size=0.5,
                random_state=child_seed(seed, fold, 2),
                stratify=remainder_strata,
            )
            indices = {"train": train, "meta": meta, "tune": tune, "test": test}
            bundles.append(self._build_bundle(X, y, indices, seed, fold=fold))
        return bundles

    def _four_way_split(self, y: np.ndarray, seed: int, task: str) -> dict[str, np.ndarray]:
        all_indices = np.arange(len(y))
        strata = y if task == "classification" else self._regression_strata(y)
        expected_classes = set(np.unique(y)) if task == "classification" else None
        for attempt in range(100):
            split_seed = child_seed(seed, attempt)
            try:
                train, remainder = train_test_split(
                    all_indices, train_size=0.7, random_state=split_seed, stratify=strata
                )
                remainder_strata = strata[remainder]
                meta, rest = train_test_split(
                    remainder,
                    train_size=1 / 3,
                    random_state=child_seed(split_seed, 1),
                    stratify=remainder_strata,
                )
                rest_strata = strata[rest]
                tune, test = train_test_split(
                    rest,
                    test_size=0.5,
                    random_state=child_seed(split_seed, 2),
                    stratify=rest_strata,
                )
            except ValueError:
                continue
            result = {"train": train, "meta": meta, "tune": tune, "test": test}
            if expected_classes is None or all(
                set(np.unique(y[part])) == expected_classes for part in result.values()
            ):
                return result
        raise ValueError("无法生成每个子集均包含所有类别的 7:1:1:1 分层划分")

    def _regression_strata(self, y: np.ndarray) -> np.ndarray:
        bins = min(self.config.regression_strata, max(2, len(y) // 8))
        order = np.argsort(np.argsort(np.asarray(y), kind="mergesort"), kind="mergesort")
        return np.minimum(order * bins // len(y), bins - 1)

    def _build_bundle(
        self,
        X_complete: np.ndarray,
        y_base: np.ndarray,
        indices: dict[str, np.ndarray],
        seed: int,
        fold: int,
        synthetic: bool = False,
    ) -> BIAMDataBundle:
        final_indices = {name: np.asarray(value, dtype=np.int64) for name, value in indices.items()}
        train_indices = final_indices["train"]
        if self.task == "classification" and self.imbalance_ratio < 1:
            keep = self._long_tail_indices(np.asarray(y_base)[train_indices], child_seed(seed, fold, 20))
            train_indices = train_indices[keep]
            final_indices["train"] = train_indices

        X_missing, missing_artifacts = self._apply_missingness(
            X_complete, final_indices, child_seed(seed, fold, 10)
        )
        if synthetic:
            y_clean = self._clean_response(X_complete, np.isnan(X_missing))
        else:
            y_clean = np.asarray(y_base).copy()

        target_artifacts = {}
        if self.task == "regression" and self.config.standardize_target:
            target_mean = float(np.mean(y_clean[train_indices]))
            target_std = float(np.std(y_clean[train_indices]))
            if not np.isfinite(target_std) or target_std < 1e-12:
                target_std = 1.0
            y_clean = (y_clean - target_mean) / target_std
            target_artifacts = {
                "target_mean": target_mean,
                "target_std": target_std,
            }

        y_train_clean = y_clean[train_indices].copy()
        if self.task == "classification":
            boundary_scores = (
                self._classification_score(X_complete, np.isnan(X_missing))[train_indices]
                if synthetic
                else None
            )
            y_train, corruption = self._corrupt_labels(
                y_train_clean, child_seed(seed, fold, 30), boundary_scores
            )
            if boundary_scores is not None:
                corruption["classification_boundary_scores"] = boundary_scores
        else:
            y_train, corruption = self._corrupt_regression(
                y_train_clean, child_seed(seed, fold, 30)
            )

        if "label_noise_local_indices" in corruption:
            local = corruption["label_noise_local_indices"]
            corruption["label_noise_global_indices"] = train_indices[local]
        if "response_outlier_local_indices" in corruption:
            local = corruption["response_outlier_local_indices"]
            corruption["response_outlier_global_indices"] = train_indices[local]

        artifacts = {
            **missing_artifacts,
            **corruption,
            **target_artifacts,
            "seed": seed,
            "fold": fold,
        }
        return BIAMDataBundle(
            X_train=X_missing[train_indices],
            y_train=y_train,
            X_meta=X_missing[final_indices["meta"]],
            y_meta=y_clean[final_indices["meta"]],
            X_tune=X_missing[final_indices["tune"]],
            y_tune=y_clean[final_indices["tune"]],
            X_test=X_missing[final_indices["test"]],
            y_test=y_clean[final_indices["test"]],
            indices=final_indices,
            y_train_clean=y_train_clean,
            artifacts=artifacts,
            fold=fold,
        )

    def _clean_response(self, X: np.ndarray, missing: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return (
                2.0 * np.sin(np.pi * X[:, 0])
                + 1.5 * (X[:, 1] ** 2 - 1.0)
                + 1.2 * X[:, 2]
                + X[:, 3] * X[:, 4]
                + 0.8 * missing[:, 0]
                + 0.6 * missing[:, 1] * X[:, 2]
            )
        score = self._classification_score(X, missing)
        return (score >= 0).astype(np.int64)

    @staticmethod
    def _classification_score(X: np.ndarray, missing: np.ndarray) -> np.ndarray:
        return (
            1.5 * X[:, 0]
            + 1.2 * X[:, 1] ** 2
            - X[:, 2] ** 3
            + np.sin(np.pi * X[:, 3])
            + 0.8 * X[:, 4] * X[:, 5]
            + 0.7 * missing[:, 0]
            + 0.5 * missing[:, 1] * X[:, 2]
        )

    def _apply_missingness(
        self, X: np.ndarray, indices: dict[str, np.ndarray], seed: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        mechanism = self.config.missing_mechanism.upper()
        natural = np.isnan(X)
        if mechanism == "NONE" or self.missing_ratio == 0:
            return X.copy(), {
                "artificial_missing_mask": np.zeros_like(natural),
                "candidate_features": np.array([], dtype=np.int64),
            }

        train = indices["train"]
        p = X.shape[1]
        candidate_count = max(1, p // 2) if self.dataset == "synthetic" else max(1, int(np.ceil(0.3 * p)))
        rng = np.random.default_rng(seed)
        if self.dataset == "synthetic":
            candidates = np.arange(candidate_count, dtype=np.int64)
        else:
            candidates = np.sort(rng.choice(p, candidate_count, replace=False))
        outside = np.setdiff1d(np.arange(p), candidates)
        if len(outside) == 0 and mechanism == "MAR":
            raise ValueError("MAR 至少需要一个不会被人工遮蔽的锚点特征")

        means = np.nanmean(X[train], axis=0)
        stds = np.nanstd(X[train], axis=0)
        stds[~np.isfinite(stds) | (stds < 1e-12)] = 1.0
        standardized = (X - means) / stds
        standardized[~np.isfinite(standardized)] = 0.0
        anchors = rng.choice(outside, len(candidates), replace=True) if len(outside) else np.full(len(candidates), -1)
        directions = rng.choice(np.array([-1.0, 1.0]), len(candidates))
        intercepts = np.zeros(len(candidates), dtype=np.float64)
        artificial = np.zeros_like(natural)

        for position, feature in enumerate(candidates):
            if mechanism == "MCAR":
                probabilities = np.full(len(X), self.missing_ratio)
            elif mechanism == "MAR":
                signal = 1.5 * standardized[:, anchors[position]]
                intercepts[position] = self._calibrate_intercept(signal[train])
                probabilities = self._sigmoid(intercepts[position] + signal)
            else:
                signal = 1.5 * directions[position] * standardized[:, feature]
                intercepts[position] = self._calibrate_intercept(signal[train])
                probabilities = self._sigmoid(intercepts[position] + signal)
            artificial[:, feature] = rng.random(len(X)) < probabilities

        artificial &= ~natural
        X_missing = X.copy()
        X_missing[artificial] = np.nan
        return X_missing, {
            "artificial_missing_mask": artificial,
            "natural_missing_mask": natural,
            "candidate_features": candidates,
            "mar_anchors": anchors,
            "mnar_directions": directions,
            "missing_intercepts": intercepts,
        }

    def _calibrate_intercept(self, signal: np.ndarray) -> float:
        low, high = -30.0, 30.0
        for _ in range(80):
            middle = (low + high) / 2
            if self._sigmoid(middle + signal).mean() < self.missing_ratio:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-value))

    def _long_tail_indices(self, y: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        classes, counts = np.unique(y, return_counts=True)
        order = np.argsort(-counts)
        classes, counts = classes[order], counts[order]
        kappa = 1.0 / self.imbalance_ratio
        selected = []
        for rank, (label, count) in enumerate(zip(classes, counts)):
            target = int(np.floor(counts[0] * kappa ** (-rank / max(1, len(classes) - 1))))
            class_indices = np.flatnonzero(y == label)
            selected.append(rng.choice(class_indices, min(count, max(1, target)), replace=False))
        result = np.concatenate(selected)
        rng.shuffle(result)
        return result

    def _corrupt_labels(
        self,
        y: np.ndarray,
        seed: int,
        boundary_scores: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        result = y.copy()
        count = int(np.floor(len(y) * self.noise_ratio))
        if not count:
            noisy = np.array([], dtype=np.int64)
        elif boundary_scores is None:
            noisy = rng.choice(len(y), count, replace=False)
        else:
            tie_breaker = rng.random(len(y))
            noisy = np.lexsort((tie_breaker, np.abs(boundary_scores)))[:count]
        classes = np.unique(y)
        for index in noisy:
            alternatives = classes[classes != result[index]]
            result[index] = rng.choice(alternatives)
        return result, {"label_noise_local_indices": noisy}

    def _corrupt_regression(self, y: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        scale = self.config.noise_scale
        kind = self.config.noise_type.lower()
        if kind == "student_t":
            noise = scale * rng.standard_t(3, len(y)) / np.sqrt(3.0)
        elif kind == "chi_square":
            noise = scale * (rng.chisquare(3, len(y)) - 3.0) / np.sqrt(6.0)
        else:
            noise = rng.normal(0.0, scale, len(y))
        outliers = np.flatnonzero(rng.random(len(y)) < self.noise_ratio)
        contamination = np.zeros(len(y))
        if len(outliers):
            contamination[outliers] = rng.normal(0.0, 5.0 * scale, len(outliers))
        return y + noise + contamination, {
            "response_outlier_local_indices": outliers,
            "base_noise": noise,
            "response_contamination": contamination,
        }
