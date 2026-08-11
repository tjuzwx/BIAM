from __future__ import annotations

import numpy as np

from models.biam_additive_model import BIAMFeatureBuilder


class BIAMBinarizer:
    """兼容接口：使用论文的中心化 Hinge 设计替代旧二值分箱。"""

    def __init__(
        self,
        hinge_bins: int = 8,
        min_interaction_support: int = 20,
        label: str = "label",
        **_,
    ):
        self.label = label
        self.builder = BIAMFeatureBuilder(hinge_bins, min_interaction_support)

    def fit(self, X_train: np.ndarray) -> "BIAMBinarizer":
        self.builder.fit(X_train)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.builder.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def binarize_and_augment(self, train_df, test_df, **_):
        train_X = train_df.drop(columns=[self.label]).to_numpy(dtype=float)
        test_X = test_df.drop(columns=[self.label]).to_numpy(dtype=float)
        return (
            self.fit_transform(train_X),
            self.transform(test_X),
            train_df[self.label].to_numpy(),
            test_df[self.label].to_numpy(),
        )
