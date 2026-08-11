from __future__ import annotations

from pathlib import Path

import numpy as np


class BIAMDatasetLoader:
    """加载不改变样本值、缺失状态或标签的外部 NPZ 数据。"""

    def __init__(self, config=None):
        self.config = config

    def load_npz(self, path: str | Path, target_key: str = "y"):
        values = np.load(Path(path), allow_pickle=False)
        if "X" not in values or target_key not in values:
            raise ValueError("NPZ 文件必须包含 X 和目标数组")
        X = np.asarray(values["X"], dtype=np.float64)
        y = np.asarray(values[target_key])
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X 必须为二维数组，目标必须为等长一维数组")
        return X, y

    @staticmethod
    def get_available_datasets():
        return ["synthetic", "npz"]
