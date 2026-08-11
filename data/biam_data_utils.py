from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class BIAMDataUtils:
    """BIAM 数据张量与划分索引辅助函数。"""

    @staticmethod
    def create_data_loader(X, y, batch_size=64, shuffle=True, seed=0):
        generator = torch.Generator().manual_seed(seed)
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=0,
        )

    @staticmethod
    def save_split_indices(path: str | Path, indices: dict[str, np.ndarray]) -> None:
        np.savez_compressed(Path(path), **indices)

    @staticmethod
    def load_split_indices(path: str | Path) -> dict[str, np.ndarray]:
        values = np.load(Path(path), allow_pickle=False)
        return {name: values[name] for name in values.files}
