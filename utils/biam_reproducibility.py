from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """统一控制 Python、NumPy 与 PyTorch 随机状态。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def child_seed(seed: int, *parts: int) -> int:
    sequence = np.random.SeedSequence([seed, *parts])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
