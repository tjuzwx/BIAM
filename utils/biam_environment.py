from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch


PAPER_ENVIRONMENT = {
    "os": "Ubuntu 20.04 LTS",
    "cpu": "Intel Xeon Platinum 8175M",
    "gpu": "NVIDIA RTX A6000",
    "gpu_memory_gb": 48,
    "system_memory_gb": 128,
    "cuda": "12.1",
    "pytorch": "2.1.0",
}


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _system_memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def collect_environment() -> dict[str, Any]:
    release = _os_release()
    gpu_name = None
    gpu_memory = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        gpu_memory = properties.total_memory / 1024**3
    return {
        "paper_required": PAPER_ENVIRONMENT,
        "actual": {
            "os": release.get("PRETTY_NAME", platform.platform()),
            "os_id": release.get("ID"),
            "os_version": release.get("VERSION_ID"),
            "cpu": _cpu_model(),
            "gpu": gpu_name,
            "gpu_memory_gb": gpu_memory,
            "system_memory_gb": _system_memory_gb(),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def validate_paper_environment(strict: bool = True) -> dict[str, Any]:
    report = collect_environment()
    actual = report["actual"]
    torch_version = str(actual["pytorch"]).split("+", 1)[0]
    checks = {
        "Ubuntu 20.04 LTS": actual["os_id"] == "ubuntu" and actual["os_version"] == "20.04",
        "Intel Xeon Platinum 8175M": "8175M" in str(actual["cpu"]),
        "NVIDIA RTX A6000": "RTX A6000" in str(actual["gpu"]),
        "GPU 48GB": actual["gpu_memory_gb"] is not None
        and 47.0 <= actual["gpu_memory_gb"] <= 49.0,
        "内存 128GB": actual["system_memory_gb"] is not None
        and 120.0 <= actual["system_memory_gb"] <= 136.0,
        "CUDA 12.1": actual["cuda"] == "12.1",
        "PyTorch 2.1.0": torch_version == "2.1.0",
    }
    report["checks"] = checks
    report["matched"] = all(checks.values())
    report["strict_validation"] = strict
    report["formal_environment_validated"] = strict and report["matched"]
    if strict and not report["matched"]:
        failures = [name for name, passed in checks.items() if not passed]
        details = "、".join(failures)
        raise RuntimeError(
            f"当前环境不符合大论文表 3-2：{details}。"
            "正式实验必须在指定环境运行；仅做兼容性检查时可使用 --allow-environment-mismatch。"
        )
    return report
