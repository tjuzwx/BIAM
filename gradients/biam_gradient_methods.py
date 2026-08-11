from __future__ import annotations

import torch


class BIAMGradientMethods:
    """论文离散结构梯度的独立计算工具。"""

    @staticmethod
    def antithetic_samples(
        probabilities: torch.Tensor, sample_count: int, generator=None
    ) -> torch.Tensor:
        if sample_count < 4 or sample_count % 2:
            raise ValueError("sample_count 必须是不小于 4 的偶数")
        uniform = torch.rand(
            sample_count // 2,
            len(probabilities),
            generator=generator,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        first = uniform <= probabilities
        second = (1.0 - uniform) <= probabilities
        return torch.stack([first, second], dim=1).reshape(sample_count, -1).float()

    @staticmethod
    def leave_pair_out_baseline(risks: torch.Tensor) -> torch.Tensor:
        if len(risks) < 4 or len(risks) % 2:
            raise ValueError("风险数量必须是不小于 4 的偶数")
        result = []
        for sample in range(len(risks)):
            partner = sample + 1 if sample % 2 == 0 else sample - 1
            keep = torch.ones(len(risks), dtype=torch.bool, device=risks.device)
            keep[sample] = False
            keep[partner] = False
            result.append(risks[keep].mean())
        return torch.stack(result)

    @classmethod
    def score_gradient(
        cls,
        structures: torch.Tensor,
        probabilities: torch.Tensor,
        risks: torch.Tensor,
    ) -> torch.Tensor:
        baseline = cls.leave_pair_out_baseline(risks)
        return ((risks - baseline)[:, None] * (structures - probabilities)).mean(dim=0)
