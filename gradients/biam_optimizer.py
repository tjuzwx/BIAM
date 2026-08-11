from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


class BIAMOptimizer:
    """论文算法 1 的反变量结构采样与截断双层更新。"""

    def __init__(self, config, biam_model, weighting_network=None):
        self.config = config
        self.biam_model = biam_model
        self.model = biam_model.additive_model
        self.weighting_network = weighting_network or biam_model.weighting_network
        self.biam_model.weighting_network = self.weighting_network
        self.device = torch.device(config.device)
        self.cache: dict[tuple[int, ...], torch.Tensor] = {}
        self.training_history: dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "tune_loss": [],
            "selected_group_ratio": [],
            "cache_size": [],
        }
        self.early_stopping: dict[str, Any] = {}
        self.structure_logits: torch.nn.Parameter | None = None
        self._rng: np.random.Generator | None = None

    def fit(self, bundle, seed: int) -> dict[str, Any]:
        self.cache.clear()
        for values in self.training_history.values():
            values.clear()
        self.early_stopping = {}
        self.model.fit_preprocessor(bundle.X_train)
        designs = {
            "train": self.model.design(bundle.X_train),
            "meta": self.model.design(bundle.X_meta),
            "tune": self.model.design(bundle.X_tune),
        }
        targets = {
            "train": self._target_tensor(bundle.y_train),
            "meta": self._target_tensor(bundle.y_meta),
            "tune": self._target_tensor(bundle.y_tune),
        }
        group_count = len(self.model.builder.groups)
        initial = np.log(
            self.config.initial_gate_probability / (1.0 - self.config.initial_gate_probability)
        )
        self.structure_logits = torch.nn.Parameter(
            torch.full((group_count,), float(initial), device=self.device)
        )
        structure_optimizer = torch.optim.SGD(
            [self.structure_logits], lr=self.config.structure_lr
        )
        weight_optimizer = torch.optim.SGD(
            self.weighting_network.parameters(), lr=self.config.weight_lr
        )
        self._rng = np.random.default_rng(seed)

        best = None
        stale_epochs = 0
        stopped_epoch = self.config.epochs
        steps_per_epoch = max(1, math.ceil(len(designs["train"]) / self.config.batch_size))

        for epoch in range(1, self.config.epochs + 1):
            losses = []
            for _ in range(steps_per_epoch):
                train_index = self._batch_indices(
                    len(designs["train"]), self.config.batch_size
                )
                meta_index = self._batch_indices(
                    len(designs["meta"]), self.config.meta_batch_size
                )
                loss = self._joint_step(
                    designs["train"].index_select(0, train_index),
                    targets["train"].index_select(0, train_index),
                    designs["meta"].index_select(0, meta_index),
                    targets["meta"].index_select(0, meta_index),
                    structure_optimizer,
                    weight_optimizer,
                )
                losses.append(loss)

            if epoch % self.config.eval_interval:
                continue
            gates = self._hard_gates()
            theta = self._temporary_refit(
                gates,
                designs["train"],
                targets["train"],
                self.config.tune_refit_steps,
            )
            tune_loss = self._mean_loss(
                self.model.linear_from_design(
                    designs["tune"], theta, self.model.active_columns(gates)
                ),
                targets["tune"],
            ).item()
            selected_ratio = float(gates.float().mean()) if len(gates) else 0.0
            self.training_history["epoch"].append(epoch)
            self.training_history["train_loss"].append(float(np.mean(losses)))
            self.training_history["tune_loss"].append(tune_loss)
            self.training_history["selected_group_ratio"].append(selected_ratio)
            self.training_history["cache_size"].append(len(self.cache))

            if best is None or tune_loss < best["tune_loss"] - 1e-12:
                best = {
                    "epoch": epoch,
                    "tune_loss": tune_loss,
                    "structure_logits": self.structure_logits.detach().clone(),
                    "weighting_network": copy.deepcopy(self.weighting_network.state_dict()),
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.config.patience:
                stopped_epoch = epoch
                break

        if best is None:
            raise RuntimeError("训练过程中没有产生可用的调参集状态")
        with torch.no_grad():
            self.structure_logits.copy_(best["structure_logits"])
        self.weighting_network.load_state_dict(best["weighting_network"])
        final_gates = self._hard_gates()
        final_design = torch.cat([designs["train"], designs["meta"]], dim=0)
        final_target = torch.cat([targets["train"], targets["meta"]], dim=0)
        final_theta = self._temporary_refit(
            final_gates,
            final_design,
            final_target,
            self.config.final_refit_steps,
        )
        self.model.set_final_state(final_gates, final_theta)
        self.early_stopping = {
            "best_epoch": best["epoch"],
            "stopped_epoch": stopped_epoch,
            "best_tune_loss": best["tune_loss"],
            "patience": self.config.patience,
        }
        return {
            "early_stopping": self.early_stopping,
            "history": self.training_history,
            "cache_size": len(self.cache),
            "candidate_groups": len(final_gates),
            "selected_groups": int(final_gates.sum().item()),
            "selected_group_names": self.model.selected_groups(),
            "group_probabilities": self.model.group_probabilities(self.structure_logits),
        }

    def _joint_step(
        self,
        train_design: torch.Tensor,
        train_target: torch.Tensor,
        meta_design: torch.Tensor,
        meta_target: torch.Tensor,
        structure_optimizer,
        weight_optimizer,
    ) -> float:
        probabilities = torch.sigmoid(self.structure_logits.detach())
        structures = self._antithetic_structures(probabilities)
        snapshot = dict(self.cache)
        records = []
        risks = []

        for gates in structures:
            key = tuple(int(value) for value in gates.tolist())
            columns = self.model.active_columns(gates)
            theta = snapshot.get(key)
            if theta is None:
                theta = torch.zeros(
                    len(columns), self.model.output_dim, device=self.device
                )
            else:
                theta = theta.detach().clone()

            for _ in range(self.config.inner_steps - 1):
                theta = self._ordinary_inner_update(
                    theta, columns, train_design, train_target
                )
            theta_bar = theta.detach().requires_grad_(True)
            train_eta = self.model.linear_from_design(train_design, theta_bar, columns)
            train_losses = self._sample_losses(train_eta, train_target)
            weights = self.weighting_network(train_losses.detach()).squeeze(1)
            objective = (weights * train_losses).mean()
            objective = objective + self.config.lambda_l2 * self.model.regularization(
                theta_bar, columns
            )
            gradient = torch.autograd.grad(objective, theta_bar, create_graph=True)[0]
            gradient = self._clip_gradient(gradient)
            virtual_theta = theta_bar - self.config.lower_lr * gradient
            meta_eta = self.model.linear_from_design(meta_design, virtual_theta, columns)
            risk = self._mean_loss(meta_eta, meta_target)
            risks.append(risk)
            records.append((key, gates, columns, theta_bar.detach()))

        detached_risks = torch.stack([risk.detach() for risk in risks])
        baselines = self._leave_pair_out_baselines(detached_risks)
        centered = detached_risks - baselines
        score = (centered[:, None] * (structures - probabilities)).mean(dim=0)
        pi0 = self.config.prior_probability
        bounded = probabilities.clamp(1e-7, 1.0 - 1e-7)
        sparsity = self.config.lambda_l0 * bounded * (1.0 - bounded)
        kl = (
            self.config.lambda_kl
            * bounded
            * (1.0 - bounded)
            * torch.log(bounded * (1.0 - pi0) / (pi0 * (1.0 - bounded)))
        )
        structure_optimizer.zero_grad()
        self.structure_logits.grad = score + sparsity + kl

        weight_optimizer.zero_grad()
        torch.stack(risks).mean().backward()
        torch.nn.utils.clip_grad_norm_(
            self.weighting_network.parameters(), self.config.gradient_clip
        )
        structure_optimizer.step()
        weight_optimizer.step()

        pending: dict[tuple[int, ...], list[torch.Tensor]] = defaultdict(list)
        actual_losses = []
        for key, _, columns, theta_bar in records:
            theta_actual, loss_value = self._actual_inner_update(
                theta_bar, columns, train_design, train_target
            )
            pending[key].append(theta_actual)
            actual_losses.append(loss_value)
        for key, states in pending.items():
            self.cache[key] = torch.stack(states).mean(dim=0).detach()
        return float(np.mean(actual_losses))

    def _ordinary_inner_update(
        self,
        theta: torch.Tensor,
        columns: torch.Tensor,
        design: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        theta = theta.detach().requires_grad_(True)
        eta = self.model.linear_from_design(design, theta, columns)
        losses = self._sample_losses(eta, target)
        with torch.no_grad():
            weights = self.weighting_network(losses.detach()).squeeze(1)
        objective = (weights * losses).mean()
        objective = objective + self.config.lambda_l2 * self.model.regularization(theta, columns)
        gradient = torch.autograd.grad(objective, theta)[0]
        return (theta - self.config.lower_lr * self._clip_gradient(gradient)).detach()

    def _actual_inner_update(
        self,
        theta_bar: torch.Tensor,
        columns: torch.Tensor,
        design: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        theta = theta_bar.detach().requires_grad_(True)
        eta = self.model.linear_from_design(design, theta, columns)
        losses = self._sample_losses(eta, target)
        with torch.no_grad():
            weights = self.weighting_network(losses.detach()).squeeze(1)
        objective = (weights * losses).mean()
        objective = objective + self.config.lambda_l2 * self.model.regularization(theta, columns)
        gradient = torch.autograd.grad(objective, theta)[0]
        updated = theta - self.config.lower_lr * self._clip_gradient(gradient)
        return updated.detach(), float(losses.detach().mean())

    def _temporary_refit(
        self,
        gates: torch.Tensor,
        design: torch.Tensor,
        target: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        columns = self.model.active_columns(gates)
        key = tuple(int(value) for value in gates.tolist())
        cached = self.cache.get(key)
        theta = (
            cached.detach().clone()
            if cached is not None
            else torch.zeros(len(columns), self.model.output_dim, device=self.device)
        )
        for _ in range(steps):
            theta = self._ordinary_inner_update(theta, columns, design, target)
        return theta

    def _antithetic_structures(self, probabilities: torch.Tensor) -> torch.Tensor:
        half = self.config.structure_samples // 2
        uniform = torch.as_tensor(
            self._rng.random((half, len(probabilities))),
            dtype=probabilities.dtype,
            device=self.device,
        )
        first = uniform <= probabilities
        second = (1.0 - uniform) <= probabilities
        return torch.stack([first, second], dim=1).reshape(-1, len(probabilities)).float()

    @staticmethod
    def _leave_pair_out_baselines(risks: torch.Tensor) -> torch.Tensor:
        baselines = []
        for sample in range(len(risks)):
            partner = sample + 1 if sample % 2 == 0 else sample - 1
            keep = torch.ones(len(risks), dtype=torch.bool, device=risks.device)
            keep[sample] = False
            keep[partner] = False
            baselines.append(risks[keep].mean())
        return torch.stack(baselines)

    def _hard_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.structure_logits.detach()) >= self.config.gate_threshold

    def _batch_indices(self, size: int, batch_size: int) -> torch.Tensor:
        count = min(size, batch_size)
        values = self._rng.choice(size, count, replace=False)
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def _target_tensor(self, values: np.ndarray) -> torch.Tensor:
        dtype = torch.float32 if self.config.task == "regression" else torch.long
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def _sample_losses(self, eta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.task == "regression":
            return (eta.squeeze(1) - target.float()).square()
        if self.config.num_classes == 2:
            return F.binary_cross_entropy_with_logits(
                eta.squeeze(1), target.float(), reduction="none"
            )
        reference = torch.zeros((len(eta), 1), dtype=eta.dtype, device=eta.device)
        logits = torch.cat([eta, reference], dim=1)
        return F.cross_entropy(logits, target.long(), reduction="none")

    def _mean_loss(self, eta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._sample_losses(eta, target).mean()

    def _clip_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(gradient)
        if norm <= self.config.gradient_clip:
            return gradient
        return gradient * (self.config.gradient_clip / (norm + 1e-12))

    def evaluate(self, X, y) -> dict[str, Any]:
        self.biam_model.eval()
        with torch.no_grad():
            output = self.biam_model(X)
            truth = np.asarray(y)
            if self.config.task == "regression":
                predictions = output.squeeze(1).cpu().numpy()
                return {
                    "metrics": {
                        "mse": float(mean_squared_error(truth, predictions)),
                        "r2": float(r2_score(truth, predictions)),
                    },
                    "predictions": predictions,
                }
            probabilities = torch.softmax(output, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            return {
                "metrics": {
                    "accuracy": float(accuracy_score(truth, predictions)),
                    "macro_f1": float(f1_score(truth, predictions, average="macro")),
                },
                "predictions": predictions,
                "probabilities": probabilities,
            }

    def get_training_history(self) -> dict[str, list]:
        return self.training_history

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(
            {
                "model": self.biam_model.state_dict(),
                "weighting_network": self.weighting_network.state_dict(),
                "structure_logits": self.structure_logits.detach().cpu(),
                "early_stopping": self.early_stopping,
                "history": self.training_history,
                "config": self.config.to_dict(),
            },
            path,
        )
