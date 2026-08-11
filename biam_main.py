from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.biam_data_generator import BIAMDataGenerator
from gradients.biam_optimizer import BIAMOptimizer
from models.biam_model import BIAMModel
from utils.biam_config import BIAMConfig
from utils.biam_environment import validate_paper_environment
from utils.biam_reproducibility import child_seed, set_global_seed


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2)


def _git_provenance() -> dict[str, Any]:
    result = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
        result["git_dirty"] = None
    return result


def _save_reproduction_artifacts(run_dir: Path, bundle, model: BIAMModel) -> None:
    np.savez_compressed(
        run_dir / "split_indices.npz",
        **{name: values for name, values in bundle.indices.items()},
    )
    arrays = {
        name: value
        for name, value in bundle.artifacts.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(run_dir / "data_artifacts.npz", **arrays)
    np.savez_compressed(
        run_dir / "preprocessing.npz", **model.additive_model.builder.metadata()
    )


def _save_predictions(run_dir: Path, bundle, evaluation: dict[str, Any]) -> None:
    table = {
        "sample_index": bundle.indices["test"],
        "y_true": bundle.y_test,
        "y_pred": evaluation["predictions"],
    }
    if "probabilities" in evaluation:
        for label in range(evaluation["probabilities"].shape[1]):
            table[f"probability_{label}"] = evaluation["probabilities"][:, label]
    pd.DataFrame(table).to_csv(run_dir / "predictions.csv", index=False)


def run_experiment(config: BIAMConfig) -> dict[str, Any]:
    environment = validate_paper_environment(config.strict_environment)
    output_root = Path(config.output_dir) / f"{config.task}_{config.dataset}"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "config.json", config.to_dict())
    provenance = _git_provenance()
    provenance["environment"] = environment
    _write_json(output_root / "provenance.json", provenance)

    seed_records = []
    for seed in config.seeds:
        set_global_seed(int(seed), config.deterministic)
        bundles = BIAMDataGenerator(config).generate_runs(int(seed))
        fold_records = []
        for bundle in bundles:
            run_dir = output_root / f"seed_{seed}" / f"fold_{bundle.fold}"
            run_dir.mkdir(parents=True, exist_ok=True)
            optimizer_seed = (
                int(seed)
                if len(bundles) == 1
                else child_seed(int(seed), int(bundle.fold), 99)
            )
            set_global_seed(optimizer_seed, config.deterministic)
            bundle.artifacts["optimizer_seed"] = optimizer_seed
            model = BIAMModel(config, config.device)
            optimizer = BIAMOptimizer(config, model, model.weighting_network)
            training = optimizer.fit(bundle, optimizer_seed)
            evaluation = optimizer.evaluate(bundle.X_test, bundle.y_test)

            _save_reproduction_artifacts(run_dir, bundle, model)
            _save_predictions(run_dir, bundle, evaluation)
            _write_json(run_dir / "metrics.json", evaluation["metrics"])
            _write_json(run_dir / "training.json", training)
            artifact_record = {
                name: (
                    {"file": "data_artifacts.npz", "shape": list(value.shape), "dtype": str(value.dtype)}
                    if isinstance(value, np.ndarray)
                    else value
                )
                for name, value in bundle.artifacts.items()
            }
            _write_json(run_dir / "data_run.json", artifact_record)
            optimizer.save_checkpoint(run_dir / "model.pt")
            fold_records.append(
                {"fold": bundle.fold, "metrics": evaluation["metrics"], "path": str(run_dir)}
            )
            print(f"种子 {seed}，折 {bundle.fold}：{evaluation['metrics']}")

        metric_names = fold_records[0]["metrics"].keys()
        seed_metrics = {
            name: float(np.mean([record["metrics"][name] for record in fold_records]))
            for name in metric_names
        }
        seed_record = {"seed": int(seed), "metrics": seed_metrics, "folds": fold_records}
        seed_records.append(seed_record)
        _write_json(output_root / f"seed_{seed}" / "seed_summary.json", seed_record)

    metric_names = seed_records[0]["metrics"].keys()
    summary_metrics = {}
    for name in metric_names:
        values = np.asarray([record["metrics"][name] for record in seed_records], dtype=float)
        summary_metrics[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    summary = {
        "seeds": [int(seed) for seed in config.seeds],
        "num_runs": len(seed_records),
        "metrics": summary_metrics,
        "runs": seed_records,
    }
    _write_json(output_root / "summary.json", summary)
    print(f"汇总结果已写入 {output_root / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="概率双层交互可加模型 BIAM")
    parser.add_argument("--config", default="configs/biam_default.yaml")
    parser.add_argument("--task", choices=["regression", "classification"])
    parser.add_argument("--dataset", choices=["synthetic", "npz"])
    parser.add_argument("--data-path")
    parser.add_argument("--target-key")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument(
        "--allow-environment-mismatch",
        dest="strict_environment",
        action="store_false",
        default=None,
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--n-samples", dest="n_samples", type=int)
    parser.add_argument("--input-dim", dest="input_dim", type=int)
    parser.add_argument("--num-classes", dest="num_classes", type=int)
    parser.add_argument(
        "--standardize-target", dest="standardize_target", action="store_true", default=None
    )
    parser.add_argument("--missing-mechanism", choices=["NONE", "MCAR", "MAR", "MNAR"])
    parser.add_argument("--missing-ratio", type=float)
    parser.add_argument("--noise-ratio", type=float)
    parser.add_argument("--imbalance-ratio", type=float)
    parser.add_argument("--structure-samples", type=int)
    parser.add_argument("--inner-steps", type=int)
    parser.add_argument("--hinge-bins", type=int)
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    quick = args.quick
    delattr(args, "quick")
    config = BIAMConfig.from_yaml(args.config, args)
    if quick:
        config.update(
            seeds=[config.seeds[0]],
            epochs=2,
            patience=2,
            n_samples=min(config.n_samples, 160),
            input_dim=min(config.input_dim, 8),
            structure_samples=4,
            inner_steps=2,
            tune_refit_steps=2,
            final_refit_steps=4,
            min_interaction_support=3,
        )
    run_experiment(config)


if __name__ == "__main__":
    main()
