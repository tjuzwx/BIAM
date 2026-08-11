import json

import numpy as np

from biam_main import run_experiment
from utils.biam_config import BIAMConfig


def test_multiseed_run_writes_complete_reproduction_record(tmp_path):
    config = BIAMConfig(
        task="regression",
        input_dim=6,
        n_samples=120,
        seeds=[11, 22],
        epochs=1,
        patience=1,
        structure_samples=4,
        inner_steps=1,
        min_interaction_support=2,
        tune_refit_steps=1,
        final_refit_steps=2,
        batch_size=32,
        meta_batch_size=16,
        output_dir=str(tmp_path),
        device="cpu",
        strict_environment=False,
    )
    summary = run_experiment(config)
    root = tmp_path / "regression_synthetic"

    assert summary["seeds"] == [11, 22]
    assert summary["num_runs"] == 2
    assert set(summary["metrics"]) == {"mse", "r2"}
    assert np.isfinite(summary["metrics"]["mse"]["std"])
    for seed in config.seeds:
        run = root / f"seed_{seed}" / "fold_0"
        assert (run / "split_indices.npz").exists()
        assert (run / "data_artifacts.npz").exists()
        assert (run / "preprocessing.npz").exists()
        assert (run / "predictions.csv").exists()
        assert (run / "training.json").exists()
    with (root / "summary.json").open(encoding="utf-8") as handle:
        assert json.load(handle)["num_runs"] == 2
