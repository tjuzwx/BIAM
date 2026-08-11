import numpy as np

from data.biam_data_generator import BIAMDataGenerator
from utils.biam_config import BIAMConfig
from utils.biam_environment import PAPER_ENVIRONMENT, validate_paper_environment


def small_config(**overrides):
    values = {
        "task": "regression",
        "input_dim": 8,
        "n_samples": 200,
        "missing_mechanism": "MAR",
        "missing_ratio": 0.3,
        "noise_ratio": 0.2,
        "min_interaction_support": 3,
        "seeds": [11],
    }
    values.update(overrides)
    return BIAMConfig(**values)


def test_environment_report_uses_paper_requirements():
    report = validate_paper_environment(strict=False)

    assert report["paper_required"] == PAPER_ENVIRONMENT
    assert set(report["checks"]) == {
        "Ubuntu 20.04 LTS",
        "Intel Xeon Platinum 8175M",
        "NVIDIA RTX A6000",
        "GPU 48GB",
        "内存 128GB",
        "CUDA 12.1",
        "PyTorch 2.1.0",
    }
    assert report["matched"] == all(report["checks"].values())
    assert report["strict_validation"] is False
    assert report["formal_environment_validated"] is False


def test_four_way_split_is_disjoint_and_reproducible():
    config = small_config()
    first = BIAMDataGenerator(config).generate_data(11)
    second = BIAMDataGenerator(config).generate_data(11)
    combined = np.concatenate(list(first.indices.values()))

    assert len(first.indices["train"]) == 140
    assert len(first.indices["meta"]) == 20
    assert len(first.indices["tune"]) == 20
    assert len(first.indices["test"]) == 20
    assert len(np.unique(combined)) == 200
    for name in first.indices:
        np.testing.assert_array_equal(first.indices[name], second.indices[name])
    np.testing.assert_array_equal(
        first.artifacts["artificial_missing_mask"],
        second.artifacts["artificial_missing_mask"],
    )


def test_corruption_only_changes_training_targets():
    bundle = BIAMDataGenerator(small_config()).generate_data(11)
    changed = np.flatnonzero(bundle.y_train != bundle.y_train_clean)

    assert 10 <= len(bundle.artifacts["response_outlier_local_indices"]) <= 50
    assert len(changed) > 0
    assert np.isfinite(bundle.y_meta).all()
    assert np.isfinite(bundle.y_tune).all()
    assert np.isfinite(bundle.y_test).all()


def test_missing_calibration_uses_candidate_features_only():
    bundle = BIAMDataGenerator(small_config()).generate_data(11)
    mask = bundle.artifacts["artificial_missing_mask"]
    candidates = bundle.artifacts["candidate_features"]
    train = bundle.indices["train"]
    outside = np.setdiff1d(np.arange(mask.shape[1]), candidates)

    assert abs(mask[np.ix_(train, candidates)].mean() - 0.3) < 0.08
    assert not mask[:, outside].any()
    assert np.all(bundle.artifacts["mar_anchors"] >= len(candidates))


def test_classification_training_split_is_long_tailed_and_noisy():
    config = small_config(task="classification", imbalance_ratio=0.2, noise_ratio=0.1)
    bundle = BIAMDataGenerator(config).generate_data(11)
    counts = np.bincount(bundle.y_train_clean.astype(int))
    selected = bundle.artifacts["label_noise_local_indices"]
    distances = np.abs(bundle.artifacts["classification_boundary_scores"])

    assert counts.min() / counts.max() <= 0.3
    assert len(selected) == int(0.1 * len(bundle.y_train))
    assert distances[selected].max() <= np.partition(distances, len(selected) - 1)[len(selected) - 1]
    assert set(np.unique(bundle.y_meta)) == {0, 1}


def test_external_regression_uses_outer_folds_and_training_target_statistics(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 6))
    y = 3.0 + 2.0 * X[:, 0] + rng.normal(scale=0.1, size=100)
    path = tmp_path / "regression.npz"
    np.savez(path, X=X, y=y)
    config = small_config(
        dataset="npz",
        data_path=str(path),
        standardize_target=True,
        missing_mechanism="NONE",
        regression_outer_folds=5,
    )
    bundles = BIAMDataGenerator(config).generate_runs(11)

    assert len(bundles) == 5
    np.testing.assert_array_equal(
        np.sort(np.concatenate([bundle.indices["test"] for bundle in bundles])),
        np.arange(100),
    )
    for bundle in bundles:
        assert abs(bundle.y_train_clean.mean()) < 1e-10
        assert abs(bundle.y_train_clean.std() - 1.0) < 1e-10
        assert bundle.artifacts["target_std"] > 0
