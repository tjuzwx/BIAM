import numpy as np
import torch

from gradients.biam_optimizer import BIAMOptimizer
from models.biam_additive_model import BIAMFeatureBuilder
from models.biam_model import BIAMModel
from utils.biam_config import BIAMConfig


def test_effect_bases_are_centered_on_training_support():
    X = np.array(
        [
            [np.nan, 0.0, 1.0],
            [1.0, 1.0, 2.0],
            [np.nan, 2.0, 3.0],
            [3.0, 3.0, 4.0],
            [4.0, 4.0, np.nan],
        ]
    )
    builder = BIAMFeatureBuilder(hinge_bins=2, min_support=1).fit(X)
    design = builder.transform(X)

    for group in builder.groups:
        if group.kind in {"obs", "int"}:
            np.testing.assert_allclose(design[:, group.start : group.stop].sum(axis=0), 0.0, atol=1e-6)
        elif group.kind == "miss":
            np.testing.assert_allclose(design[:, group.start].mean(), 0.0, atol=1e-6)


def test_interactions_are_directed_and_pass_support_filter():
    X = np.array(
        [
            [np.nan, 1.0, 1.0],
            [np.nan, 2.0, 2.0],
            [1.0, np.nan, 3.0],
            [2.0, 4.0, 4.0],
        ]
    )
    builder = BIAMFeatureBuilder(hinge_bins=1, min_support=2).fit(X)
    pairs = {tuple(pair) for pair in builder.interactions.tolist()}

    assert (0, 1) in pairs
    assert (1, 0) not in pairs
    assert np.all(builder.interaction_supports >= 2)


def test_hard_structure_controls_complete_function_groups():
    config = BIAMConfig(
        task="classification",
        input_dim=3,
        hinge_bins=2,
        min_interaction_support=1,
        seeds=[11],
    )
    model = BIAMModel(config, "cpu")
    X = np.array([[np.nan, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 2.0, 3.0]])
    model.additive_model.fit_preprocessor(X)
    gates = torch.zeros(len(model.additive_model.builder.groups), dtype=torch.bool)
    gates[0] = True
    columns = model.additive_model.active_columns(gates)
    theta = torch.zeros(len(columns), 1)
    theta[0] = 1.5
    model.additive_model.set_final_state(gates, theta)

    output = model(X)
    assert output.shape == (3, 2)
    assert model.additive_model.selected_groups() == ["O_0"]


def test_leave_pair_out_baseline_excludes_antithetic_partner():
    risks = torch.tensor([1.0, 2.0, 3.0, 4.0])
    baseline = BIAMOptimizer._leave_pair_out_baselines(risks)
    torch.testing.assert_close(baseline, torch.tensor([3.5, 3.5, 1.5, 1.5]))
