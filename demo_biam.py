from biam_main import run_experiment
from utils.biam_config import BIAMConfig


def main() -> None:
    config = BIAMConfig(
        task="classification",
        input_dim=8,
        n_samples=200,
        seeds=[145],
        epochs=3,
        patience=3,
        structure_samples=4,
        inner_steps=2,
        min_interaction_support=3,
        tune_refit_steps=2,
        final_refit_steps=5,
        strict_environment=False,
        output_dir="results/demo",
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
