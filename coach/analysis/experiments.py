from __future__ import annotations

import argparse
from pathlib import Path

from coach.analysis.batch_predict import run_batch_predictions
from coach.analysis.batch_strategy import run_batch_strategy
from coach.analysis.plots import plot_prediction_probabilities, plot_strategy_deltas


def run_experiments(output_dir: str | Path = "runs/experiments", mode: str = "mock") -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = run_batch_predictions(
        output_csv=out_dir / "predictions.csv",
        mode=mode,
        window=30,
        limit=10,
    )
    strat_csv = run_batch_strategy(
        output_csv=out_dir / "strategy.csv",
        mode=mode,
        window=30,
        budget=60,
        limit=5,
    )

    pred_plot = out_dir / "predictions.png"
    strat_plot = out_dir / "strategy_deltas.png"

    plot_prediction_probabilities(pred_csv, pred_plot)
    plot_strategy_deltas(strat_csv, strat_plot)

    return {
        "predictions_csv": pred_csv,
        "strategy_csv": strat_csv,
        "predictions_plot": pred_plot,
        "strategy_plot": strat_plot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible badminton coach experiments")
    parser.add_argument("--output-dir", default="runs/experiments")
    parser.add_argument("--mode", default="mock", choices=["mock", "real"])
    args = parser.parse_args()

    outputs = run_experiments(output_dir=args.output_dir, mode=args.mode)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
