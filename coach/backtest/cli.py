from __future__ import annotations

import argparse
from pathlib import Path

from coach.backtest.data import build_historical_dataset, fetch_and_build_dataset
from coach.backtest.engine import BacktestConfig, TimeMachineBacktester
from coach.backtest.feature_contract import write_feature_contract


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time-machine backtesting tools for the badminton PCSP model")
    sub = parser.add_subparsers(dest="command", required=True)

    contract = sub.add_parser("contract", help="Export the current PCSP feature contract")
    contract.add_argument("--output", default="runs/backtests/feature_contract.md")
    contract.add_argument("--format", default="markdown", choices=["markdown", "json"])

    ingest = sub.add_parser("ingest", help="Fetch ShuttleSet and build the historical dataset")
    ingest.add_argument("--raw-set-dir", default=None, help="Existing ShuttleSet set/ directory")
    ingest.add_argument("--workdir", default="data/raw/backtest")
    ingest.add_argument("--players-out", default="coach/data/sample_players.csv")
    ingest.add_argument("--matches-out", default="coach/data/sample_matches.csv")

    run = sub.add_parser("run", help="Run leakage-safe walk-forward backtesting")
    run.add_argument("--players", default="coach/data/sample_players.csv")
    run.add_argument("--matches", default="coach/data/sample_matches.csv")
    run.add_argument("--mode", default="mock", choices=["mock", "real"])
    run.add_argument("--window", type=int, default=30)
    run.add_argument("--pat-path", default=None)
    run.add_argument("--timeout", type=int, default=None)
    run.add_argument("--timestamp-column", default=None)
    run.add_argument("--start-date", default=None)
    run.add_argument("--end-date", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--calibration-bins", type=int, default=10)
    run.add_argument("--persist-match-artifacts", action="store_true")
    run.add_argument("--disable-fast-mock", action="store_true")
    run.add_argument("--runs-root", default="runs/backtests")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "contract":
        output = write_feature_contract(args.output, format=args.format)
        print(output)
        return

    if args.command == "ingest":
        if args.raw_set_dir:
            paths = build_historical_dataset(
                raw_set_dir=args.raw_set_dir,
                players_out=args.players_out,
                matches_out=args.matches_out,
            )
        else:
            paths = fetch_and_build_dataset(
                working_dir=args.workdir,
                players_out=args.players_out,
                matches_out=args.matches_out,
            )
        print(f"players_csv: {paths.players_csv}")
        print(f"matches_csv: {paths.matches_csv}")
        print(f"raw_set_dir: {paths.raw_set_dir}")
        return

    config = BacktestConfig(
        mode=args.mode,
        window=args.window,
        pat_path=args.pat_path,
        timeout_s=args.timeout,
        timestamp_column=args.timestamp_column,
        fast_mock=not args.disable_fast_mock,
        persist_match_artifacts=args.persist_match_artifacts,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        calibration_bins=args.calibration_bins,
    )
    backtester = TimeMachineBacktester.from_paths(
        players_path=args.players,
        matches_path=args.matches,
        config=config,
        runs_root=Path(args.runs_root),
    )
    report = backtester.run()
    overall = report.metrics["overall"]
    rankings = report.metrics["rankings"]
    print(f"run_id: {report.run_id}")
    print(f"run_dir: {report.run_dir}")
    print(f"matches_total: {report.metrics['matches_total']}")
    print(f"matches_scored: {report.metrics['matches_scored']}")
    print(f"coverage: {report.metrics['coverage']:.6f}")
    print(f"best_calibrated_log_loss_model: {rankings['best_calibrated_log_loss']}")
    print(f"best_calibrated_accuracy_model: {rankings['best_calibrated_accuracy']}")

    for model_name in ("pcsp", "elo", "recent_form"):
        if model_name not in overall:
            continue
        calibrated = overall[model_name]["calibrated"]
        print(f"{model_name}_accuracy: {calibrated['accuracy']:.6f}")
        print(f"{model_name}_precision: {calibrated['precision']:.6f}")
        print(f"{model_name}_recall: {calibrated['recall']:.6f}")
        print(f"{model_name}_log_loss: {calibrated['log_loss']:.6f}")
        print(f"{model_name}_calibration: {report.metrics['calibration_summary'][model_name]['calibrated']}")


if __name__ == "__main__":
    main()
