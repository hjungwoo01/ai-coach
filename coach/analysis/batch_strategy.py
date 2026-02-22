from __future__ import annotations

import argparse
import csv
from pathlib import Path

from coach.data.adapters.local_csv import LocalCSVAdapter
from coach.service import BadmintonCoachService


def default_strategy_queries(adapter: LocalCSVAdapter, limit: int = 5) -> list[tuple[str, str]]:
    names = adapter.players_df["name"].tolist()
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs.append((a, b))
            if len(pairs) >= limit:
                return pairs
    return pairs


def run_batch_strategy(
    output_csv: str | Path,
    mode: str = "mock",
    window: int = 30,
    budget: int = 60,
    limit: int = 5,
) -> Path:
    adapter = LocalCSVAdapter()
    service = BadmintonCoachService(adapter=adapter)

    rows: list[dict[str, str | float]] = []
    for a, b in default_strategy_queries(adapter, limit=limit):
        result = service.strategy(
            player_a=a,
            player_b=b,
            window=window,
            mode=mode,
            budget=budget,
        )
        rows.append(
            {
                "run_id": result.run_id,
                "player_a": result.player_a,
                "player_b": result.player_b,
                "baseline_probability": round(result.baseline_probability, 6),
                "improved_probability": round(result.improved_probability, 6),
                "delta": round(result.delta, 6),
                "best_serve_short_delta": round(result.best_candidate.serve_short_delta, 6),
                "best_attack_delta": round(result.best_candidate.attack_delta, 6),
                "mode": result.mode,
                "run_dir": str(result.run_dir),
            }
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch badminton strategy optimization")
    parser.add_argument("--output", default="runs/analysis/strategy.csv")
    parser.add_argument("--mode", default="mock", choices=["mock", "real"])
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    out = run_batch_strategy(
        output_csv=args.output,
        mode=args.mode,
        window=args.window,
        budget=args.budget,
        limit=args.limit,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
