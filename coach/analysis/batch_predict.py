from __future__ import annotations

import argparse
import csv
from pathlib import Path

from coach.data.adapters.local_csv import LocalCSVAdapter
from coach.service import BadmintonCoachService


def load_matchups(path: str | Path) -> list[tuple[str, str]]:
    matchups: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matchups.append((row["player_a"], row["player_b"]))
    return matchups


def default_matchups(adapter: LocalCSVAdapter, limit: int = 10) -> list[tuple[str, str]]:
    names = adapter.players_df["name"].tolist()
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs.append((a, b))
            if len(pairs) >= limit:
                return pairs
    return pairs


def run_batch_predictions(
    output_csv: str | Path,
    mode: str = "mock",
    window: int = 30,
    limit: int = 10,
    matchups_file: str | Path | None = None,
) -> Path:
    adapter = LocalCSVAdapter()
    service = BadmintonCoachService(adapter=adapter)

    if matchups_file:
        matchups = load_matchups(matchups_file)
    else:
        matchups = default_matchups(adapter=adapter, limit=limit)

    rows: list[dict[str, str | float]] = []
    for a, b in matchups:
        result = service.predict(player_a=a, player_b=b, window=window, mode=mode)
        rows.append(
            {
                "run_id": result.run_id,
                "player_a": result.player_a,
                "player_b": result.player_b,
                "probability_a_win": round(result.probability, 6),
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
    parser = argparse.ArgumentParser(description="Batch badminton matchup prediction")
    parser.add_argument("--output", default="runs/analysis/predictions.csv")
    parser.add_argument("--mode", default="mock", choices=["mock", "real"])
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--matchups-file", default=None)
    args = parser.parse_args()

    out = run_batch_predictions(
        output_csv=args.output,
        mode=args.mode,
        window=args.window,
        limit=args.limit,
        matchups_file=args.matchups_file,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
