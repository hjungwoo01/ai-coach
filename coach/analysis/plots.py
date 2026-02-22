from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "runs" / "mplcache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_prediction_probabilities(csv_path: str | Path, output_path: str | Path) -> Path:
    labels: list[str] = []
    values: list[float] = []

    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(f"{row['player_a']} vs {row['player_b']}")
            values.append(float(row["probability_a_win"]))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(values)), values, color="#2f6f9f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(A wins)")
    ax.set_title("Batch Matchup Predictions")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_strategy_deltas(csv_path: str | Path, output_path: str | Path) -> Path:
    labels: list[str] = []
    deltas: list[float] = []

    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(f"{row['player_a']} vs {row['player_b']}")
            deltas.append(float(row["delta"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f8a70" if d >= 0 else "#b93a32" for d in deltas]
    ax.bar(range(len(deltas)), deltas, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Delta P(A wins)")
    ax.set_title("Strategy Search Improvements")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
