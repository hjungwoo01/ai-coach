from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    log_loss: float
    support: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "log_loss": self.log_loss,
            "support": self.support,
        }


def classification_metrics(
    frame: pd.DataFrame,
    *,
    probability_col: str = "probability_a_win",
    label_col: str = "actual_a_win",
    threshold: float = 0.5,
) -> ClassificationMetrics:
    if frame.empty:
        return ClassificationMetrics(accuracy=0.0, precision=0.0, recall=0.0, log_loss=0.0, support=0)

    probs = frame[probability_col].astype(float).clip(lower=1e-6, upper=1 - 1e-6)
    labels = frame[label_col].astype(int)
    preds = (probs >= threshold).astype(int)

    accuracy = float((preds == labels).mean())
    true_positive = int(((preds == 1) & (labels == 1)).sum())
    predicted_positive = int((preds == 1).sum())
    actual_positive = int((labels == 1).sum())
    precision = float(true_positive / predicted_positive) if predicted_positive else 0.0
    recall = float(true_positive / actual_positive) if actual_positive else 0.0
    log_loss_value = _binary_log_loss(labels, probs)
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        log_loss=log_loss_value,
        support=int(len(frame)),
    )


def _binary_log_loss(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    losses: list[float] = []
    for label, probability in zip(labels, probabilities):
        prob = min(max(float(probability), 1e-6), 1 - 1e-6)
        losses.append(-(label * math.log(prob) + (1 - label) * math.log(1 - prob)))
    return float(sum(losses) / max(len(losses), 1))


def segmented_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    probability_col: str = "probability_a_win",
    label_col: str = "actual_a_win",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    usable_groups = [column for column in group_columns if column in frame.columns]
    if frame.empty or not usable_groups:
        return pd.DataFrame(rows)

    for group_column in usable_groups:
        grouped = frame.groupby(group_column, dropna=False)
        for group_value, group_frame in grouped:
            metrics = classification_metrics(
                group_frame,
                probability_col=probability_col,
                label_col=label_col,
            )
            row: dict[str, object] = {
                "segment": group_column,
                "value": group_value if group_value is not None else "unknown",
                **metrics.to_dict(),
            }
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["segment", "support", "accuracy"], ascending=[True, False, False])


def calibration_table(
    frame: pd.DataFrame,
    *,
    probability_col: str = "probability_a_win",
    label_col: str = "actual_a_win",
    bins: int = 10,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "bin_index",
                "bin_lower",
                "bin_upper",
                "count",
                "mean_prediction",
                "empirical_win_rate",
                "calibration_gap",
            ]
        )

    clipped = frame[probability_col].astype(float).clip(lower=0.0, upper=1.0)
    labels = frame[label_col].astype(int)
    bin_ids = pd.cut(clipped, bins=bins, labels=False, include_lowest=True, duplicates="drop")

    rows: list[dict[str, float | int]] = []
    step = 1.0 / bins
    for bin_index in sorted(bin_ids.dropna().unique()):
        mask = bin_ids == bin_index
        bin_probs = clipped.loc[mask]
        bin_labels = labels.loc[mask]
        mean_prediction = float(bin_probs.mean())
        empirical = float(bin_labels.mean())
        rows.append(
            {
                "bin_index": int(bin_index),
                "bin_lower": float(bin_index * step),
                "bin_upper": float((bin_index + 1) * step),
                "count": int(mask.sum()),
                "mean_prediction": mean_prediction,
                "empirical_win_rate": empirical,
                "calibration_gap": empirical - mean_prediction,
            }
        )

    return pd.DataFrame(rows).sort_values("bin_index").reset_index(drop=True)


def expected_calibration_error(calibration_df: pd.DataFrame) -> float:
    if calibration_df.empty:
        return 0.0
    total = float(calibration_df["count"].sum())
    if total <= 0:
        return 0.0
    weighted_gap = (calibration_df["count"] * calibration_df["calibration_gap"].abs()).sum()
    return float(weighted_gap / total)


def calibration_summary_text(calibration_df: pd.DataFrame) -> str:
    if calibration_df.empty:
        return "No calibration data available."

    ece = expected_calibration_error(calibration_df)
    largest_gap = calibration_df.loc[calibration_df["calibration_gap"].abs().idxmax()]
    direction = "underconfident" if largest_gap["calibration_gap"] > 0 else "overconfident"
    return (
        f"Expected calibration error: {ece:.4f}. "
        f"Largest deviation is in bin {int(largest_gap['bin_index'])} "
        f"({largest_gap['bin_lower']:.1f}-{largest_gap['bin_upper']:.1f}), where the model is {direction} "
        f"by {abs(float(largest_gap['calibration_gap'])):.4f}."
    )
