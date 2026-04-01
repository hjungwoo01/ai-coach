from __future__ import annotations

import datetime as dt
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from coach.backtest.data import (
    create_snapshot_adapter,
    load_backtest_frames,
    prepare_chronological_matches,
)
from coach.backtest.feature_contract import write_feature_contract
from coach.backtest.metrics import (
    calibration_summary_text,
    calibration_table,
    classification_metrics,
    expected_calibration_error,
    segmented_metrics,
)
from coach.backtest.models import (
    EloRatingModel,
    OnlineRecentFormLogisticModel,
    RollingPlattCalibrator,
    extract_recent_form_feature_vector,
    summarize_model_rankings,
)
from coach.config import CoachConfig
from coach.data.stats_builder import build_matchup_params
from coach.model.builder import build_matchup_model
from coach.model.params import MatchupParams
from coach.pat.mock_pat import mock_probability
from coach.pat.parser import parse_probability, read_pat_output
from coach.pat.runner import run_pat
from coach.runs import new_run_dir
from coach.utils import write_json


@dataclass(frozen=True)
class BacktestConfig:
    mode: Literal["mock", "real"] = "mock"
    window: int = 30
    pat_path: str | None = None
    timeout_s: int | None = None
    timestamp_column: str | None = None
    fast_mock: bool = True
    persist_match_artifacts: bool = False
    start_date: str | None = None
    end_date: str | None = None
    limit: int | None = None
    calibration_bins: int = 10


@dataclass(frozen=True)
class BacktestArtifacts:
    predictions_csv: Path
    skipped_csv: Path
    model_metrics_csv: Path
    segments_csv: Path
    calibration_csv: Path
    metrics_json: Path
    summary_txt: Path
    feature_contract_md: Path


@dataclass(frozen=True)
class BacktestReport:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    skipped_matches: pd.DataFrame
    calibration: pd.DataFrame
    segments: pd.DataFrame
    artifacts: BacktestArtifacts


class TimeMachineBacktester:
    """Leakage-safe walk-forward evaluator for the current PCSP model."""

    MODEL_COLUMNS = {
        "pcsp": {
            "raw": "pcsp_raw_probability_a_win",
            "calibrated": "pcsp_probability_a_win",
        },
        "elo": {
            "raw": "elo_raw_probability_a_win",
            "calibrated": "elo_probability_a_win",
        },
        "recent_form": {
            "raw": "recent_form_raw_probability_a_win",
            "calibrated": "recent_form_probability_a_win",
        },
    }

    def __init__(
        self,
        *,
        players_df: pd.DataFrame,
        matches_df: pd.DataFrame,
        config: BacktestConfig | None = None,
        runs_root: str | Path | None = None,
    ) -> None:
        self.players_df = players_df.copy().reset_index(drop=True)
        self.matches_df = matches_df.copy().reset_index(drop=True)
        self.config = config or BacktestConfig()
        self.coach_config = CoachConfig.from_env()
        self.runs_root = Path(runs_root) if runs_root is not None else Path("runs") / "backtests"
        self._snapshot_cache: dict[str, object] = {}

    @classmethod
    def from_paths(
        cls,
        *,
        players_path: str | Path,
        matches_path: str | Path,
        config: BacktestConfig | None = None,
        runs_root: str | Path | None = None,
    ) -> "TimeMachineBacktester":
        players_df, matches_df = load_backtest_frames(players_path, matches_path)
        return cls(players_df=players_df, matches_df=matches_df, config=config, runs_root=runs_root)

    def run(self) -> BacktestReport:
        run_id, run_dir = new_run_dir(prefix="backtest", base_dir=self.runs_root)
        prepared = self._prepare_matches()

        prediction_rows: list[dict[str, Any]] = []
        skipped_rows: list[dict[str, Any]] = []
        elo_model = EloRatingModel()
        recent_form_model = OnlineRecentFormLogisticModel()
        calibrators = {
            "pcsp": RollingPlattCalibrator(),
            "elo": RollingPlattCalibrator(),
            "recent_form": RollingPlattCalibrator(),
        }

        for row in prepared.itertuples(index=False):
            cutoff_key = str(row.cutoff_key)
            snapshot_adapter = self._get_snapshot_adapter(
                cutoff_key=cutoff_key,
                cutoff_timestamp=pd.Timestamp(row.cutoff_timestamp),
            )
            try:
                params, stats = build_matchup_params(
                    adapter=snapshot_adapter,
                    player_a_ref=str(row.playerA_id),
                    player_b_ref=str(row.playerB_id),
                    window=self.config.window,
                    as_of_date=cutoff_key,
                    allow_cold_start=True,
                )
            except ValueError as exc:
                skipped_rows.append(self._build_skipped_row(row=row, reason=str(exc)))
                continue

            try:
                pcsp_prediction = self._predict_match(
                    match_id=str(row.match_id),
                    params=params,
                    run_dir=run_dir,
                )
            except Exception as exc:  # noqa: BLE001
                skipped_rows.append(self._build_skipped_row(row=row, reason=str(exc)))
                continue

            actual_a_win = int(row.actual_a_win)
            feature_vector = extract_recent_form_feature_vector(params)

            raw_probabilities = {
                "pcsp": float(pcsp_prediction["probability"]),
                "elo": elo_model.predict(str(row.playerA_id), str(row.playerB_id)),
                "recent_form": recent_form_model.predict(feature_vector),
            }
            calibrated_probabilities = {
                model_name: calibrators[model_name].transform(raw_probability)
                for model_name, raw_probability in raw_probabilities.items()
            }

            row_payload = {
                "run_id": run_id,
                "match_id": str(row.match_id),
                "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "cutoff_timestamp": pd.Timestamp(row.cutoff_timestamp).isoformat(),
                "snapshot_policy": str(row.snapshot_policy),
                "player_a_id": str(row.playerA_id),
                "player_b_id": str(row.playerB_id),
                "winner_id": str(row.winner_id),
                "actual_a_win": actual_a_win,
                "player_a_history_matches": int(params.player_a.sample_matches),
                "player_b_history_matches": int(params.player_b.sample_matches),
                "pA_srv_win": float(pcsp_prediction["effective_probabilities"]["pA_srv_win"]),
                "pA_rcv_win": float(pcsp_prediction["effective_probabilities"]["pA_rcv_win"]),
                "round": getattr(row, "round", None),
                "round_bucket": getattr(row, "round_bucket", "unknown"),
                "tournament": getattr(row, "tournament", None),
                "discipline": getattr(row, "discipline", None),
                "calendar_year": int(getattr(row, "calendar_year", pd.Timestamp(row.date).year)),
                "mode": self.config.mode,
                "artifact_dir": pcsp_prediction.get("artifact_dir"),
                "player_a_name": stats.player_a.name,
                "player_b_name": stats.player_b.name,
                "recent_form_edge": float(params.player_a.recent_form - params.player_b.recent_form),
                "rest_edge": float((params.player_a.rest_days - params.player_b.rest_days) / 14.0),
            }

            for model_name, column_map in self.MODEL_COLUMNS.items():
                raw_probability = float(raw_probabilities[model_name])
                calibrated_probability = float(calibrated_probabilities[model_name])
                row_payload[column_map["raw"]] = raw_probability
                row_payload[column_map["calibrated"]] = calibrated_probability
                row_payload[f"{model_name}_raw_predicted_a_win"] = int(raw_probability >= 0.5)
                row_payload[f"{model_name}_predicted_a_win"] = int(calibrated_probability >= 0.5)
                row_payload[f"{model_name}_raw_correct"] = int((raw_probability >= 0.5) == actual_a_win)
                row_payload[f"{model_name}_correct"] = int((calibrated_probability >= 0.5) == actual_a_win)

            prediction_rows.append(row_payload)

            for model_name, raw_probability in raw_probabilities.items():
                calibrators[model_name].update(raw_probability, actual_a_win)

            elo_model.update(str(row.playerA_id), str(row.playerB_id), actual_a_win)
            recent_form_model.update(feature_vector, actual_a_win)

        predictions_df = pd.DataFrame(prediction_rows)
        skipped_df = pd.DataFrame(skipped_rows)
        model_metrics_df, calibration_df, segments_df, metrics_payload = self._build_comparison_outputs(
            prepared_matches=prepared,
            predictions=predictions_df,
            skipped=skipped_df,
        )

        artifacts = self._write_report_artifacts(
            run_dir=run_dir,
            predictions=predictions_df,
            skipped=skipped_df,
            model_metrics=model_metrics_df,
            calibration=calibration_df,
            segments=segments_df,
            metrics_payload=metrics_payload,
        )

        return BacktestReport(
            run_id=run_id,
            run_dir=run_dir,
            metrics=metrics_payload,
            predictions=predictions_df,
            skipped_matches=skipped_df,
            calibration=calibration_df,
            segments=segments_df,
            artifacts=artifacts,
        )

    def _prepare_matches(self) -> pd.DataFrame:
        prepared = prepare_chronological_matches(
            self.matches_df,
            timestamp_column=self.config.timestamp_column,
        )

        if self.config.start_date is not None:
            start = pd.Timestamp(self.config.start_date)
            prepared = prepared.loc[prepared["date"] >= start]
        if self.config.end_date is not None:
            end = pd.Timestamp(self.config.end_date)
            prepared = prepared.loc[prepared["date"] <= end]
        if self.config.limit is not None:
            prepared = prepared.head(self.config.limit)

        return prepared.reset_index(drop=True)

    def _get_snapshot_adapter(self, *, cutoff_key: str, cutoff_timestamp: pd.Timestamp) -> object:
        snapshot = self._snapshot_cache.get(cutoff_key)
        if snapshot is None:
            snapshot = create_snapshot_adapter(
                players_df=self.players_df,
                matches_df=self.matches_df,
                cutoff=cutoff_timestamp,
                timestamp_column=self.config.timestamp_column,
            )
            self._snapshot_cache[cutoff_key] = snapshot
        return snapshot

    def _predict_match(
        self,
        *,
        match_id: str,
        params: MatchupParams,
        run_dir: Path,
    ) -> dict[str, Any]:
        context = params.to_template_context()
        effective_probabilities = params.effective_probabilities()

        if self.config.mode == "mock" and self.config.fast_mock:
            return {
                "probability": mock_probability(context),
                "effective_probabilities": effective_probabilities,
                "artifact_dir": None,
            }

        artifact_dir_path: Path | None = None
        if self.config.persist_match_artifacts:
            artifact_dir_path = run_dir / "matches" / match_id
            artifact_dir_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"{match_id}_", dir=None if artifact_dir_path else None) as tmp_dir:
            work_dir = artifact_dir_path or Path(tmp_dir)
            pcsp_path = work_dir / "matchup.pcsp"
            out_path = work_dir / "pat_output.txt"
            build_matchup_model(params=params, out_path=pcsp_path)

            result = run_pat(
                pcsp_path=pcsp_path,
                out_path=out_path,
                mode=self.config.mode,
                pat_console_path=(
                    Path(self.config.pat_path).expanduser()
                    if self.config.pat_path
                    else self.coach_config.pat_console_path
                ),
                timeout_s=self.config.timeout_s or self.coach_config.pat_timeout_s,
                use_mono=self.coach_config.resolve_use_mono(
                    Path(self.config.pat_path).expanduser()
                    if self.config.pat_path
                    else self.coach_config.pat_console_path
                ),
            )

            probability = result.get("probability")
            if probability is None and out_path.exists():
                probability = parse_probability(read_pat_output(out_path))
            if probability is None:
                raise RuntimeError("PAT execution did not produce a probability.")

            return {
                "probability": float(probability),
                "effective_probabilities": effective_probabilities,
                "artifact_dir": str(artifact_dir_path) if artifact_dir_path is not None else None,
            }

    @staticmethod
    def _build_skipped_row(*, row: Any, reason: str) -> dict[str, Any]:
        return {
            "match_id": str(row.match_id),
            "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
            "player_a_id": str(row.playerA_id),
            "player_b_id": str(row.playerB_id),
            "winner_id": str(row.winner_id),
            "cutoff_timestamp": pd.Timestamp(row.cutoff_timestamp).isoformat(),
            "snapshot_policy": str(row.snapshot_policy),
            "reason": reason,
            "round": getattr(row, "round", None),
            "tournament": getattr(row, "tournament", None),
        }

    def _build_metrics_payload(
        self,
        *,
        prepared_matches: pd.DataFrame,
        predictions: pd.DataFrame,
        skipped: pd.DataFrame,
    ) -> dict[str, Any]:
        _, _, _, payload = self._build_comparison_outputs(
            prepared_matches=prepared_matches,
            predictions=predictions,
            skipped=skipped,
        )
        return payload

    def _build_comparison_outputs(
        self,
        *,
        prepared_matches: pd.DataFrame,
        predictions: pd.DataFrame,
        skipped: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        def calibration_subset(model_name: str, probability_kind: str) -> pd.DataFrame:
            if calibration_df.empty or "model" not in calibration_df.columns:
                return pd.DataFrame()
            return calibration_df[
                (calibration_df["model"] == model_name)
                & (calibration_df["probability_kind"] == probability_kind)
            ]

        coverage = float(len(predictions) / max(len(prepared_matches), 1))
        model_metric_rows: list[dict[str, Any]] = []
        calibration_frames: list[pd.DataFrame] = []
        segment_frames: list[pd.DataFrame] = []
        overall_metrics: dict[str, dict[str, dict[str, float | int]]] = {}

        for model_name, column_map in self.MODEL_COLUMNS.items():
            overall_metrics[model_name] = {}
            for probability_kind, probability_column in column_map.items():
                metrics = classification_metrics(
                    predictions,
                    probability_col=probability_column,
                    label_col="actual_a_win",
                )
                overall_metrics[model_name][probability_kind] = metrics.to_dict()
                model_metric_rows.append(
                    {
                        "model": model_name,
                        "probability_kind": probability_kind,
                        **metrics.to_dict(),
                    }
                )

                calibration_frame = calibration_table(
                    predictions,
                    probability_col=probability_column,
                    label_col="actual_a_win",
                    bins=self.config.calibration_bins,
                )
                if not calibration_frame.empty:
                    calibration_frame.insert(0, "probability_kind", probability_kind)
                    calibration_frame.insert(0, "model", model_name)
                    calibration_frames.append(calibration_frame)

                segment_frame = segmented_metrics(
                    predictions,
                    group_columns=["discipline", "round_bucket", "round", "calendar_year", "tournament"],
                    probability_col=probability_column,
                    label_col="actual_a_win",
                )
                if not segment_frame.empty:
                    segment_frame.insert(0, "probability_kind", probability_kind)
                    segment_frame.insert(0, "model", model_name)
                    segment_frames.append(segment_frame)

        model_metrics_df = pd.DataFrame(model_metric_rows)
        calibration_df = (
            pd.concat(calibration_frames, ignore_index=True)
            if calibration_frames
            else pd.DataFrame()
        )
        segments_df = (
            pd.concat(segment_frames, ignore_index=True)
            if segment_frames
            else pd.DataFrame()
        )

        calibrated_rankings = summarize_model_rankings(
            {name: metrics["calibrated"] for name, metrics in overall_metrics.items()},
            sort_key="log_loss",
            ascending=True,
        )
        calibrated_accuracy_rankings = summarize_model_rankings(
            {name: metrics["calibrated"] for name, metrics in overall_metrics.items()},
            sort_key="accuracy",
            ascending=False,
        )
        raw_rankings = summarize_model_rankings(
            {name: metrics["raw"] for name, metrics in overall_metrics.items()},
            sort_key="log_loss",
            ascending=True,
        )

        payload = {
            "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
            "mode": self.config.mode,
            "window": self.config.window,
            "matches_total": int(len(prepared_matches)),
            "matches_scored": int(len(predictions)),
            "matches_skipped": int(len(skipped)),
            "coverage": coverage,
            "snapshot_policies": sorted({str(policy) for policy in prepared_matches["snapshot_policy"].unique()}),
            "overall": overall_metrics,
            "rankings": {
                "best_calibrated_log_loss": calibrated_rankings[0][0] if calibrated_rankings else None,
                "best_calibrated_accuracy": calibrated_accuracy_rankings[0][0] if calibrated_accuracy_rankings else None,
                "best_raw_log_loss": raw_rankings[0][0] if raw_rankings else None,
            },
            "expected_calibration_error": {
                model_name: {
                    probability_kind: expected_calibration_error(calibration_subset(model_name, probability_kind))
                    for probability_kind in ("raw", "calibrated")
                }
                for model_name in self.MODEL_COLUMNS
            },
            "calibration_summary": {
                model_name: {
                    probability_kind: calibration_summary_text(calibration_subset(model_name, probability_kind))
                    for probability_kind in ("raw", "calibrated")
                }
                for model_name in self.MODEL_COLUMNS
            },
        }
        return model_metrics_df, calibration_df, segments_df, payload

    def _write_report_artifacts(
        self,
        *,
        run_dir: Path,
        predictions: pd.DataFrame,
        skipped: pd.DataFrame,
        model_metrics: pd.DataFrame,
        calibration: pd.DataFrame,
        segments: pd.DataFrame,
        metrics_payload: dict[str, Any],
    ) -> BacktestArtifacts:
        run_dir.mkdir(parents=True, exist_ok=True)

        predictions_csv = run_dir / "predictions.csv"
        skipped_csv = run_dir / "skipped_matches.csv"
        model_metrics_csv = run_dir / "model_metrics.csv"
        segments_csv = run_dir / "segmented_metrics.csv"
        calibration_csv = run_dir / "calibration.csv"
        metrics_json = run_dir / "metrics.json"
        summary_txt = run_dir / "summary.txt"
        feature_contract_md = run_dir / "feature_contract.md"

        predictions.to_csv(predictions_csv, index=False)
        skipped.to_csv(skipped_csv, index=False)
        model_metrics.to_csv(model_metrics_csv, index=False)
        segments.to_csv(segments_csv, index=False)
        calibration.to_csv(calibration_csv, index=False)
        write_json(metrics_json, metrics_payload)
        write_feature_contract(feature_contract_md, format="markdown")
        summary_txt.write_text(self._summary_text(metrics_payload, predictions, skipped), encoding="utf-8")

        return BacktestArtifacts(
            predictions_csv=predictions_csv,
            skipped_csv=skipped_csv,
            model_metrics_csv=model_metrics_csv,
            segments_csv=segments_csv,
            calibration_csv=calibration_csv,
            metrics_json=metrics_json,
            summary_txt=summary_txt,
            feature_contract_md=feature_contract_md,
        )

    @staticmethod
    def _summary_text(
        metrics_payload: dict[str, Any],
        predictions: pd.DataFrame,
        skipped: pd.DataFrame,
    ) -> str:
        rankings = metrics_payload["rankings"]
        lines = [
            "Time-Machine Backtest Summary",
            "",
            f"Mode: {metrics_payload['mode']}",
            f"Historical window: {metrics_payload['window']}",
            f"Matches total: {metrics_payload['matches_total']}",
            f"Matches scored: {metrics_payload['matches_scored']}",
            f"Matches skipped: {metrics_payload['matches_skipped']}",
            f"Coverage: {metrics_payload['coverage']:.4f}",
            f"Snapshot policy: {', '.join(metrics_payload['snapshot_policies'])}",
            "",
            f"Best calibrated log-loss model: {rankings['best_calibrated_log_loss']}",
            f"Best calibrated accuracy model: {rankings['best_calibrated_accuracy']}",
            "",
        ]

        for model_name, probability_metrics in metrics_payload["overall"].items():
            calibrated = probability_metrics["calibrated"]
            raw = probability_metrics["raw"]
            lines.extend(
                [
                    f"{model_name}: calibrated accuracy={calibrated['accuracy']:.4f}, "
                    f"calibrated log-loss={calibrated['log_loss']:.4f}, "
                    f"raw log-loss={raw['log_loss']:.4f}",
                    f"{model_name} calibration: {metrics_payload['calibration_summary'][model_name]['calibrated']}",
                ]
            )

        if not predictions.empty and "round_bucket" in predictions.columns:
            best_round = predictions.groupby("round_bucket")["pcsp_correct"].mean().sort_values(ascending=False).head(1)
            worst_round = predictions.groupby("round_bucket")["pcsp_correct"].mean().sort_values(ascending=True).head(1)
            if not best_round.empty:
                lines.append(
                    f"Best PCSP round bucket: {best_round.index[0]} ({float(best_round.iloc[0]):.4f} accuracy)"
                )
            if not worst_round.empty:
                lines.append(
                    f"Worst PCSP round bucket: {worst_round.index[0]} ({float(worst_round.iloc[0]):.4f} accuracy)"
                )

        lines.append("")
        lines.append("Model comparison:")
        for model_name, probability_metrics in metrics_payload["overall"].items():
            calibrated = probability_metrics["calibrated"]
            lines.append(
                f"- {model_name}: support={int(calibrated['support'])}, "
                f"accuracy={calibrated['accuracy']:.4f}, log-loss={calibrated['log_loss']:.4f}"
            )

        if not skipped.empty:
            top_skip = skipped["reason"].value_counts().head(3)
            lines.append("")
            lines.append("Top skip reasons:")
            for reason, count in top_skip.items():
                lines.append(f"- {reason}: {int(count)}")

        return "\n".join(lines) + "\n"
