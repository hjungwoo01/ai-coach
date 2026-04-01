from __future__ import annotations

import difflib
from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PlayerRecord:
    player_id: str
    name: str
    country: str | None = None
    handedness: str | None = None


class LocalCSVAdapter:
    """Local CSV-backed stats adapter used by tools and CLI."""

    def __init__(
        self,
        players_path: str | Path | None = None,
        matches_path: str | Path | None = None,
        laplace_alpha: float = 2.0,
    ) -> None:
        base = Path(__file__).resolve().parents[1]
        self.players_path = Path(players_path) if players_path else base / "sample_players.csv"
        self.matches_path = Path(matches_path) if matches_path else base / "sample_matches.csv"
        self.laplace_alpha = laplace_alpha

        self._players_df: pd.DataFrame | None = None
        self._matches_df: pd.DataFrame | None = None
        self._players_lookup_df: pd.DataFrame | None = None
        self._player_params_cache: dict[tuple[str, int, str | None, str], dict[str, Any]] = {}
        self._head_to_head_cache: dict[tuple[str, str, int, str | None], dict[str, Any]] = {}
        self._global_prior_cache: dict[str | None, dict[str, Any]] = {}

    @property
    def players_df(self) -> pd.DataFrame:
        if self._players_df is None:
            self._players_df = pd.read_csv(self.players_path)
        return self._players_df

    @property
    def players_lookup_df(self) -> pd.DataFrame:
        if self._players_lookup_df is None:
            players = self.players_df.copy()
            players["_norm"] = players["name"].map(self._normalize_name)
            self._players_lookup_df = players
        return self._players_lookup_df

    @property
    def matches_df(self) -> pd.DataFrame:
        if self._matches_df is None:
            df = pd.read_csv(self.matches_path)
            df["date"] = pd.to_datetime(df["date"], utc=False)
            df = df.sort_values("date")
            self._matches_df = df.reset_index(drop=True)
        return self._matches_df

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.lower().strip().split())

    def resolve_player(self, name: str) -> PlayerRecord:
        normalized = self._normalize_name(name)
        players = self.players_lookup_df

        exact = players[players["_norm"] == normalized]
        if not exact.empty:
            row = exact.iloc[0]
            return PlayerRecord(
                player_id=str(row["player_id"]),
                name=str(row["name"]),
                country=str(row.get("country", "")) or None,
                handedness=str(row.get("handedness", "")) or None,
            )

        contains = players[players["_norm"].str.contains(normalized, regex=False)]
        if len(contains) == 1:
            row = contains.iloc[0]
            return PlayerRecord(
                player_id=str(row["player_id"]),
                name=str(row["name"]),
                country=str(row.get("country", "")) or None,
                handedness=str(row.get("handedness", "")) or None,
            )

        candidates = players["name"].tolist()
        close = difflib.get_close_matches(name, candidates, n=3, cutoff=0.55)
        if close:
            raise ValueError(f"Player '{name}' not found. Did you mean: {', '.join(close)}?")
        raise ValueError(f"Player '{name}' not found in local player table.")

    def _window_filter(self, window: int, as_of_date: str | None = None) -> pd.DataFrame:
        df = self.matches_df
        if as_of_date:
            cutoff = pd.to_datetime(as_of_date)
            df = df[df["date"] <= cutoff]
        return df.tail(max(window * 4, window))

    def get_player_matches(self, player_id: str, window: int = 30, as_of_date: str | None = None) -> pd.DataFrame:
        df = self._window_filter(window=window, as_of_date=as_of_date)
        mask = (df["playerA_id"] == player_id) | (df["playerB_id"] == player_id)
        player_df = df.loc[mask].copy().sort_values("date")
        if player_df.empty:
            raise ValueError(f"No matches found for player_id='{player_id}'.")
        return player_df.tail(window)

    @staticmethod
    def _series_or_default(df: pd.DataFrame, column: str, default: float | int | str) -> pd.Series:
        if column in df.columns:
            return df[column]
        return pd.Series(default, index=df.index)

    @classmethod
    def _select_side_series(
        cls,
        df: pd.DataFrame,
        *,
        player_is_a: pd.Series,
        a_column: str,
        b_column: str,
        default: float | int | str,
    ) -> pd.Series:
        a_values = cls._series_or_default(df, a_column, default)
        b_values = cls._series_or_default(df, b_column, default)
        return a_values.where(player_is_a, b_values)

    def _perspective_frame(self, df: pd.DataFrame, player_id: str) -> pd.DataFrame:
        player_is_a = df["playerA_id"] == player_id

        serve_rallies = self._select_side_series(
            df,
            player_is_a=player_is_a,
            a_column="a_serve_rallies",
            b_column="b_serve_rallies",
            default=0,
        )
        receive_rallies = self._select_side_series(
            df,
            player_is_a=player_is_a,
            a_column="b_serve_rallies",
            b_column="a_serve_rallies",
            default=0,
        )
        opponent_serve_wins = self._select_side_series(
            df,
            player_is_a=player_is_a,
            a_column="b_serve_wins",
            b_column="a_serve_wins",
            default=0,
        )
        won = (df["winner_id"] == df["playerA_id"]).where(player_is_a, df["winner_id"] == df["playerB_id"])

        perspective = pd.DataFrame(
            {
                "date": df["date"],
                "player_id": player_id,
                "opponent_id": df["playerB_id"].where(player_is_a, df["playerA_id"]),
                "serve_rallies": serve_rallies.astype(int),
                "serve_wins": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_serve_wins",
                    b_column="b_serve_wins",
                    default=0,
                ).astype(int),
                "receive_rallies": receive_rallies.astype(int),
                "receive_wins": (receive_rallies - opponent_serve_wins).astype(int),
                "short_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_short_serve_rate",
                    b_column="b_short_serve_rate",
                    default=0.5,
                ).astype(float),
                "flick_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_flick_serve_rate",
                    b_column="b_flick_serve_rate",
                    default=0.5,
                ).astype(float),
                "attack_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_attack_rate",
                    b_column="b_attack_rate",
                    default=1 / 3,
                ).astype(float),
                "neutral_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_neutral_rate",
                    b_column="b_neutral_rate",
                    default=1 / 3,
                ).astype(float),
                "safe_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_safe_rate",
                    b_column="b_safe_rate",
                    default=1 / 3,
                ).astype(float),
                "points_for": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_points",
                    b_column="b_points",
                    default=0,
                ).astype(int),
                "points_against": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="b_points",
                    b_column="a_points",
                    default=0,
                ).astype(int),
                "avg_rally_len": self._series_or_default(df, "avg_rally_len", 0.0).astype(float),
                "long_rally_share": self._series_or_default(df, "long_rally_share", 0.0).astype(float),
                "short_serve_win_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_short_serve_win_rate",
                    b_column="b_short_serve_win_rate",
                    default=0.5,
                ).astype(float),
                "long_serve_win_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_long_serve_win_rate",
                    b_column="b_long_serve_win_rate",
                    default=0.5,
                ).astype(float),
                "short_serve_samples": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_short_serve_samples",
                    b_column="b_short_serve_samples",
                    default=0,
                ).astype(int),
                "long_serve_samples": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_long_serve_samples",
                    b_column="b_long_serve_samples",
                    default=0,
                ).astype(int),
                "backhand_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_backhand_rate",
                    b_column="b_backhand_rate",
                    default=0.0,
                ).astype(float),
                "aroundhead_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_aroundhead_rate",
                    b_column="b_aroundhead_rate",
                    default=0.0,
                ).astype(float),
                "net_error_lost_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_net_error_lost_rate",
                    b_column="b_net_error_lost_rate",
                    default=0.0,
                ).astype(float),
                "out_error_lost_rate": self._select_side_series(
                    df,
                    player_is_a=player_is_a,
                    a_column="a_out_error_lost_rate",
                    b_column="b_out_error_lost_rate",
                    default=0.0,
                ).astype(float),
                "won": won.astype(int),
            }
        )
        return perspective.sort_values("date").reset_index(drop=True)

    def _smooth_probability(self, wins: float, trials: float, alpha: float | None = None) -> float:
        alpha = self.laplace_alpha if alpha is None else alpha
        return float((wins + alpha) / (trials + 2.0 * alpha))

    @staticmethod
    def _resolve_reference_time(
        perspective: pd.DataFrame,
        as_of_date: str | None,
    ) -> pd.Timestamp | None:
        if as_of_date:
            return pd.Timestamp(as_of_date)
        if perspective.empty:
            return None
        return pd.Timestamp(perspective["date"].max()) + pd.Timedelta(days=1)

    @classmethod
    def _recency_weights(
        cls,
        perspective: pd.DataFrame,
        *,
        as_of_date: str | None,
        half_life_days: float = 45.0,
    ) -> pd.Series:
        if perspective.empty:
            return pd.Series(dtype=float)

        reference_time = cls._resolve_reference_time(perspective, as_of_date)
        if reference_time is None:
            return pd.Series(1.0, index=perspective.index, dtype=float)

        days_ago = (reference_time - pd.to_datetime(perspective["date"], utc=False)).dt.days.clip(lower=0)
        decay = float(half_life_days / max(half_life_days, 1.0))
        weights = 0.5 ** (days_ago / max(half_life_days, 1.0))
        return pd.Series(weights * decay, index=perspective.index, dtype=float)

    @staticmethod
    def _weighted_ratio(numerator: pd.Series, denominator: pd.Series) -> float:
        total = float(denominator.sum())
        if total <= 0.0:
            return 0.0
        return float(numerator.sum() / total)

    def _build_global_priors(self, *, as_of_date: str | None = None) -> dict[str, Any]:
        cached = self._global_prior_cache.get(as_of_date)
        if cached is not None:
            return deepcopy(cached)

        df = self.matches_df
        if as_of_date:
            cutoff = pd.to_datetime(as_of_date, utc=False)
            df = df[df["date"] <= cutoff]

        if df.empty:
            priors = {
                "base_srv_win": 0.5,
                "base_rcv_win": 0.5,
                "unforced_error_rate": 0.18,
                "return_pressure": 0.5,
                "clutch_point_win": 0.5,
                "serve_mix": {"short": 0.5, "flick": 0.5},
                "rally_style": {"attack": 1 / 3, "neutral": 1 / 3, "safe": 1 / 3},
                "short_serve_skill": 0.5,
                "long_serve_skill": 0.5,
                "rally_tolerance": 0.5,
                "net_error_rate": 0.1,
                "out_error_rate": 0.1,
                "backhand_rate": 0.15,
                "aroundhead_rate": 0.08,
                "reliability": 0.0,
                "recent_form": 0.5,
                "rest_days": 7.0,
            }
            self._global_prior_cache[as_of_date] = deepcopy(priors)
            return deepcopy(priors)

        serve_trials = float(df["a_serve_rallies"].sum() + df["b_serve_rallies"].sum())
        serve_wins = float(df["a_serve_wins"].sum() + df["b_serve_wins"].sum())
        receive_trials = serve_trials
        receive_wins = float(
            (df["a_serve_rallies"] - df["a_serve_wins"]).sum() + (df["b_serve_rallies"] - df["b_serve_wins"]).sum()
        )

        serve_weight_a = df["a_serve_rallies"].clip(lower=1)
        serve_weight_b = df["b_serve_rallies"].clip(lower=1)
        rally_weight_a = (df["a_points"] + df["b_points"]).clip(lower=1)
        rally_weight_b = rally_weight_a

        short_a = df["a_short_serve_rate"] * serve_weight_a
        short_b = df["b_short_serve_rate"] * serve_weight_b
        short_total = float(short_a.sum() + short_b.sum())
        serve_weight_total = float(serve_weight_a.sum() + serve_weight_b.sum())
        short_rate = short_total / max(serve_weight_total, 1.0)
        short_rate = (short_rate + 0.02) / (1.0 + 2.0 * 0.02)

        attack = float(
            ((df["a_attack_rate"] * rally_weight_a).sum() + (df["b_attack_rate"] * rally_weight_b).sum())
            / max(float(rally_weight_a.sum() + rally_weight_b.sum()), 1.0)
        )
        safe = float(
            ((df["a_safe_rate"] * rally_weight_a).sum() + (df["b_safe_rate"] * rally_weight_b).sum())
            / max(float(rally_weight_a.sum() + rally_weight_b.sum()), 1.0)
        )
        attack = max(0.05, min(0.9, attack))
        safe = max(0.05, min(0.9, safe))
        neutral = max(0.05, 1.0 - attack - safe)
        total_style = attack + neutral + safe
        attack, neutral, safe = attack / total_style, neutral / total_style, safe / total_style

        ue_a = self._estimate_unforced_error_proxy(
            attack_rate=df["a_attack_rate"],
            safe_rate=df["a_safe_rate"],
            flick_rate=df["a_flick_serve_rate"],
            points_for=df["a_points"],
            points_against=df["b_points"],
        )
        ue_b = self._estimate_unforced_error_proxy(
            attack_rate=df["b_attack_rate"],
            safe_rate=df["b_safe_rate"],
            flick_rate=df["b_flick_serve_rate"],
            points_for=df["b_points"],
            points_against=df["a_points"],
        )
        rally_weight_total = float(rally_weight_a.sum() + rally_weight_b.sum())
        unforced_error_rate = float(
            ((ue_a * rally_weight_a).sum() + (ue_b * rally_weight_b).sum()) / max(rally_weight_total, 1.0)
        )

        point_share_a = df["a_points"] / (df["a_points"] + df["b_points"]).clip(lower=1)
        point_share_b = df["b_points"] / (df["a_points"] + df["b_points"]).clip(lower=1)
        recent_form = float((point_share_a.mean() + point_share_b.mean()) / 2.0)

        priors = {
            "base_srv_win": self._smooth_probability(serve_wins, serve_trials),
            "base_rcv_win": self._smooth_probability(receive_wins, receive_trials),
            "unforced_error_rate": unforced_error_rate,
            "return_pressure": float(
                min(
                    0.99,
                    max(
                        0.01,
                        0.58 * self._smooth_probability(receive_wins, receive_trials)
                        + 0.22 * attack
                        + 0.20 * recent_form,
                    ),
                )
            ),
            "clutch_point_win": float(min(0.99, max(0.01, recent_form))),
            "serve_mix": {"short": short_rate, "flick": 1.0 - short_rate},
            "rally_style": {"attack": attack, "neutral": neutral, "safe": safe},
            "short_serve_skill": self._smooth_probability(
                float((df["a_short_serve_win_rate"] * df["a_short_serve_samples"]).sum())
                + float((df["b_short_serve_win_rate"] * df["b_short_serve_samples"]).sum()),
                float(df["a_short_serve_samples"].sum() + df["b_short_serve_samples"].sum()),
                alpha=1.2,
            ),
            "long_serve_skill": self._smooth_probability(
                float((df["a_long_serve_win_rate"] * df["a_long_serve_samples"]).sum())
                + float((df["b_long_serve_win_rate"] * df["b_long_serve_samples"]).sum()),
                float(df["a_long_serve_samples"].sum() + df["b_long_serve_samples"].sum()),
                alpha=1.2,
            ),
            "rally_tolerance": float(
                min(
                    0.99,
                    max(
                        0.01,
                        0.45
                        * (
                            (
                                float((df["avg_rally_len"] * rally_weight_a).sum())
                                + float((df["avg_rally_len"] * rally_weight_b).sum())
                            )
                            / max(rally_weight_total, 1.0)
                        )
                        / 12.0
                        + 0.55
                        * (
                            (
                                float((df["long_rally_share"] * rally_weight_a).sum())
                                + float((df["long_rally_share"] * rally_weight_b).sum())
                            )
                            / max(rally_weight_total, 1.0)
                        ),
                    ),
                )
            ),
            "net_error_rate": float((df["a_net_error_lost_rate"].mean() + df["b_net_error_lost_rate"].mean()) / 2.0),
            "out_error_rate": float((df["a_out_error_lost_rate"].mean() + df["b_out_error_lost_rate"].mean()) / 2.0),
            "backhand_rate": float((df["a_backhand_rate"].mean() + df["b_backhand_rate"].mean()) / 2.0),
            "aroundhead_rate": float((df["a_aroundhead_rate"].mean() + df["b_aroundhead_rate"].mean()) / 2.0),
            "reliability": 0.15,
            "recent_form": float(min(0.99, max(0.01, recent_form))),
            "rest_days": 7.0,
        }
        self._global_prior_cache[as_of_date] = deepcopy(priors)
        return deepcopy(priors)

    def _cold_start_player_params(self, player_id: str, *, as_of_date: str | None = None) -> dict[str, Any]:
        priors = self._build_global_priors(as_of_date=as_of_date)
        player_row = self.players_df[self.players_df["player_id"] == player_id].iloc[0]
        handedness = str(player_row.get("handedness", "")).upper()

        return {
            "player_id": player_id,
            "name": str(player_row["name"]),
            "matches": 0,
            "win_rate": 0.5,
            "base_srv_win": priors["base_srv_win"],
            "base_rcv_win": priors["base_rcv_win"],
            "unforced_error_rate": priors["unforced_error_rate"],
            "return_pressure": priors["return_pressure"],
            "clutch_point_win": priors["clutch_point_win"],
            "serve_mix": deepcopy(priors["serve_mix"]),
            "rally_style": deepcopy(priors["rally_style"]),
            "short_serve_skill": priors["short_serve_skill"],
            "long_serve_skill": priors["long_serve_skill"],
            "rally_tolerance": priors["rally_tolerance"],
            "net_error_rate": priors["net_error_rate"],
            "out_error_rate": priors["out_error_rate"],
            "backhand_rate": priors["backhand_rate"],
            "aroundhead_rate": priors["aroundhead_rate"],
            "handedness_flag": 1.0 if handedness == "L" else 0.0,
            "reliability": 0.0,
            "serve_trials": 0,
            "receive_trials": 0,
            "recent_form": priors["recent_form"],
            "rest_days": priors["rest_days"],
        }

    @staticmethod
    def _blend_metric(value: float, prior: float, sample_size: int, prior_strength: float = 8.0) -> float:
        weight = sample_size / (sample_size + prior_strength) if sample_size > 0 else 0.0
        return float((weight * value) + ((1.0 - weight) * prior))

    @staticmethod
    def _estimate_unforced_error_proxy(
        *,
        attack_rate: float | pd.Series,
        safe_rate: float | pd.Series,
        flick_rate: float | pd.Series,
        points_for: float | pd.Series,
        points_against: float | pd.Series,
    ) -> float | pd.Series:
        total = points_for + points_against
        total = total.clip(lower=1.0) if isinstance(total, pd.Series) else max(total, 1.0)
        point_loss = points_against / total
        proxy = (
            0.08
            + 0.22 * attack_rate
            + 0.08 * flick_rate
            + 0.11 * point_loss
            - 0.09 * safe_rate
        )
        if isinstance(proxy, pd.Series):
            return proxy.clip(lower=0.01, upper=0.6)
        return float(min(0.6, max(0.01, proxy)))

    def get_player_params(
        self,
        player_id: str,
        window: int = 30,
        as_of_date: str | None = None,
        allow_cold_start: bool = False,
    ) -> dict[str, Any]:
        cache_key = (player_id, window, as_of_date, "cold" if allow_cold_start else "strict")
        cached = self._player_params_cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached)

        try:
            raw = self.get_player_matches(player_id=player_id, window=window, as_of_date=as_of_date)
        except ValueError:
            if not allow_cold_start:
                raise
            result = self._cold_start_player_params(player_id, as_of_date=as_of_date)
            self._player_params_cache[cache_key] = deepcopy(result)
            return deepcopy(result)

        perspective = self._perspective_frame(raw, player_id=player_id)
        recency = self._recency_weights(perspective, as_of_date=as_of_date)
        priors = self._build_global_priors(as_of_date=as_of_date)
        reference_time = self._resolve_reference_time(perspective, as_of_date)

        weighted_serve_trials = perspective["serve_rallies"] * recency
        weighted_receive_trials = perspective["receive_rallies"] * recency
        serve_trials = float(perspective["serve_rallies"].sum())
        serve_wins = float((perspective["serve_wins"] * recency).sum())
        receive_trials = float(perspective["receive_rallies"].sum())
        receive_wins = float((perspective["receive_wins"] * recency).sum())

        base_srv = self._smooth_probability(serve_wins, float(weighted_serve_trials.sum()))
        base_rcv = self._smooth_probability(receive_wins, float(weighted_receive_trials.sum()))

        serve_weight = (perspective["serve_rallies"].clip(lower=1) * recency).clip(lower=1e-6)
        rally_weight = ((perspective["points_for"] + perspective["points_against"]).clip(lower=1) * recency).clip(
            lower=1e-6
        )

        short = float((perspective["short_rate"] * serve_weight).sum() / serve_weight.sum())
        attack = float((perspective["attack_rate"] * rally_weight).sum() / rally_weight.sum())
        safe = float((perspective["safe_rate"] * rally_weight).sum() / rally_weight.sum())

        # Dirichlet-style smoothing on mix vectors.
        alpha_mix = 0.02
        short = (short + alpha_mix) / (1.0 + 2.0 * alpha_mix)
        flick = 1.0 - short

        attack = max(0.05, min(0.9, attack))
        safe = max(0.05, min(0.9, safe))
        neutral = max(0.05, 1.0 - attack - safe)
        total = attack + neutral + safe
        attack, neutral, safe = attack / total, neutral / total, safe / total

        wins = int(perspective["won"].sum())
        matches = int(len(perspective))
        points_for_total = float(perspective["points_for"].sum())
        points_against_total = float(perspective["points_against"].sum())
        point_share = points_for_total / max(points_for_total + points_against_total, 1.0)
        weighted_point_share = float((perspective["points_for"] * recency).sum()) / max(
            float(((perspective["points_for"] + perspective["points_against"]) * recency).sum()),
            1.0,
        )

        ue_proxy_series = self._estimate_unforced_error_proxy(
            attack_rate=perspective["attack_rate"],
            safe_rate=perspective["safe_rate"],
            flick_rate=perspective["flick_rate"],
            points_for=perspective["points_for"],
            points_against=perspective["points_against"],
        )
        unforced_error_rate = float((ue_proxy_series * rally_weight).sum() / rally_weight.sum())

        return_pressure = float(
            min(
                0.99,
                max(
                    0.01,
                    0.58 * base_rcv + 0.22 * attack + 0.20 * point_share,
                ),
            )
        )

        close_match = (perspective["points_for"] - perspective["points_against"]).abs() <= 6
        close_df = perspective.loc[close_match]
        if close_df.empty:
            clutch_point_win = point_share
        else:
            close_points_for = float(close_df["points_for"].sum())
            close_points_against = float(close_df["points_against"].sum())
            close_point_share = close_points_for / max(close_points_for + close_points_against, 1.0)
            close_win_rate = self._smooth_probability(float(close_df["won"].sum()), float(len(close_df)), alpha=1.0)
            clutch_point_win = 0.65 * close_point_share + 0.35 * close_win_rate
        clutch_point_win = float(min(0.99, max(0.01, clutch_point_win)))

        short_serve_trials = float(perspective["short_serve_samples"].sum())
        long_serve_trials = float(perspective["long_serve_samples"].sum())
        short_serve_wins = float((perspective["short_serve_win_rate"] * perspective["short_serve_samples"]).sum())
        long_serve_wins = float((perspective["long_serve_win_rate"] * perspective["long_serve_samples"]).sum())
        short_serve_skill = self._smooth_probability(short_serve_wins, short_serve_trials, alpha=1.2)
        long_serve_skill = self._smooth_probability(long_serve_wins, long_serve_trials, alpha=1.2)

        avg_rally_len_raw = float((perspective["avg_rally_len"] * rally_weight).sum() / rally_weight.sum())
        avg_rally_len_norm = min(1.0, max(0.0, avg_rally_len_raw / 12.0))
        long_rally_share = float((perspective["long_rally_share"] * rally_weight).sum() / rally_weight.sum())
        rally_tolerance = min(0.99, max(0.01, 0.45 * avg_rally_len_norm + 0.55 * long_rally_share))

        net_error_rate = float((perspective["net_error_lost_rate"] * rally_weight).sum() / rally_weight.sum())
        out_error_rate = float((perspective["out_error_lost_rate"] * rally_weight).sum() / rally_weight.sum())
        backhand_rate = float((perspective["backhand_rate"] * rally_weight).sum() / rally_weight.sum())
        aroundhead_rate = float((perspective["aroundhead_rate"] * rally_weight).sum() / rally_weight.sum())
        net_error_rate = min(1.0, max(0.0, net_error_rate))
        out_error_rate = min(1.0, max(0.0, out_error_rate))
        backhand_rate = min(1.0, max(0.0, backhand_rate))
        aroundhead_rate = min(1.0, max(0.0, aroundhead_rate))
        weighted_win_rate = self._smooth_probability(float((perspective["won"] * recency).sum()), float(recency.sum()), alpha=1.5)
        recent_form = float(min(0.99, max(0.01, (0.55 * weighted_win_rate) + (0.45 * weighted_point_share))))

        if reference_time is None:
            rest_days = priors["rest_days"]
        else:
            last_match_time = pd.Timestamp(perspective["date"].max())
            rest_days = float((reference_time - last_match_time).days)
        rest_days = float(min(90.0, max(0.0, rest_days)))

        player_row = self.players_df[self.players_df["player_id"] == player_id].iloc[0]
        handedness = str(player_row.get("handedness", "")).upper()
        handedness_flag = 1.0 if handedness == "L" else 0.0
        reliability = min(1.0, max(0.0, sqrt(max(0.0, serve_trials + receive_trials) / 80.0)))
        result = {
            "player_id": player_id,
            "name": str(player_row["name"]),
            "matches": matches,
            "win_rate": self._smooth_probability(wins, matches, alpha=1.5),
            "base_srv_win": self._blend_metric(base_srv, priors["base_srv_win"], matches),
            "base_rcv_win": self._blend_metric(base_rcv, priors["base_rcv_win"], matches),
            "unforced_error_rate": self._blend_metric(unforced_error_rate, priors["unforced_error_rate"], matches),
            "return_pressure": self._blend_metric(return_pressure, priors["return_pressure"], matches),
            "clutch_point_win": self._blend_metric(clutch_point_win, priors["clutch_point_win"], matches),
            "serve_mix": {
                "short": self._blend_metric(short, priors["serve_mix"]["short"], matches),
                "flick": self._blend_metric(flick, priors["serve_mix"]["flick"], matches),
            },
            "rally_style": {
                "attack": self._blend_metric(attack, priors["rally_style"]["attack"], matches),
                "neutral": self._blend_metric(neutral, priors["rally_style"]["neutral"], matches),
                "safe": self._blend_metric(safe, priors["rally_style"]["safe"], matches),
            },
            "short_serve_skill": self._blend_metric(short_serve_skill, priors["short_serve_skill"], matches),
            "long_serve_skill": self._blend_metric(long_serve_skill, priors["long_serve_skill"], matches),
            "rally_tolerance": self._blend_metric(rally_tolerance, priors["rally_tolerance"], matches),
            "net_error_rate": self._blend_metric(net_error_rate, priors["net_error_rate"], matches),
            "out_error_rate": self._blend_metric(out_error_rate, priors["out_error_rate"], matches),
            "backhand_rate": self._blend_metric(backhand_rate, priors["backhand_rate"], matches),
            "aroundhead_rate": self._blend_metric(aroundhead_rate, priors["aroundhead_rate"], matches),
            "handedness_flag": handedness_flag,
            "reliability": reliability,
            "serve_trials": int(serve_trials),
            "receive_trials": int(receive_trials),
            "recent_form": self._blend_metric(recent_form, priors["recent_form"], matches),
            "rest_days": rest_days if matches > 0 else priors["rest_days"],
        }

        rally_style_total = sum(float(result["rally_style"][name]) for name in ("attack", "neutral", "safe"))
        if rally_style_total > 0.0:
            for key in ("attack", "neutral", "safe"):
                result["rally_style"][key] = float(result["rally_style"][key]) / rally_style_total
        serve_short = float(result["serve_mix"]["short"])
        serve_short = min(0.99, max(0.01, serve_short))
        result["serve_mix"] = {"short": serve_short, "flick": 1.0 - serve_short}

        self._player_params_cache[cache_key] = deepcopy(result)
        return deepcopy(result)

    def get_head_to_head(
        self,
        player_a_id: str,
        player_b_id: str,
        window: int = 30,
        as_of_date: str | None = None,
    ) -> dict[str, Any]:
        cache_key = (player_a_id, player_b_id, window, as_of_date)
        cached = self._head_to_head_cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached)

        df = self._window_filter(window=window * 2, as_of_date=as_of_date)
        h2h = df[
            ((df["playerA_id"] == player_a_id) & (df["playerB_id"] == player_b_id))
            | ((df["playerA_id"] == player_b_id) & (df["playerB_id"] == player_a_id))
        ].copy()

        if h2h.empty:
            result = {
                "matches": 0,
                "a_win_rate": 0.5,
                "a_srv_win": 0.5,
                "a_rcv_win": 0.5,
            }
            self._head_to_head_cache[cache_key] = deepcopy(result)
            return deepcopy(result)

        perspective = self._perspective_frame(h2h, player_id=player_a_id)
        matches = int(len(perspective))
        wins = int(perspective["won"].sum())

        srv_trials = float(perspective["serve_rallies"].sum())
        srv_wins = float(perspective["serve_wins"].sum())
        rcv_trials = float(perspective["receive_rallies"].sum())
        rcv_wins = float(perspective["receive_wins"].sum())

        result = {
            "matches": matches,
            "a_win_rate": self._smooth_probability(wins, matches, alpha=1.0),
            "a_srv_win": self._smooth_probability(srv_wins, srv_trials, alpha=1.5),
            "a_rcv_win": self._smooth_probability(rcv_wins, rcv_trials, alpha=1.5),
        }
        self._head_to_head_cache[cache_key] = deepcopy(result)
        return deepcopy(result)
