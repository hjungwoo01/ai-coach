from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from coach.model.params import MatchupParams
from coach.utils import clamp


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def extract_recent_form_feature_vector(params: MatchupParams) -> np.ndarray:
    a = params.player_a
    b = params.player_b
    serve_skill_edge = 0.5 * (a.short_serve_skill - b.short_serve_skill) + 0.5 * (
        a.long_serve_skill - b.long_serve_skill
    )
    return np.asarray(
        [
            a.base_srv_win - b.base_srv_win,
            a.base_rcv_win - b.base_rcv_win,
            a.recent_form - b.recent_form,
            clamp((a.rest_days - b.rest_days) / 14.0, low=-1.0, high=1.0),
            b.unforced_error_rate - a.unforced_error_rate,
            a.return_pressure - b.return_pressure,
            a.clutch_point_win - b.clutch_point_win,
            serve_skill_edge,
            a.rally_style.attack - b.rally_style.attack,
            b.rally_style.safe - a.rally_style.safe,
            b.backhand_rate - a.backhand_rate,
            a.aroundhead_rate - b.aroundhead_rate,
        ],
        dtype=float,
    )


@dataclass
class EloRatingModel:
    default_rating: float = 1500.0
    k_factor: float = 24.0
    scale: float = 400.0
    ratings: dict[str, float] = field(default_factory=dict)

    def predict(self, player_a_id: str, player_b_id: str) -> float:
        rating_a = self.ratings.get(player_a_id, self.default_rating)
        rating_b = self.ratings.get(player_b_id, self.default_rating)
        exponent = -(rating_a - rating_b) / self.scale
        return float(1.0 / (1.0 + (10.0 ** exponent)))

    def update(self, player_a_id: str, player_b_id: str, actual_a_win: int) -> None:
        expected_a = self.predict(player_a_id, player_b_id)
        expected_b = 1.0 - expected_a
        rating_a = self.ratings.get(player_a_id, self.default_rating)
        rating_b = self.ratings.get(player_b_id, self.default_rating)

        score_a = float(actual_a_win)
        score_b = 1.0 - score_a
        self.ratings[player_a_id] = rating_a + self.k_factor * (score_a - expected_a)
        self.ratings[player_b_id] = rating_b + self.k_factor * (score_b - expected_b)


@dataclass
class OnlineRecentFormLogisticModel:
    learning_rate: float = 0.15
    l2_penalty: float = 0.02
    feature_count: int = 12
    weights: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=float))
    bias: float = 0.0

    def predict(self, features: np.ndarray) -> float:
        score = float(np.dot(self.weights, features) + self.bias)
        return _sigmoid(score)

    def update(self, features: np.ndarray, actual_a_win: int) -> None:
        prediction = self.predict(features)
        error = prediction - float(actual_a_win)
        self.weights -= self.learning_rate * ((error * features) + (self.l2_penalty * self.weights))
        self.bias -= self.learning_rate * error


@dataclass
class RollingPlattCalibrator:
    window_size: int = 200
    min_samples: int = 20
    learning_rate: float = 0.05
    max_iterations: int = 250
    _history: deque[tuple[float, int]] = field(default_factory=deque)
    _dirty: bool = True
    _coef: float = 1.0
    _intercept: float = 0.0

    def update(self, raw_probability: float, actual_a_win: int) -> None:
        self._history.append((float(raw_probability), int(actual_a_win)))
        while len(self._history) > self.window_size:
            self._history.popleft()
        self._dirty = True

    def transform(self, raw_probability: float) -> float:
        if len(self._history) < self.min_samples:
            return float(raw_probability)
        if self._dirty:
            self._fit()
        calibrated = _sigmoid((self._coef * _logit(raw_probability)) + self._intercept)
        return float(min(max(calibrated, 1e-6), 1 - 1e-6))

    def _fit(self) -> None:
        if len(self._history) < self.min_samples:
            self._coef = 1.0
            self._intercept = 0.0
            self._dirty = False
            return

        xs = np.asarray([_logit(prob) for prob, _ in self._history], dtype=float)
        ys = np.asarray([label for _, label in self._history], dtype=float)
        coef = self._coef
        intercept = self._intercept

        for _ in range(self.max_iterations):
            logits = (coef * xs) + intercept
            preds = np.asarray([_sigmoid(value) for value in logits], dtype=float)
            error = preds - ys
            grad_coef = float(np.mean(error * xs))
            grad_intercept = float(np.mean(error))
            coef -= self.learning_rate * grad_coef
            intercept -= self.learning_rate * grad_intercept

        self._coef = coef
        self._intercept = intercept
        self._dirty = False


def summarize_model_rankings(
    metrics_by_model: dict[str, dict[str, float | int]],
    *,
    sort_key: str,
    ascending: bool,
) -> list[tuple[str, dict[str, float | int]]]:
    return sorted(
        metrics_by_model.items(),
        key=lambda item: float(item[1][sort_key]),
        reverse=not ascending,
    )
