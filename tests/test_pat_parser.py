from __future__ import annotations

import pytest

from coach.pat.parser import parse_probability


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Probability = 0.123", 0.123),
        ("probability: 0.456", 0.456),
        ("The assertion holds with prob 0.789", 0.789),
        ("The Assertion is Valid with Probability [0.6125, 0.6125];", 0.6125),
        ("Probability = 6.0e-1", 0.6),
    ],
)
def test_parse_probability_variants(text: str, expected: float) -> None:
    assert abs(parse_probability(text) - expected) < 1e-9


def test_parse_probability_uses_last_contextual_match() -> None:
    text = "score=21\nprobability: 0.21\nother stats\nwith prob 0.84\n"
    assert abs(parse_probability(text) - 0.84) < 1e-9


def test_parse_probability_ignores_non_probability_numbers_on_later_lines() -> None:
    text = (
        "The Assertion is Valid with Probability [0.6, 0.6];\n"
        "Maximum difference threshold : 1E-06\n"
    )
    assert abs(parse_probability(text) - 0.6) < 1e-9


def test_parse_probability_failure_has_excerpt() -> None:
    with pytest.raises(ValueError) as err:
        parse_probability("No probability keyword exists in this output")
    assert "Could not parse probability" in str(err.value)
    assert "Excerpt" in str(err.value)
