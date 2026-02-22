from __future__ import annotations

import re
from pathlib import Path


_PROB_CONTEXT_PATTERN = re.compile(r"(?is)(?:prob(?:ability)?|with\s+prob)")
_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?")


def parse_probability(text: str) -> float:
    """Parse probability from PAT textual output.

    Accepts strings such as:
    - "Probability = 0.123"
    - "probability: 0.123"
    - "with prob 0.123"
    - "The Assertion ... Probability [0.123, 0.123]"

    If multiple contextual matches exist, the last one is used.
    """

    values: list[float] = []
    for context_match in _PROB_CONTEXT_PATTERN.finditer(text):
        segment_start = context_match.end()
        line_end = text.find("\n", segment_start)
        if line_end == -1:
            line_end = len(text)

        line_segment = text[segment_start:line_end]
        line_values = _extract_probabilities(line_segment)
        if line_values:
            values.extend(line_values)
            continue

        # Fallback for unusual formatting where the numeric value follows on next characters.
        nearby_segment = text[segment_start : min(len(text), segment_start + 40)]
        values.extend(_extract_probabilities(nearby_segment))

    if values:
        return values[-1]

    raise ValueError(f"Could not parse probability from PAT output. Excerpt: {_excerpt(text)}")


def read_pat_output(path: Path) -> str:
    """Read PAT output text with tolerant decoding."""

    encodings = ("utf-8", "utf-8-sig", "latin-1")
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return path.read_bytes().decode("utf-8", errors="replace")


def _excerpt(text: str, max_len: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _extract_probabilities(segment: str) -> list[float]:
    values: list[float] = []
    for number_match in _NUMBER_PATTERN.finditer(segment):
        value = float(number_match.group(0))
        if 0.0 <= value <= 1.0:
            values.append(value)
    return values
