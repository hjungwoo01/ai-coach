from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


_PARAM_PATTERN = re.compile(
    r"\b(pA_srv_win|pA_rcv_win|serve_mix_A_short|serve_mix_B_short|"
    r"rally_style_A_attack|rally_style_B_attack|rally_style_A_safe|rally_style_B_safe)\b"
    r"\s*[:=]\s*([+-]?(?:\d*\.\d+|\d+))",
    flags=re.IGNORECASE,
)


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def mock_probability(params: dict[str, Any]) -> float:
    """Deterministic monotonic mapping from rally params to match win probability."""

    p_a_srv = float(params.get("pA_srv_win", 0.5))
    p_a_rcv = float(params.get("pA_rcv_win", 0.5))
    serve_edge = float(params.get("serve_mix_A_short", 0.5)) - float(params.get("serve_mix_B_short", 0.5))
    attack_edge = float(params.get("rally_style_A_attack", 0.33)) - float(params.get("rally_style_B_attack", 0.33))
    safe_edge = float(params.get("rally_style_B_safe", 0.33)) - float(params.get("rally_style_A_safe", 0.33))

    linear = (
        2.8 * (p_a_srv - 0.5)
        + 2.2 * (p_a_rcv - 0.5)
        + 0.7 * serve_edge
        + 0.9 * attack_edge
        - 0.6 * safe_edge
    )
    probability = _logistic(linear)
    return max(0.01, min(0.99, probability))


def _extract_params_from_pcsp(pcsp_text: str) -> dict[str, float]:
    params: dict[str, float] = {}
    for match in _PARAM_PATTERN.finditer(pcsp_text):
        key = match.group(1)
        value = float(match.group(2))
        params[key] = value
    return params


def mock_run(pcsp_path: Path, out_path: Path) -> dict[str, Any]:
    pcsp_text = pcsp_path.read_text(encoding="utf-8", errors="replace")
    params = _extract_params_from_pcsp(pcsp_text)
    probability = mock_probability(params)

    out_text = (
        "PAT Mock Verification Result\n"
        f"with prob {probability:.6f}\n"
        "module: -pcsp\n"
    )
    out_path.write_text(out_text, encoding="utf-8")

    stdout = f"[mock] PAT finished for {pcsp_path.name}. Probability = {probability:.6f}\n"
    return {
        "ok": True,
        "returncode": 0,
        "cmd": ["mock_pat", "-pcsp", str(pcsp_path), str(out_path)],
        "stdout": stdout,
        "stderr": "",
        "probability": probability,
        "pat_out_path": str(out_path),
    }
