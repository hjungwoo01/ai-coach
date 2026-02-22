from __future__ import annotations

import json
from pathlib import Path

from coach.pat.parser import parse_probability, read_pat_output
from coach.pat.runner import run_pat


def test_run_pat_mock_writes_artifacts_and_summary(tmp_path: Path) -> None:
    pcsp_path = tmp_path / "minimal.pcsp"
    pcsp_path.write_text(
        "// inline params for mock parser\n"
        "pA_srv_win = 0.62\n"
        "pA_rcv_win = 0.57\n"
        "#assert M reaches X with prob;\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "pat_output.txt"
    result = run_pat(
        pcsp_path=pcsp_path,
        out_path=out_path,
        mode="mock",
        pat_console_path=None,
        timeout_s=30,
        use_mono=None,
    )

    assert result["ok"] is True
    assert out_path.exists()
    prob = parse_probability(read_pat_output(out_path))
    assert 0.0 <= prob <= 1.0

    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(summary.get("probability"), float)

    assert (tmp_path / "pat_stdout.txt").exists()
    assert (tmp_path / "pat_stderr.txt").exists()
    assert (tmp_path / "pat_run.json").exists()
