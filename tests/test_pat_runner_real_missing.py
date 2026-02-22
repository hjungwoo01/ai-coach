from __future__ import annotations

from pathlib import Path

import pytest

from coach.pat.runner import run_pat


def test_run_pat_real_missing_executable_has_clear_error(tmp_path: Path) -> None:
    pcsp_path = tmp_path / "minimal.pcsp"
    pcsp_path.write_text("#assert M reaches X with prob;\n", encoding="utf-8")
    out_path = tmp_path / "pat_output.txt"

    missing = tmp_path / "does_not_exist" / "PAT.Console.exe"

    with pytest.raises(RuntimeError) as err:
        run_pat(
            pcsp_path=pcsp_path,
            out_path=out_path,
            mode="real",
            pat_console_path=missing,
            timeout_s=30,
            use_mono=None,
        )

    message = str(err.value)
    assert "PAT Console not found" in message
    assert "PAT_CONSOLE_PATH" in message
