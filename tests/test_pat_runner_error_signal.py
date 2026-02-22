from __future__ import annotations

from coach.pat.runner import _extract_pat_model_error


def test_extract_pat_model_error_parsing_line() -> None:
    stdout = "Parsing Error: At least one expression is needed in the expression block.\nPAT finished successfully."
    stderr = ""
    detail = _extract_pat_model_error(stdout=stdout, stderr=stderr)
    assert detail is not None
    assert "Parsing Error:" in detail


def test_extract_pat_model_error_none_for_clean_output() -> None:
    stdout = "PAT finished successfully.\n"
    stderr = ""
    assert _extract_pat_model_error(stdout=stdout, stderr=stderr) is None
