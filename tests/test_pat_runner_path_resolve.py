from __future__ import annotations

from pathlib import Path

from coach.pat.runner import resolve_pat_console_path


def test_resolve_pat_console_path_accepts_directory_with_pat3_console(tmp_path: Path) -> None:
    pat_dir = tmp_path / "pat"
    pat_dir.mkdir(parents=True, exist_ok=True)
    pat3 = pat_dir / "PAT3.Console.exe"
    pat3.write_text("", encoding="utf-8")

    resolved = resolve_pat_console_path(pat_dir)
    assert resolved == pat3


def test_resolve_pat_console_path_prefers_pat_console_when_multiple_exist(tmp_path: Path) -> None:
    pat_dir = tmp_path / "pat"
    pat_dir.mkdir(parents=True, exist_ok=True)
    pat = pat_dir / "PAT.Console.exe"
    pat3 = pat_dir / "PAT3.Console.exe"
    pat.write_text("", encoding="utf-8")
    pat3.write_text("", encoding="utf-8")

    resolved = resolve_pat_console_path(pat_dir)
    assert resolved == pat
