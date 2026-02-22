from __future__ import annotations

from pathlib import Path

from coach.pat.runner import _resolve_use_mono, _should_try_pat3_mono_compat_fallback


def test_resolve_use_mono_auto_uses_mono_for_exe_paths() -> None:
    assert _resolve_use_mono(use_mono=None, pat_console_path=Path("PAT.Console.exe")) is True


def test_resolve_use_mono_respects_explicit_override() -> None:
    assert _resolve_use_mono(use_mono=False, pat_console_path=Path("PAT.Console.exe")) is False


def test_pat3_mono_compat_fallback_triggers_on_nesc_startup_signature(tmp_path: Path) -> None:
    out_path = tmp_path / "pat_output.txt"
    stdout = "For all modules except UML:\nInvalid arguments. invalid image"

    assert (
        _should_try_pat3_mono_compat_fallback(
            pat_console_path=Path("PAT3.Console.exe"),
            use_mono=True,
            stdout=stdout,
            stderr="",
            out_path=out_path,
        )
        is True
    )
