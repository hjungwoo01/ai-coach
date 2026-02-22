from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from coach.config import CoachConfig
from coach.pat.mock_pat import mock_run
from coach.pat.parser import parse_probability, read_pat_output


def run_pat(
    pcsp_path: Path,
    out_path: Path,
    *,
    mode: Literal["real", "mock"],
    pat_console_path: Path | None,
    timeout_s: int,
    use_mono: bool | None,
) -> dict:
    """Run PAT Console in real or deterministic mock mode."""

    pcsp_path = pcsp_path.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    run_dir = out_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "pat_stdout.txt"
    stderr_path = run_dir / "pat_stderr.txt"
    pat_run_json = run_dir / "pat_run.json"
    summary_json = run_dir / "summary.json"

    if mode == "mock":
        result = mock_run(pcsp_path=pcsp_path, out_path=out_path)
        stdout_path.write_text(str(result.get("stdout", "")), encoding="utf-8")
        stderr_path.write_text(str(result.get("stderr", "")), encoding="utf-8")
        payload = {
            "ok": True,
            "returncode": 0,
            "cmd": result.get("cmd", ["mock_pat"]),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "pat_out_path": str(out_path),
            "probability": result.get("probability"),
        }
        pat_run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_summary(summary_json=summary_json, probability=result.get("probability"))
        return payload

    if mode != "real":
        raise ValueError("mode must be 'real' or 'mock'")

    if pat_console_path is None:
        raise RuntimeError(
            "PAT real mode requires PAT_CONSOLE_PATH.\n"
            "Download PAT from https://www.comp.nus.edu.sg/~pat/patdownload.htm\n"
            "Set PAT_CONSOLE_PATH=/path/to/PAT.Console.exe and retry, or use --mode mock."
        )

    resolved_pat_path = resolve_pat_console_path(Path(pat_console_path))
    if not resolved_pat_path.exists():
        raise RuntimeError(
            f"PAT Console not found at: {resolved_pat_path}\n"
            "Download PAT from https://www.comp.nus.edu.sg/~pat/patdownload.htm\n"
            "Set PAT_CONSOLE_PATH to the PAT.Console.exe path, then retry."
        )
    if resolved_pat_path.is_dir():
        raise RuntimeError(
            f"PAT_CONSOLE_PATH points to a directory without a console executable: {resolved_pat_path}\n"
            "Set PAT_CONSOLE_PATH to PAT.Console.exe or PAT3.Console.exe."
        )

    resolved_use_mono = _resolve_use_mono(use_mono=use_mono, pat_console_path=resolved_pat_path)
    base_cmd = _build_pat_command(
        pat_console_path=resolved_pat_path,
        pcsp_path=pcsp_path,
        out_path=out_path,
        use_mono=resolved_use_mono,
    )

    try:
        attempts: list[_PATCommandResult] = []
        fallback_error: str | None = None

        primary = _run_pat_command(
            cmd=base_cmd,
            timeout_s=timeout_s,
            use_mono=resolved_use_mono,
            cwd=resolved_pat_path.parent,
        )
        attempts.append(primary)

        if _should_try_pat3_mono_compat_fallback(
            pat_console_path=resolved_pat_path,
            use_mono=resolved_use_mono,
            stdout=primary.stdout,
            stderr=primary.stderr,
            out_path=out_path,
        ):
            try:
                compat_console = _prepare_pat3_mono_compat_runtime(
                    pat_console_path=resolved_pat_path,
                    run_dir=run_dir,
                )
                compat_cmd = _build_pat_command(
                    pat_console_path=compat_console,
                    pcsp_path=pcsp_path,
                    out_path=out_path,
                    use_mono=resolved_use_mono,
                )
                compat = _run_pat_command(
                    cmd=compat_cmd,
                    timeout_s=timeout_s,
                    use_mono=resolved_use_mono,
                    cwd=compat_console.parent,
                )
                attempts.append(compat)
            except Exception as exc:
                fallback_error = str(exc)

        attempt_meta = _write_attempt_logs(run_dir=run_dir, attempts=attempts)
        selected = attempts[-1]
        stdout = selected.stdout
        stderr = selected.stderr
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        model_error = _extract_pat_model_error(stdout=stdout, stderr=stderr)
        probability: float | None = None
        if out_path.exists():
            try:
                pat_out_text = read_pat_output(out_path)
            except Exception:
                pat_out_text = ""
            if not pat_out_text.strip():
                if model_error is None:
                    model_error = "PAT produced an empty output file."
            else:
                try:
                    probability = parse_probability(pat_out_text)
                except ValueError as exc:
                    if model_error is None:
                        model_error = str(exc)
        elif selected.returncode == 0 and model_error is None:
            model_error = "PAT did not produce the expected output file."

        ok = selected.returncode == 0 and out_path.exists() and model_error is None and probability is not None

        payload: dict[str, object] = {
            "ok": ok,
            "returncode": selected.returncode,
            "cmd": [str(part) for part in selected.cmd],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "pat_out_path": str(out_path),
            "probability": probability,
            "attempts": attempt_meta,
            "fallback_applied": len(attempts) > 1,
        }
        if not ok:
            message = "PAT execution failed. Check pat_stdout.txt and pat_stderr.txt in the run directory."
            detail = model_error or _first_nonempty_line(stderr) or _first_nonempty_line(stdout)
            if detail:
                message = f"{message} Detail: {detail}"
            extra_hint = _infer_hint_from_output(stdout=stdout, stderr=stderr)
            if extra_hint:
                message = f"{message} Hint: {extra_hint}"
            if fallback_error:
                message = (
                    f"{message} Fallback detail: attempted PAT3 Mono compatibility shim but it failed: "
                    f"{fallback_error}"
                )
            payload["error"] = message

        pat_run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_summary(summary_json=summary_json, probability=probability)
        return payload
    except FileNotFoundError:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        payload = {
            "ok": False,
            "returncode": -1,
            "cmd": [str(part) for part in base_cmd],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "pat_out_path": str(out_path),
            "probability": None,
            "error": (
                "PAT command could not be started. Ensure PAT_CONSOLE_PATH is correct and "
                "Mono is installed and available on PATH when required."
            ),
        }
        pat_run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_summary(summary_json=summary_json, probability=None)
        return payload
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        payload = {
            "ok": False,
            "returncode": -1,
            "cmd": [str(part) for part in base_cmd],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "pat_out_path": str(out_path),
            "probability": None,
            "error": f"PAT timed out after {timeout_s}s",
        }
        pat_run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_summary(summary_json=summary_json, probability=None)
        return payload


def resolve_pat_console_path(pat_console_path: Path) -> Path:
    """Resolve PAT path, allowing either an executable or a PAT directory."""

    expanded = pat_console_path.expanduser()
    if not expanded.exists():
        return expanded.resolve(strict=False)
    if expanded.is_file():
        return expanded.resolve()

    candidates = (
        "PAT.Console.exe",
        "PAT3.Console.exe",
        "PAT4.Console.exe",
    )
    for name in candidates:
        candidate = expanded / name
        if candidate.exists():
            return candidate.resolve()

    globbed = sorted(expanded.glob("*Console*.exe"))
    if len(globbed) == 1:
        return globbed[0].resolve()

    return expanded.resolve()


def _resolve_use_mono(use_mono: bool | None, pat_console_path: Path) -> bool:
    if use_mono is not None:
        return use_mono
    return pat_console_path.suffix.lower() == ".exe"


def _build_pat_command(
    *,
    pat_console_path: Path,
    pcsp_path: Path,
    out_path: Path,
    use_mono: bool,
) -> list[str]:
    if use_mono:
        settings = CoachConfig.from_env()
        return [settings.mono_path, str(pat_console_path), "-pcsp", str(pcsp_path), str(out_path)]
    return [str(pat_console_path), "-pcsp", str(pcsp_path), str(out_path)]


@dataclass(frozen=True)
class _PATCommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


def _run_pat_command(
    *,
    cmd: list[str],
    timeout_s: int,
    use_mono: bool,
    cwd: Path,
) -> _PATCommandResult:
    proc_env = os.environ.copy()
    if use_mono:
        proc_env.pop("MONO_PATH", None)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env=proc_env,
        cwd=str(cwd),
    )
    return _PATCommandResult(
        cmd=[str(part) for part in cmd],
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def _write_attempt_logs(run_dir: Path, attempts: list[_PATCommandResult]) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    for idx, attempt in enumerate(attempts, start=1):
        out = run_dir / f"pat_stdout_attempt{idx}.txt"
        err = run_dir / f"pat_stderr_attempt{idx}.txt"
        out.write_text(attempt.stdout, encoding="utf-8")
        err.write_text(attempt.stderr, encoding="utf-8")
        metadata.append(
            {
                "index": idx,
                "cmd": [str(part) for part in attempt.cmd],
                "returncode": attempt.returncode,
                "stdout_path": str(out),
                "stderr_path": str(err),
            }
        )
    return metadata


def _should_try_pat3_mono_compat_fallback(
    *,
    pat_console_path: Path,
    use_mono: bool,
    stdout: str,
    stderr: str,
    out_path: Path,
) -> bool:
    if not use_mono:
        return False
    if out_path.exists():
        return False

    if "pat3.console" not in pat_console_path.name.lower():
        return False

    combined = f"{stdout}\n{stderr}".lower()
    has_pat_usage = "for all modules except uml:" in combined
    has_nesc_startup_symptom = (
        "invalid arguments." in combined
        and ("invalid image" in combined or "object reference not set to an instance of an object" in combined)
    )
    return has_pat_usage and has_nesc_startup_symptom


def _prepare_pat3_mono_compat_runtime(*, pat_console_path: Path, run_dir: Path) -> Path:
    """Create isolated PAT runtime and inject a NESC shim for PAT3-on-Mono compatibility."""

    compat_root = run_dir / "pat3_mono_compat_runtime"
    if compat_root.exists():
        shutil.rmtree(compat_root)
    shutil.copytree(pat_console_path.parent, compat_root)

    shim_src = compat_root / "Modules" / "NESC" / "nesc_shim.cs"
    shim_src.parent.mkdir(parents=True, exist_ok=True)
    shim_src.write_text(_nesc_shim_source(), encoding="utf-8")

    shim_out = (compat_root / "Modules" / "NESC" / "PAT.Module.NESC.dll").resolve()
    pat_common = (compat_root / "PAT.Common.dll").resolve()
    shim_src_abs = shim_src.resolve()

    mcs_cmd = [
        _resolve_mcs_path(),
        "-target:library",
        f"-out:{shim_out}",
        f"-r:{pat_common}",
        str(shim_src_abs),
    ]
    try:
        mcs_proc = subprocess.run(
            mcs_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=str(compat_root),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "mcs compiler not found. PAT3 Mono compatibility fallback requires Mono C# compiler (`mcs`)."
        ) from exc

    (run_dir / "pat3_shim_compile_stdout.txt").write_text(mcs_proc.stdout or "", encoding="utf-8")
    (run_dir / "pat3_shim_compile_stderr.txt").write_text(mcs_proc.stderr or "", encoding="utf-8")
    (run_dir / "pat3_shim_compile_cmd.txt").write_text(" ".join(mcs_cmd), encoding="utf-8")

    if mcs_proc.returncode != 0:
        detail = _first_nonempty_line(mcs_proc.stderr or "") or _first_nonempty_line(mcs_proc.stdout or "")
        raise RuntimeError(f"Failed to compile PAT3 NESC shim. {detail or 'Unknown compile failure.'}")

    return compat_root / pat_console_path.name


def _resolve_mcs_path() -> str:
    settings = CoachConfig.from_env()
    mono_path = Path(settings.mono_path).expanduser()
    if mono_path.is_absolute():
        sibling = mono_path.with_name("mcs")
        if sibling.exists():
            return str(sibling)

    detected = shutil.which("mcs")
    if detected:
        return detected
    return "mcs"


def _nesc_shim_source() -> str:
    return textwrap.dedent(
        """\
        using System;
        using System.Collections.Generic;
        using PAT.Common;
        using PAT.Common.Classes.ModuleInterface;

        namespace PAT.NESC
        {
            // PAT3 console invokes these methods on startup even when using non-NESC modules.
            public static class NCSetting
            {
                public static void SetBufferSize(int size) {}
                public static void SetAbstractionLevel(int level) {}
            }

            // Optional no-op facade so reflective module loading remains valid.
            public sealed class ModuleFacade : ModuleFacadeBase
            {
                protected override SpecificationBase InstanciateSpecification(string text, string options, string filePath)
                {
                    throw new NotSupportedException("NESC shim is compatibility-only.");
                }

                public override string ModuleName => "NESC";
                public override List<string> GetTemplateTypes() => new List<string>();
                public override SortedList<string, string> GetTemplateNames(string type) => new SortedList<string, string>();
                public override string GetTemplateModel(string templateName) => string.Empty;
            }
        }
        """
    )


def _write_summary(*, summary_json: Path, probability: float | None) -> None:
    summary_json.write_text(
        json.dumps(
            {
                "question": None,
                "players": None,
                "params_used": None,
                "probability": probability,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _first_nonempty_line(text: str) -> str | None:
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            return line
    return None


def _extract_pat_model_error(*, stdout: str, stderr: str) -> str | None:
    signals = (
        "parsing error:",
        "runtime exception occurred:",
        "error occurred:",
        "invalid file name:",
        "invalid folder name:",
        "invalid arguments.",
    )
    for raw in f"{stdout}\n{stderr}".splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(token in lowered for token in signals):
            return line
    return None


def _infer_hint_from_output(*, stdout: str, stderr: str) -> str | None:
    combined = f"{stdout}\n{stderr}".lower()
    if "invalid arguments. invalid image" in combined:
        return (
            "PAT3.Console under Mono can fail due NESC startup checks. "
            "The runner will attempt a compatibility shim automatically; if this still fails, install full Mono (with mcs) or use PAT4."
        )
    if "object reference not set to an instance of an object" in combined:
        return (
            "This is usually PAT3 NESC startup failure on Mono. "
            "Try setting PAT_CONSOLE_PATH to PAT3.Console.exe and let the runner apply compatibility fallback."
        )
    return None
