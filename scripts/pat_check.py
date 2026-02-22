from __future__ import annotations

import platform
import subprocess
import sys
import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coach.config import CoachConfig
from coach.pat.parser import parse_probability, read_pat_output
from coach.pat.runner import resolve_pat_console_path, run_pat


def main() -> None:
    cfg = CoachConfig.from_env()
    os_name = platform.system()
    pat_path = cfg.pat_console_path
    resolved_pat = resolve_pat_console_path(pat_path) if pat_path else None
    use_mono = cfg.resolve_use_mono(resolved_pat)

    print(f"OS: {os_name}")
    print(f"PAT_CONSOLE_PATH: {pat_path}")
    print(f"resolved PAT console: {resolved_pat}")
    print(f"MONO_PATH: {cfg.mono_path}")
    print(f"mono will be used: {use_mono}")

    if resolved_pat is None:
        _print_next_steps("PAT_CONSOLE_PATH is not set")
        return

    if not resolved_pat.exists():
        _print_next_steps(f"PAT executable not found: {resolved_pat}")
        return

    if resolved_pat.is_dir():
        _print_next_steps(f"PAT_CONSOLE_PATH points to a directory without PAT.Console.exe: {resolved_pat}")
        return

    cmd: list[str]
    if use_mono:
        cmd = [cfg.mono_path, str(resolved_pat), "-ver"]
    else:
        cmd = [str(resolved_pat), "-ver"]

    print(f"Checking command: {' '.join(cmd)}")
    try:
        proc_env = os.environ.copy()
        if use_mono:
            proc_env.pop("MONO_PATH", None)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=min(cfg.pat_timeout_s, 20),
            check=False,
            env=proc_env,
        )
    except FileNotFoundError as exc:
        _print_next_steps(f"Command failed: {exc}")
        return
    except subprocess.TimeoutExpired:
        print("PAT check timed out.")
        print("Try increasing PAT_TIMEOUT_S or run command manually.")
        return

    print(f"return code: {proc.returncode}")
    if proc.stdout:
        print("stdout:")
        print(proc.stdout.strip())
    if proc.stderr:
        print("stderr:")
        print(proc.stderr.strip())

    if proc.returncode == 0:
        print("PAT connectivity check passed.")
        _run_smoke_check(
            pat_console_path=resolved_pat,
            use_mono=use_mono,
            timeout_s=min(cfg.pat_timeout_s, 30),
        )
    else:
        _print_next_steps("PAT command ran but returned a non-zero code")


def _run_smoke_check(*, pat_console_path: Path, use_mono: bool, timeout_s: int) -> None:
    minimal = REPO_ROOT / "examples" / "minimal.pcsp"
    if not minimal.exists():
        print("Skipped module smoke check: examples/minimal.pcsp not found.")
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="pat_check_"))
    out_path = temp_dir / "pat_output.txt"

    result = run_pat(
        pcsp_path=minimal,
        out_path=out_path,
        mode="real",
        pat_console_path=pat_console_path,
        timeout_s=timeout_s,
        use_mono=use_mono,
    )
    if not bool(result.get("ok", False)):
        print("PAT module smoke check failed.")
        print(f"pat_error: {result.get('error', 'unknown PAT error')}")
        print(f"pat_stdout: {result.get('stdout_path')}")
        print(f"pat_stderr: {result.get('stderr_path')}")
        return

    try:
        probability = parse_probability(read_pat_output(out_path))
    except Exception as exc:
        print(f"PAT module smoke check output parse failed: {exc}")
        print(f"raw_output: {out_path}")
        return

    print(f"PAT module smoke check passed. Probability: {probability:.6f}")


def _print_next_steps(reason: str) -> None:
    print(f"PAT check failed: {reason}")
    print("Next steps:")
    print("1. Download PAT from https://www.comp.nus.edu.sg/~pat/patdownload.htm")
    print("2. Set PAT_CONSOLE_PATH=.../PAT.Console.exe")
    print("3. Install Mono and ensure `mono` is on PATH")


if __name__ == "__main__":
    main()
