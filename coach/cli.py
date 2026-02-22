from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

from coach.agent.llm_client import LLMClient
from coach.agent.planner import AgentExecutor
from coach.config import CoachConfig
from coach.data.adapters.local_csv import LocalCSVAdapter
from coach.pat.parser import parse_probability, read_pat_output
from coach.pat.runner import run_pat
from coach.runs import new_run_dir
from coach.service import BadmintonCoachService
from coach.utils import write_json


def _build_service() -> BadmintonCoachService:
    config = CoachConfig.from_env()
    adapter = LocalCSVAdapter()
    return BadmintonCoachService(adapter=adapter, runs_root=config.runs_dir, config=config)


def command_predict(args: argparse.Namespace) -> None:
    service = _build_service()
    result = service.predict(
        player_a=args.a,
        player_b=args.b,
        window=args.window,
        mode=args.mode,
        timeout_s=args.timeout,
        pat_path=args.pat_path,
    )

    print(f"run_id: {result.run_id}")
    print(f"run_dir: {result.run_dir}")
    print(f"player_a: {result.player_a}")
    print(f"player_b: {result.player_b}")
    print(f"win_probability_a: {result.probability:.6f}")
    print(f"mode: {result.mode}")


def command_strategy(args: argparse.Namespace) -> None:
    service = _build_service()
    result = service.strategy(
        player_a=args.a,
        player_b=args.b,
        window=args.window,
        mode=args.mode,
        budget=args.budget,
        timeout_s=args.timeout,
        pat_path=args.pat_path,
    )

    print(f"run_id: {result.run_id}")
    print(f"run_dir: {result.run_dir}")
    print(f"player_a: {result.player_a}")
    print(f"player_b: {result.player_b}")
    print(f"baseline_probability: {result.baseline_probability:.6f}")
    print(f"improved_probability: {result.improved_probability:.6f}")
    print(f"delta: {result.delta:.6f}")
    print(
        "best_changes: "
        f"serve_mix.short {result.best_candidate.serve_short_delta:+.3f}, "
        f"rally_style.attack {result.best_candidate.attack_delta:+.3f}"
    )

    print("top_alternatives:")
    for cand in result.top_alternatives:
        print(
            f"  rank={cand.rank} prob={cand.probability:.6f} "
            f"serve_short_delta={cand.serve_short_delta:+.3f} "
            f"attack_delta={cand.attack_delta:+.3f} l1={cand.l1_change:.3f}"
        )


def command_pat_run(args: argparse.Namespace) -> None:
    config = CoachConfig.from_env()

    pcsp_src = Path(args.pcsp).expanduser().resolve()
    if not pcsp_src.exists():
        raise SystemExit(f"PCSP file not found: {pcsp_src}")

    run_id, run_dir = new_run_dir(prefix="pat", base_dir=config.runs_dir)
    pcsp_dst = run_dir / pcsp_src.name
    shutil.copy2(pcsp_src, pcsp_dst)

    out_path = run_dir / "pat_output.txt"

    use_mono: bool | None
    if args.use_mono == "auto":
        use_mono = None
    else:
        use_mono = args.use_mono == "true"

    pat_console_path = Path(args.pat_path).expanduser() if args.pat_path else config.pat_console_path
    timeout_s = args.timeout if args.timeout is not None else config.pat_timeout_s

    result = run_pat(
        pcsp_path=pcsp_dst,
        out_path=out_path,
        mode=args.mode,
        pat_console_path=pat_console_path,
        timeout_s=timeout_s,
        use_mono=use_mono,
    )

    if not bool(result.get("ok", False)):
        summary = {
            "question": None,
            "players": None,
            "params_used": None,
            "probability": None,
            "timestamps": {"generated_utc": dt.datetime.now(dt.UTC).isoformat()},
            "pat_result": result,
            "parse_error": None,
        }
        write_json(run_dir / "summary.json", summary)

        print(f"run_id: {run_id}")
        print(f"run_dir: {run_dir}")
        print("probability: <pat-failed>")
        print(f"pat_error: {result.get('error', 'PAT execution failed')}")
        print(f"pat_stdout: {result.get('stdout_path')}")
        print(f"pat_stderr: {result.get('stderr_path')}")
        return

    probability: float | None = None
    parse_error: str | None = None
    try:
        probability = parse_probability(read_pat_output(out_path))
    except Exception as exc:
        parse_error = str(exc)

    summary = {
        "question": None,
        "players": None,
        "params_used": None,
        "probability": probability,
        "timestamps": {"generated_utc": dt.datetime.now(dt.UTC).isoformat()},
        "pat_result": result,
        "parse_error": parse_error,
    }
    write_json(run_dir / "summary.json", summary)

    if parse_error is not None:
        print(f"run_id: {run_id}")
        print(f"run_dir: {run_dir}")
        print("probability: <parse-failed>")
        print(f"parse_error: {parse_error}")
        return

    print(f"run_id: {run_id}")
    print(f"run_dir: {run_dir}")
    print(f"probability: {probability:.6f}")


def command_chat(args: argparse.Namespace) -> None:
    llm_client = LLMClient(model=args.model)
    service = _build_service()
    executor = AgentExecutor(service=service, llm_client=llm_client)

    print("AI Badminton Coach chat. Type 'quit' to exit.")
    while True:
        try:
            query = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break

        try:
            result = executor.run(query, mode=args.mode, window=args.window, budget=args.budget)
            print(f"Coach> {result.answer}")
            if args.show_trace:
                print(json.dumps(result.tool_trace, indent=2))
        except Exception as exc:
            print(f"Error: {exc}")


def command_experiments(args: argparse.Namespace) -> None:
    from coach.analysis.experiments import run_experiments

    outputs = run_experiments(output_dir=args.output_dir, mode=args.mode)
    for key, value in outputs.items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Badminton Coach")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Compute matchup win probability via PAT reachability")
    predict.add_argument("--a", required=True, help="Player A")
    predict.add_argument("--b", required=True, help="Player B")
    predict.add_argument("--window", type=int, default=30)
    predict.add_argument("--source", default="local", choices=["local"])
    predict.add_argument("--mode", default="mock", choices=["mock", "real"])
    predict.add_argument("--pat-path", default=None)
    predict.add_argument("--timeout", type=int, default=None)
    predict.set_defaults(func=command_predict)

    strategy = sub.add_parser("strategy", help="Optimize strategy knobs with PAT sensitivity runs")
    strategy.add_argument("--a", required=True, help="Player A")
    strategy.add_argument("--b", required=True, help="Player B")
    strategy.add_argument("--window", type=int, default=30)
    strategy.add_argument("--source", default="local", choices=["local"])
    strategy.add_argument("--mode", default="mock", choices=["mock", "real"])
    strategy.add_argument("--budget", type=int, default=60)
    strategy.add_argument("--knobs", default="serve_mix,rally_style")
    strategy.add_argument("--pat-path", default=None)
    strategy.add_argument("--timeout", type=int, default=None)
    strategy.set_defaults(func=command_strategy)

    pat_run = sub.add_parser("pat-run", help="Run PAT Console directly on a PCSP# file")
    pat_run.add_argument("--pcsp", required=True, help="Path to input .pcsp file")
    pat_run.add_argument("--mode", default="mock", choices=["mock", "real"])
    pat_run.add_argument("--pat-path", default=None, help="Override PAT_CONSOLE_PATH")
    pat_run.add_argument("--timeout", type=int, default=None)
    pat_run.add_argument("--use-mono", default="auto", choices=["auto", "true", "false"])
    pat_run.set_defaults(func=command_pat_run)

    chat = sub.add_parser("chat", help="Interactive natural-language planning and tool execution")
    chat.add_argument("--mode", default="mock", choices=["mock", "real"])
    chat.add_argument("--window", type=int, default=30)
    chat.add_argument("--budget", type=int, default=60)
    chat.add_argument("--pat-path", default=None)
    chat.add_argument("--timeout", type=int, default=None)
    chat.add_argument("--model", default="gemini-1.5-flash")
    chat.add_argument("--show-trace", action="store_true")
    chat.set_defaults(func=command_chat)

    exp = sub.add_parser("experiments", help="Run reproducible prediction + strategy experiment batch")
    exp.add_argument("--output-dir", default="runs/experiments")
    exp.add_argument("--mode", default="mock", choices=["mock", "real"])
    exp.set_defaults(func=command_experiments)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    Path(CoachConfig.from_env().runs_dir).mkdir(parents=True, exist_ok=True)
    try:
        args.func(args)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
