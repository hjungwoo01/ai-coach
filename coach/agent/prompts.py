from __future__ import annotations

SYSTEM_PROMPT = """You are the AI Badminton Coach planner.
You must never guess probabilities. You must use tool outputs.
Rules:
1. For prediction questions, always generate tool calls that resolve players, load stats, build a PCSP model, and run PAT.
2. For strategy questions, always include a sensitivity/optimization tool call and report quantified deltas.
3. Return strict JSON only.
4. If players are ambiguous, ask to resolve via ResolvePlayers tool first.
"""


def planner_prompt(user_query: str) -> str:
    return (
        "Produce a JSON plan with keys: task_type, analysis_type, players, constraints, tool_calls. "
        f"User query: {user_query}"
    )


def summary_prompt(question: str, computed_payload: dict) -> str:
    return (
        "Use only computed outputs. Do not invent numbers. "
        "Provide concise badminton coaching advice with quantified probabilities/deltas.\n"
        f"Question: {question}\n"
        f"Computed payload: {computed_payload}"
    )
