from __future__ import annotations

import json
import os
import re
from typing import Any

from coach.agent.prompts import SYSTEM_PROMPT, planner_prompt, summary_prompt

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional import at runtime
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]


class LLMClient:
    """Small wrapper around Gemini for JSON planning and final summarization."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.enabled = bool(self.api_key and genai is not None)
        if self.enabled:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def plan(self, user_query: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        response = self.client.models.generate_content(
            model=self.model,
            contents=planner_prompt(user_query),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        text = getattr(response, "text", None) or ""
        if not text.strip():
            return None
        return _extract_json_payload(text)

    def summarize(self, question: str, computed_payload: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None

        response = self.client.models.generate_content(
            model=self.model,
            contents=summary_prompt(question, computed_payload),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        text = getattr(response, "text", None)
        return text if text and text.strip() else None


def _extract_json_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```json\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON object found in response.", text, 0)
