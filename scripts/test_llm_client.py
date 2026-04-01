"""
Manual smoke-test script for LLMClient.
Run from the repo root:
    python scripts/test_llm_client.py
"""
from __future__ import annotations

import os
import sys
import textwrap
import types as builtin_types
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def fail(msg: str) -> None:
    print(f"  [!!]  {msg}")

# ---------------------------------------------------------------------------
# Test 1 – real Gemini API call
# ---------------------------------------------------------------------------

def test_real_api_call() -> None:
    section("Test 1: Real Gemini API call")
    from coach.agent.llm_client import LLMClient

    client = LLMClient()
    if not client.enabled:
        fail("Skipped — client disabled.")
        return

    try:
        from google.genai import types

        response = client._generate_with_rotation(
            contents="Reply with exactly the word PONG and nothing else.",
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = (getattr(response, "text", None) or "").strip()
        print(f"  Response text: {text!r}")
        if "pong" in text.lower():
            ok("Got expected PONG response from Gemini API.")
        else:
            ok(f"API responded (unexpected text, but it works): {text[:80]!r}")
    except RuntimeError as exc:
        # All keys exhausted — likely all keys are invalid/quota-hit.
        fail(f"All keys exhausted: {exc}")
        fail("Check that the keys in your gist are valid and have quota remaining.")
    except Exception as exc:
        error_str = str(exc)
        if "API key not valid" in error_str or "INVALID_ARGUMENT" in error_str:
            fail("API key is invalid (400 INVALID_ARGUMENT).")
            fail("The keys in your gist may be from a different project or have been revoked.")
            fail("Update keys.json in your gist with fresh, valid Gemini API keys.")
        else:
            fail(f"Unexpected error: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Test 2 – circular rotation (fully offline, no real genai)
# ---------------------------------------------------------------------------

def _make_offline_client(keys: list[str], start_index: int = 0) -> object:
    """Build a minimal LLMClient-like object without touching genai or the gist."""
    from coach.agent.llm_client import LLMClient

    client = LLMClient.__new__(LLMClient)
    client.model = "gemini-2.5-flash"
    client._keys = list(keys)
    client._index = start_index
    client.enabled = True

    # Replace client with a plain namespace so we can control generate_content.
    fake_models = builtin_types.SimpleNamespace()
    client.client = builtin_types.SimpleNamespace(models=fake_models)
    return client


def test_rotation_simulation() -> None:
    section("Test 2: Circular key rotation simulation (offline)")
    from coach.agent.llm_client import LLMClient

    # --- Scenario A: first 2 keys fail, third succeeds --------------------
    client = _make_offline_client(["KEY_A", "KEY_B", "KEY_C"])
    call_count = 0

    def two_fails(*a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("resource exhausted — 429")
        ok_resp = builtin_types.SimpleNamespace(text="SUCCESS")
        return ok_resp

    client.client.models.generate_content = two_fails

    indices_visited: list[int] = []
    original_advance = LLMClient._advance

    def patched_advance(self):
        indices_visited.append(self._index)
        original_advance(self)

    with mock.patch.object(LLMClient, "_advance", patched_advance):
        client_typed: LLMClient = client  # type: ignore[assignment]
        client_typed._generate_with_rotation(contents="test", config=None)

    keys_tried = [["KEY_A", "KEY_B", "KEY_C"][i] for i in indices_visited]
    print(f"  Keys rotated : {keys_tried}  →  landed on {client_typed._keys[client_typed._index]}")
    ok(f"Rotated through {len(indices_visited)} failing key(s) and succeeded.")

    # --- Scenario B: all keys exhausted → RuntimeError -------------------
    client2 = _make_offline_client(["K1", "K2"])
    client2.client.models.generate_content = mock.MagicMock(  # type: ignore[attr-defined]
        side_effect=Exception("resource exhausted — 429")
    )

    try:
        client2._generate_with_rotation(contents="test", config=None)  # type: ignore[attr-defined]
        fail("Expected RuntimeError but nothing was raised.")
    except RuntimeError as exc:
        ok(f"All-keys-exhausted → RuntimeError: {exc}")
    except Exception as exc:
        fail(f"Wrong exception: {type(exc).__name__}: {exc}")

    # --- Scenario C: wrap-around (start at last key) ----------------------
    client3 = _make_offline_client(["X", "Y", "Z"], start_index=2)  # start at "Z"
    wrapped_indices: list[int] = []
    call_n = 0

    def one_fail_then_ok(*a, **kw):
        nonlocal call_n
        call_n += 1
        wrapped_indices.append(client3._index)  # type: ignore[attr-defined]
        if call_n == 1:
            raise Exception("quota exceeded 429")
        return builtin_types.SimpleNamespace(text="ok")

    client3.client.models.generate_content = one_fail_then_ok  # type: ignore[attr-defined]
    client3._generate_with_rotation(contents="test", config=None)  # type: ignore[attr-defined]
    print(f"  Wrap indices : {wrapped_indices}  (2 → 0, wrapping past 'Z' back to 'X')")
    if wrapped_indices == [2, 0]:
        ok("Circular wrap: index 2 → 0 correctly.")
    else:
        fail(f"Unexpected wrap path: {wrapped_indices}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(textwrap.dedent("""\
        LLMClient smoke-test
        ---------------------------------------------------------
        Test 1  ->  real network calls (Gemini API)
        Test 2  ->  fully offline simulations
    """))

    test_real_api_call()
    test_rotation_simulation()

    print("\nDone.\n")
