"""
Test script for verifying Upstash Redis key rotation in LLMClient.
Run from the repo root:
    python scripts/test_redis_rotation.py
"""
from __future__ import annotations

import os
import sys
import types as builtin_types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv(override=False)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def fail(msg: str) -> None:
    print(f"  [!!]  {msg}")


original_env_url = os.getenv("UPSTASH_REDIS_REST_URL")
original_env_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")


def test_redis_rotation() -> None:
    section("Test: Redis-backed Circular Key Rotation Simulation")

    if not original_env_url or not original_env_token:
        fail("Requires UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN in .env")
        return

    from coach.agent.llm_client import LLMClient

    try:
        from upstash_redis import Redis as UpstashRedis
    except ImportError:
        fail("Requires upstash-redis to run this test.")
        return

    # 1. Setup Redis List using Real Keys from Environment
    test_list_name = "test_gemini_mock_queue"
    os.environ["GEMINI_KEYS_REDIS_LIST"] = test_list_name

    from coach.agent.llm_client import _load_env_keys_only

    real_keys = _load_env_keys_only()

    if len(real_keys) < 3:
        fail("Requires at least 3 real Gemini API keys in .env (e.g. GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3) to test rotation.")
        return

    redis = UpstashRedis(url=original_env_url, token=original_env_token)
    redis.delete(test_list_name)
    redis.rpush(test_list_name, *real_keys)
    ok(f"Seeded Redis list '{test_list_name}' with {len(real_keys)} real keys.")

    # 2. Initialize LLMClient (will attach to Upstash because Env Vars exist)
    client = LLMClient()

    if client._redis_queue is None:
        fail("LLMClient did not properly initialize _redis_queue!")
        return

    starting_key = client._keys[0]
    ok(f"Client Initialized. Starting Key: {starting_key[:10]}...")
    if starting_key != real_keys[0]:
        fail(f"Expected starting key '{real_keys[0][:10]}...', got {starting_key[:10]}...")

    # 3. Mock the GenAI API Call to force Quota Errors
    call_count = 0

    # Replace client with a plain namespace so we can control generate_content.
    fake_models = builtin_types.SimpleNamespace()
    client.client = builtin_types.SimpleNamespace(models=fake_models)

    def mock_generate_content(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        # Succeed dynamically on the 3rd attempt
        if call_count <= 2:
            print(f"    [Mock] Simulating Quota Error 429 using key '{client.api_key}'")

            raise Exception("resource exhausted - 429")

        print(f"    [Mock] Simulating Success using key '{client.api_key}'")
        return builtin_types.SimpleNamespace(text="SUCCESS")

    client.client.models.generate_content = mock_generate_content

    # 4. Trigger Rotation Loop
    try:
        result = client._generate_with_rotation(contents="test", config=None)

        final_key = client.api_key
        target_key = real_keys[2]

        print(f"  Target result reached: {result.text}")
        ok(f"Landed successfully on Key: '{final_key[:10]}...'")

        if final_key == target_key:
            ok("Rotation behavior is fully CORRECT.")
        else:
            fail(f"Rotation behavior INCORRECT. Expected to land on '{target_key[:10]}...', landed on '{final_key[:10]}...'")

    except Exception as e:
        fail(f"Failed with unexpected Exception: {e}")

    finally:
        # Cleanup
        redis.delete(test_list_name)
        ok(f"Cleaned up list '{test_list_name}'.")

if __name__ == "__main__":
    test_redis_rotation()
