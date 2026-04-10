"""
Shared API client utilities for calling Llama 3.3 70B via free API backends.

Supported backends:
  groq       -- https://console.groq.com        (GROQ_API_KEY)
  cerebras   -- https://cloud.cerebras.ai       (CEREBRAS_API_KEY)
  openrouter -- https://openrouter.ai           (OPENROUTER_API_KEY)
  together   -- https://api.together.xyz        (TOGETHER_API_KEY)

API keys are read (in order of priority) from:
  1. The cli_key argument passed directly
  2. The environment variable named in the config (e.g. TOGETHER_API_KEY)
  3. A file in <repo_root>/api_keys/  (e.g. api_keys/togetherai.apikey)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

API_KEY_DIR = Path(__file__).parent.parent / "api_keys"

BACKEND_CONFIGS: dict[str, dict] = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "rpm": 28,
        "env_key": "GROQ_API_KEY",
        "key_file": API_KEY_DIR / "grok.apikey",
        "base_url": None,  # use groq SDK directly
    },
    "cerebras": {
        "model": "llama-3.3-70b-instruct",
        "rpm": 28,
        "env_key": "CEREBRAS_API_KEY",
        "key_file": API_KEY_DIR / "cerebra.apikey",
        "base_url": "https://api.cerebras.ai/v1",
    },
    "openrouter": {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "rpm": 18,
        "env_key": "OPENROUTER_API_KEY",
        "key_file": API_KEY_DIR / "openrouter.apikey",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "together": {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "rpm": 58,
        "env_key": "TOGETHER_API_KEY",
        "key_file": API_KEY_DIR / "togetherai.apikey",
        "base_url": "https://api.together.xyz/v1",
    },
}


def load_api_key(backend: str, cli_key: str = "") -> str:
    """Return API key for *backend*, trying cli_key → env var → key file."""
    cfg = BACKEND_CONFIGS[backend]
    key = cli_key or os.environ.get(cfg["env_key"], "")
    if not key:
        key_file: Path = cfg["key_file"]
        if key_file.exists():
            key = key_file.read_text().strip()
    return key


def make_client(backend: str, api_key: str):
    """Return an API client for *backend* (OpenAI-compatible or Groq)."""
    cfg = BACKEND_CONFIGS[backend]
    if backend == "groq":
        from groq import Groq
        return Groq(api_key=api_key)
    else:
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def call_chat(
    client,
    model: str,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str | None:
    """
    Send a chat completion request and return the response text.

    Retries up to *max_retries* times on rate-limit (429) errors with
    exponential back-off (30 s, 60 s, 90 s, …).  Returns None on failure.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries-1})...", flush=True)
                time.sleep(wait)
            else:
                print(f"  API error: {e}", flush=True)
                return None
    return None
