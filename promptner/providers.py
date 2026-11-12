"""Hosted chat-completion adapters for OpenAI and Sarvam."""

from __future__ import annotations

import os
import random
import time


def create_client(provider: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the 'openai' package to use hosted model providers") from exc

    provider = provider.lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAI(api_key=api_key, timeout=90, max_retries=0)
    if provider == "sarvam":
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY is not set")
        # Sarvam documents an OpenAI-compatible Chat Completions API on /v2.
        return OpenAI(api_key=api_key, base_url="https://api.sarvam.ai/v2", timeout=90, max_retries=0)
    raise ValueError("Provider must be 'openai' or 'sarvam'")


def default_model(provider: str) -> str:
    if provider.lower() == "sarvam":
        return os.getenv("SARVAM_MODEL", "sarvam-105b")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def complete(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1600,
    reasoning_effort: str | None = None,
    retries: int = 4,
) -> str:
    request = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(**request)
            break
        except Exception as exc:
            if getattr(exc, "status_code", None) not in (429, 503) or attempt >= retries:
                raise
            response_obj = getattr(exc, "response", None)
            headers = getattr(response_obj, "headers", {}) or {}
            retry_after = headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after is not None else min(2**attempt, 30)
            except (TypeError, ValueError):
                delay = min(2**attempt, 30)
            time.sleep(delay + random.random() * 0.25)
    content = response.choices[0].message.content
    if not content:
        choice = response.choices[0]
        message = choice.message
        reasoning = getattr(message, "reasoning_content", None)
        raise RuntimeError(
            "Hosted model returned no visible content "
            f"(finish_reason={choice.finish_reason}, reasoning_tokens_present={bool(reasoning)})"
        )
    return content
