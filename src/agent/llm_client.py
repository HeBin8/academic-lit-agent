"""Unified LLM client supporting any OpenAI-compatible API."""

import json, os
from typing import Optional
import urllib.request, urllib.error


class LLMClient:
    """LLM client that works with any OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    # ── chat completion ──────────────────────────────────────────

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: int = 4096) -> str:
        """Send a chat completion request. Returns the assistant reply text."""
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()

        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode())
                    message = data["choices"][0].get("message", {})
                    content = message.get("content") or ""
                    if not content and message.get("reasoning_content"):
                        content = message["reasoning_content"]
                    return content
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"LLM API error {e.code}: {e.read().decode()}")
            except Exception:
                import time
                time.sleep(1)
                continue

        raise RuntimeError("LLM request failed after 3 retries")

    # ── convenience wrappers ─────────────────────────────────────

    def system_message(self, text: str) -> dict:
        return {"role": "system", "content": text}

    def user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def assistant_message(self, text: str) -> dict:
        return {"role": "assistant", "content": text}

    def count_tokens(self, text: str) -> int:
        """Rough token estimation (~4 chars per token for Chinese text)."""
        return len(text) // 2
