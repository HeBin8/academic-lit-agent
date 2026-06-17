"""Multi-turn conversation memory with short-term + long-term storage."""

import json, pathlib
from datetime import datetime
from typing import Optional


MEMORY_FILE = pathlib.Path(__file__).parent.parent.parent / "data" / "conversations.json"


class ConversationMemory:
    """Manages conversation history with truncation to token budget."""

    def __init__(self, max_history_turns: int = 20):
        self.max_history_turns = max_history_turns
        self.messages: list[dict] = []
        self.system_prompt: str = ""
        self._session_id: Optional[str] = None

    def _load_store(self) -> dict:
        """Load persisted conversations from disk."""
        if not MEMORY_FILE.exists():
            return {"sessions": {}}
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"sessions": {}}

    def _save(self) -> None:
        """Persist the current session without interrupting the chat loop."""
        if not self._session_id:
            return
        try:
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = self._load_store()
            sessions = data.setdefault("sessions", {})
            first_user = next((m["content"] for m in self.messages if m.get("role") == "user"), "")
            sessions[self._session_id] = {
                "session_id": self._session_id,
                "title": (first_user[:40] or "New conversation"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "messages": self.messages,
            }
            MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            # Persistence should never make the interactive agent fail.
            return

    # ── session management ───────────────────────────────────────

    def new_session(self, system_prompt: str = "") -> str:
        """Start a fresh session. Returns session ID."""
        from uuid import uuid4
        self._session_id = str(uuid4())[:8]
        self.messages = []
        if system_prompt:
            self.system_prompt = system_prompt
            self.messages.append({"role": "system", "content": system_prompt})
        self._save()
        return self._session_id

    # ── message management ───────────────────────────────────────

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()
        self._save()

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self._save()

    def add_tool_result(self, tool_name: str, result: str) -> None:
        self.messages.append({
            "role": "system",
            "content": f"[Tool: {tool_name} returned]\n{result[:2000]}",
        })
        self._trim()
        self._save()

    def get_history(self) -> list[dict]:
        """Return all messages for LLM context."""
        return self.messages

    def _trim(self):
        """Trim oldest history turns when over budget (preserving system prompt)."""
        if len(self.messages) <= self.max_history_turns + 1:
            return
        # Keep system prompt (index 0 if it exists)
        start = 1 if self.messages[0]["role"] == "system" else 0
        excess = len(self.messages) - self.max_history_turns - 1
        if excess > 0:
            self.messages = (
                self.messages[:1] + self.messages[1 + excess:]
                if start == 1
                else self.messages[excess:]
            )

    # ── context window summary ───────────────────────────────────

    def get_token_estimate(self) -> int:
        """Rough total token estimate for all messages."""
        return sum(len(m["content"]) // 2 for m in self.messages)

    def summary(self) -> str:
        n_user = sum(1 for m in self.messages if m["role"] == "user")
        n_tool = sum(1 for m in self.messages if m["role"] == "system" and m["content"].startswith("[Tool:"))
        return f"Session {self._session_id}: {n_user} user messages, {n_tool} tool calls, ~{self.get_token_estimate()} tokens"

