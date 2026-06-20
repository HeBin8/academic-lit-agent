"""Tests for conversation memory: session management, message storage,
truncation, persistence, and token estimation.

Covers:
  - #18  Session creation and role-based message management
  - #19  Message truncation under token budget
  - #20  JSON persistence (save/load sessions)
  - #21  Token estimation
  - #22  Tool result injection and formatting
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.memory import conversation_memory as mem_module
from src.memory.conversation_memory import ConversationMemory


def _get_mem_file() -> Path:
    """Return the current MEMORY_FILE value (respects monkeypatch)."""
    return mem_module.MEMORY_FILE


# ═══════════════════════════════════════════════════════════════════
# Fixture: memory with a temp persistence file
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def memory_with_tempfile(monkeypatch):
    """ConversationMemory that writes to a temp file instead of data/."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_path = tmp.name
    tmp.close()

    monkeypatch.setattr(
        mem_module,
        "MEMORY_FILE",
        Path(tmp_path),
    )

    yield ConversationMemory()

    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════
# Test Case 18: Session creation and message management
# ═══════════════════════════════════════════════════════════════════

def test_new_session_creates_system_message(memory_with_tempfile):
    """new_session() initializes with a system prompt and returns a session ID."""
    mem = memory_with_tempfile
    sid = mem.new_session(system_prompt="You are a test assistant.")

    assert sid is not None
    assert len(sid) == 8  # UUID truncated to 8 chars
    assert len(mem.messages) == 1
    assert mem.messages[0]["role"] == "system"
    assert mem.messages[0]["content"] == "You are a test assistant."


def test_add_user_and_assistant_messages(memory_with_tempfile):
    """Messages are appended with correct roles."""
    mem = memory_with_tempfile
    mem.new_session("System prompt.")

    mem.add_user("Hello, can you help?")
    mem.add_assistant("Of course! What do you need?")

    assert len(mem.messages) == 3
    assert mem.messages[0]["role"] == "system"
    assert mem.messages[1]["role"] == "user"
    assert mem.messages[1]["content"] == "Hello, can you help?"
    assert mem.messages[2]["role"] == "assistant"
    assert mem.messages[2]["content"] == "Of course! What do you need?"


def test_add_tool_result_formatting(memory_with_tempfile):
    """Tool results are stored as system messages with [Tool: ...] prefix."""
    mem = memory_with_tempfile
    mem.new_session("System.")

    mem.add_user("Run code.")
    mem.add_tool_result("code_executor", "Output: 42")

    tool_msgs = [m for m in mem.messages if m.get("content", "").startswith("[Tool:")]
    assert len(tool_msgs) == 1
    assert "[Tool: code_executor returned]" in tool_msgs[0]["content"]
    assert "Output: 42" in tool_msgs[0]["content"]


def test_tool_result_truncation(memory_with_tempfile):
    """Tool results longer than 2000 chars are truncated."""
    mem = memory_with_tempfile
    mem.new_session("System.")

    long_result = "x" * 3000
    mem.add_tool_result("long_tool", long_result)

    tool_msg = [m for m in mem.messages if "[Tool:" in m.get("content", "")][0]
    # The stored result should be truncated to 2000 chars
    result_part = tool_msg["content"].split("]\n", 1)[1] if "]\n" in tool_msg["content"] else tool_msg["content"]
    assert len(result_part) <= 2000 + len("[Tool: long_tool returned]\n")


# ═══════════════════════════════════════════════════════════════════
# Test Case 19: Message truncation under token budget
# ═══════════════════════════════════════════════════════════════════

def test_trim_preserves_system_prompt(memory_with_tempfile):
    """Truncation keeps system prompt at index 0."""
    mem = memory_with_tempfile
    mem.max_history_turns = 3  # very small budget
    mem.new_session("KEEP_THIS_SYSTEM_PROMPT")

    # Add many messages to trigger truncation
    for i in range(20):
        mem.add_user(f"Message {i}")
        mem.add_assistant(f"Response {i}")

    # System prompt should still be first
    assert mem.messages[0]["role"] == "system"
    assert mem.messages[0]["content"] == "KEEP_THIS_SYSTEM_PROMPT"

    # Total messages should respect budget (system + max_history_turns user/assistant pairs)
    # The budget is max_history_turns messages total (excluding system prompt)
    assert len(mem.messages) <= mem.max_history_turns + 1 + 1  # system + turns + 1 buffer


def test_trim_without_system_prompt():
    """Truncation works even when there's no system message."""
    mem = ConversationMemory(max_history_turns=3)
    # Don't call new_session — manually add messages without system
    mem.messages = []
    for i in range(20):
        mem.messages.append({"role": "user", "content": f"Q{i}"})
        mem.messages.append({"role": "assistant", "content": f"A{i}"})

    mem._trim()

    assert len(mem.messages) <= 3 + 1 + 1


# ═══════════════════════════════════════════════════════════════════
# Test Case 20: JSON persistence (save/load)
# ═══════════════════════════════════════════════════════════════════

def test_save_and_reload_session(memory_with_tempfile):
    """A saved session can be read back from disk."""
    mem = memory_with_tempfile
    mem.new_session("System prompt for persistence test.")
    mem.add_user("User message 1")
    mem.add_assistant("Assistant reply 1")
    mem.add_user("User message 2")

    # Read back the JSON file directly
    with open(str(_get_mem_file()), "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "sessions" in data
    sessions = data["sessions"]
    assert len(sessions) >= 1

    session = list(sessions.values())[0]
    assert "session_id" in session
    assert "title" in session
    assert "updated_at" in session
    assert "messages" in session
    assert len(session["messages"]) >= 3


def test_multiple_sessions_persisted(memory_with_tempfile):
    """Multiple sessions are stored independently."""
    mem = memory_with_tempfile

    # Session 1
    mem.new_session("System A")
    mem.add_user("Question A")
    mem.add_assistant("Answer A")

    # Session 2
    mem.new_session("System B")
    mem.add_user("Question B")
    mem.add_assistant("Answer B")

    # Load from disk
    with open(str(_get_mem_file()), "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = data["sessions"]
    assert len(sessions) >= 2


def test_session_title_from_first_user_message(memory_with_tempfile):
    """Session title is derived from the first user message (truncated to 40 chars)."""
    mem = memory_with_tempfile
    mem.new_session("System.")
    mem.add_user("This is a very long first user message that exceeds forty characters easily.")

    with open(str(_get_mem_file()), "r", encoding="utf-8") as f:
        data = json.load(f)

    session = list(data["sessions"].values())[0]
    assert len(session["title"]) <= 45  # 40 + some margin


def test_corrupt_json_file_recovery(memory_with_tempfile):
    """If the JSON file is corrupt, _load_store returns empty dict gracefully."""
    mem = memory_with_tempfile

    # Corrupt the file
    with open(str(_get_mem_file()), "w", encoding="utf-8") as f:
        f.write("this is not valid json {{{")

    # Should not raise
    data = mem._load_store()
    assert data == {"sessions": {}}


# ═══════════════════════════════════════════════════════════════════
# Test Case 21: Token estimation
# ═══════════════════════════════════════════════════════════════════

def test_token_estimation():
    """Token estimate is roughly len(content) // 2."""
    mem = ConversationMemory()
    mem.new_session("Hello world")  # 11 chars → ~5 tokens
    mem.add_user("Short")            # 5 chars → ~2 tokens

    estimate = mem.get_token_estimate()
    # 11//2 + 5//2 = 5 + 2 = 7
    assert estimate == 7


def test_token_estimation_empty():
    """Empty memory has 0 token estimate."""
    mem = ConversationMemory()
    assert mem.get_token_estimate() == 0


def test_summary_method():
    """summary() returns a human-readable session summary."""
    mem = ConversationMemory()
    mem.new_session("System.")
    mem.add_user("Q1")
    mem.add_assistant("A1")
    mem.add_tool_result("search", "Found papers.")

    summary = mem.summary()
    assert "1 user messages" in summary
    assert "1 tool calls" in summary


# ═══════════════════════════════════════════════════════════════════
# Test Case 22: Tool result injection and history retrieval
# ═══════════════════════════════════════════════════════════════════

def test_get_history_returns_all_messages(memory_with_tempfile):
    """get_history() returns the complete message list for LLM context."""
    mem = memory_with_tempfile
    mem.new_session("System prompt.")

    mem.add_user("Search for RAG papers.")
    mem.add_assistant("Let me search...")
    mem.add_tool_result("search_papers", "Found 3 papers about RAG.")

    history = mem.get_history()
    assert len(history) == 4  # system, user, assistant, tool_result
    assert history[0]["role"] == "system"
    assert history[3]["role"] == "system"  # tool results are system messages
    assert "[Tool: search_papers returned]" in history[3]["content"]
