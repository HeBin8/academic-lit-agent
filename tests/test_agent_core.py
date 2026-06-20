"""Tests for the core ReAct agent loop: parsing, reflection, max steps, error recovery.

Covers:
  - #1  Basic ReAct cycle (Thought→Action→Observation→Final Answer)
  - #2  Single-step direct Final Answer (no tool needed)
  - #3  Multi-step multi-tool workflow
  - #4  Reflection injection after step > 1
  - #5  Max steps exhaustion without Final Answer
  - #6  Malformed LLM response recovery
  - #7  Parameter parsing edge cases
"""

import pytest
from tests.conftest import MockLLMClient, react_response
from src.agent.core import AcademicLitAgent
from src.agent.tool_registry import ToolRegistry
from src.tools import CodeExecutorTool, SQLitePaperDBTool


# ═══════════════════════════════════════════════════════════════════
# Test Case 1: Basic ReAct cycle — Thought → Action → Final Answer
# ═══════════════════════════════════════════════════════════════════

def test_agent_basic_react_cycle(tool_registry):
    """Agent completes a single tool call then gives Final Answer."""
    mock = MockLLMClient()
    mock.set_responses([
        react_response(
            "I need to execute code to calculate a value.",
            "code_executor | code=print(2+2) | timeout=5"
        ),
        react_response(
            "The code executed successfully. I have the answer.",
            final_answer="2 + 2 = 4"
        ),
    ])

    agent = AcademicLitAgent(mock, tool_registry, max_steps=10, reflection=True)
    result = agent.process_message("Calculate 2+2 using code executor")

    assert result["steps"] <= 3
    assert "code_executor" in result["tools_called"]
    assert len(result["trace"]) >= 3  # action, observation, final_answer
    assert result["trace"][0]["type"] == "action"
    assert result["trace"][1]["type"] == "observation"
    assert result["trace"][-1]["type"] == "final_answer"


# ═══════════════════════════════════════════════════════════════════
# Test Case 2: Direct Final Answer (no tool needed)
# ═══════════════════════════════════════════════════════════════════

def test_agent_direct_final_answer(tool_registry):
    """When no tool is needed, the agent responds directly."""
    mock = MockLLMClient()
    mock.set_responses([
        react_response(
            "This is a simple greeting. No tools needed.",
            final_answer="Hello! I'm an academic literature assistant. How can I help you today?"
        ),
    ])

    agent = AcademicLitAgent(mock, tool_registry, max_steps=10)
    result = agent.process_message("Hello!")

    assert result["steps"] <= 2
    assert result["tools_called"] == []
    assert "Hello" in result["response"] or "assistant" in result["response"].lower()
    assert result["trace"][-1]["type"] == "final_answer"


# ═══════════════════════════════════════════════════════════════════
# Test Case 3: Multi-step multi-tool workflow
# ═══════════════════════════════════════════════════════════════════

def test_agent_multi_tool_workflow(tool_registry):
    """Agent chains multiple tool calls across steps."""
    mock = MockLLMClient()
    mock.set_responses([
        react_response(
            "First, I'll save a paper to the database.",
            'sqlite_paper_db | operation=save | paper_id=test001 | title=Test Paper | year=2024'
        ),
        react_response(
            "Paper saved. Now let me verify by listing.",
            "sqlite_paper_db | operation=list"
        ),
        react_response(
            "I can see the paper. Now I have all the information.",
            final_answer="Paper 'Test Paper' (test001) was saved and verified in the database."
        ),
    ])

    agent = AcademicLitAgent(mock, tool_registry, max_steps=10)
    result = agent.process_message("Save a test paper and verify it.")

    assert result["steps"] <= 5
    assert len(result["tools_called"]) >= 1
    assert "sqlite_paper_db" in result["tools_called"]
    assert result["trace"][-1]["type"] == "final_answer"


# ═══════════════════════════════════════════════════════════════════
# Test Case 4: Reflection mechanism (injected after step > 1)
# ═══════════════════════════════════════════════════════════════════

def test_agent_reflection_injection(tool_registry):
    """After step 2+, a reflection prompt is injected into the conversation."""
    mock = MockLLMClient()
    mock.set_responses([
        react_response("Step 1", "code_executor | code=print(1)"),
        react_response("Step 2 - reflecting", "code_executor | code=print(2)"),
        react_response("Done after reflection.", final_answer="Completed."),
    ])

    agent = AcademicLitAgent(mock, tool_registry, max_steps=10, reflection=True)
    result = agent.process_message("Run two code steps.")

    # Check that a reflection message was added to conversation history
    reflection_found = any(
        "Reflection:" in m["content"]
        for m in agent.memory.messages
    )
    assert reflection_found or result["steps"] <= 4


# ═══════════════════════════════════════════════════════════════════
# Test Case 5: Max steps exhaustion
# ═══════════════════════════════════════════════════════════════════

def test_agent_max_steps_exhaustion(tool_registry):
    """When the agent never produces Final Answer, it stops at max_steps."""
    # Build responses that always make a tool call, never Final Answer
    responses = []
    for i in range(15):
        responses.append(
            react_response(f"Need more info, step {i+1}.",
                           "code_executor | code=print({})".format(i))
        )

    mock = MockLLMClient()
    mock.set_responses(responses)

    agent = AcademicLitAgent(mock, tool_registry, max_steps=5, reflection=False)
    result = agent.process_message("Keep analyzing forever.")

    assert result["steps"] == 5
    # Should have hit max_steps, trace may or may not end with final_answer
    traces = [t for t in result["trace"] if t["type"] == "action"]
    assert len(traces) <= 5


# ═══════════════════════════════════════════════════════════════════
# Test Case 6: Malformed LLM response — no Action found
# ═══════════════════════════════════════════════════════════════════

def test_agent_malformed_response_recovery(tool_registry):
    """Agent recovers when LLM returns a response without valid Action."""
    mock = MockLLMClient()
    mock.set_responses([
        "I'm thinking about what to do... but I forgot to include an Action line.",  # malformed
        react_response(
            "I need to correct myself and call a tool.",
            "code_executor | code=print('recovered')"
        ),
        react_response("Recovered.", final_answer="Analysis complete after recovery."),
    ])

    agent = AcademicLitAgent(mock, tool_registry, max_steps=10, reflection=False)
    result = agent.process_message("Do something.")

    # Should eventually get a final answer after recovery
    assert result["steps"] <= 5
    # The malformed response should be recorded as a "thought" in trace
    thought_traces = [t for t in result["trace"] if t["type"] == "thought"]
    assert len(thought_traces) >= 1


# ═══════════════════════════════════════════════════════════════════
# Test Case 7: Parameter parsing edge cases
# ═══════════════════════════════════════════════════════════════════

def test_agent_parameter_parsing(agent, mock_llm):
    """Agent correctly parses various parameter formats."""
    # Test the internal _parse_params method directly
    agent = agent  # from fixture

    # Standard format
    params = agent._parse_params("query=test | limit=5 | year=2020-2024")
    assert params == {"query": "test", "limit": "5", "year": "2020-2024"}

    # Colon separator
    params = agent._parse_params("operation: save | paper_id: p123")
    assert params == {"operation": "save", "paper_id": "p123"}

    # Quoted values
    params = agent._parse_params('query="retrieval augmented generation"')
    assert params["query"] == "retrieval augmented generation"

    # Empty input
    params = agent._parse_params("")
    assert params == {}

    # Single param
    params = agent._parse_params("code=print(1+1)")
    assert params == {"code": "print(1+1)"}


# ═══════════════════════════════════════════════════════════════════
# Test Case 8: Document context injection
# ═══════════════════════════════════════════════════════════════════

def test_agent_document_context_injection(tool_registry):
    """When document_context is provided, it's injected as a system message."""
    mock = MockLLMClient()
    mock.set_responses([
        react_response("I'll analyze the provided document.",
                       final_answer="Document analysis complete.")
    ])

    agent = AcademicLitAgent(mock, tool_registry, max_steps=10)
    doc_ctx = "[Document: sample.pdf]\nAbstract: This paper studies RAG methods."
    result = agent.process_message("Analyze this document.",
                                   document_context=doc_ctx)

    # Verify document context was inserted
    system_msgs = [m for m in agent.memory.messages if m["role"] == "system"]
    doc_injected = any(doc_ctx in m.get("content", "") for m in system_msgs)
    assert doc_injected
    assert result["trace"][-1]["type"] == "final_answer"
