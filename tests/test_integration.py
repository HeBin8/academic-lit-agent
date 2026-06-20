"""End-to-end integration tests: multi-turn conversations, multi-tool workflows,
document context injection, and error recovery across the full agent stack.

Covers:
  - #23  Full literature search → read → compare workflow
  - #24  Multi-turn conversation continuity
  - #25  Error recovery: tool failure → retry with different tool
  - #26  Document upload and analysis workflow
  - #27  Code execution → data analysis workflow
  - #28  Search → save → tag → search_by_tag workflow
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from tests.conftest import (
    MockLLMClient, react_response, SAMPLE_PAPERS, SAMPLE_PAPER_TEXT
)
from src.agent.core import AcademicLitAgent
from src.agent.llm_client import LLMClient
from src.agent.tool_registry import ToolRegistry
from src.tools import (
    SearchPapersTool, ReadPaperTool, SQLitePaperDBTool,
    LiteratureComparatorTool, ResearchGapAnalyzerTool,
    CitationTraverserTool, CodeExecutorTool,
)
from src.memory.conversation_memory import ConversationMemory


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_agent(responses: list[str], max_steps: int = 10,
                reflection: bool = True) -> tuple[MockLLMClient, AcademicLitAgent]:
    """Build an agent with a mock LLM pre-loaded with responses."""
    mock = MockLLMClient()
    mock.set_responses(responses)
    reg = ToolRegistry()
    for cls in [SearchPapersTool, ReadPaperTool, SQLitePaperDBTool,
                LiteratureComparatorTool, ResearchGapAnalyzerTool,
                CitationTraverserTool, CodeExecutorTool]:
        reg.register(cls())
    agent = AcademicLitAgent(mock, reg, max_steps=max_steps, reflection=reflection)
    return mock, agent


# ═══════════════════════════════════════════════════════════════════
# Test Case 23: Full search → compare → gap analysis workflow
# ═══════════════════════════════════════════════════════════════════

MOCK_SEARCH_API = {
    "data": [
        {
            "paperId": "p1", "title": "RAG Method A", "year": 2024,
            "authors": [{"name": "Alice"}], "abstract": "Improving RAG.",
            "citationCount": 100, "url": "", "venue": "ACL", "publicationDate": "2024-01-01",
        },
        {
            "paperId": "p2", "title": "RAG Method B", "year": 2024,
            "authors": [{"name": "Bob"}], "abstract": "Efficient RAG.",
            "citationCount": 80, "url": "", "venue": "EMNLP", "publicationDate": "2024-02-01",
        },
        {
            "paperId": "p3", "title": "RAG Survey", "year": 2023,
            "authors": [{"name": "Charlie"}], "abstract": "A survey of RAG.",
            "citationCount": 200, "url": "", "venue": "NAACL", "publicationDate": "2023-06-01",
        },
    ]
}


def test_full_literature_analysis_workflow():
    """Search → collect results → run comparison and gap analysis."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(MOCK_SEARCH_API).encode()
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        mock, agent = _make_agent([
            # Step 1: Search for papers
            react_response(
                "I'll search for RAG papers.",
                "search_papers | query=retrieval augmented generation | limit=5"
            ),
            # Step 2: Compare found papers
            react_response(
                "Good, found 3 papers. Let me compare them.",
                f"literature_comparator | papers_json={json.dumps(SAMPLE_PAPERS)}"
            ),
            # Step 3: Analyze gaps
            react_response(
                "Now let me analyze research gaps.",
                f"research_gap_analyzer | papers_json={json.dumps(SAMPLE_PAPERS)}"
            ),
            # Step 4: Final answer
            react_response(
                "I have completed the analysis.",
                final_answer=(
                    "Based on the search, comparison, and gap analysis:\n"
                    "1. Cross-lingual RAG is an active area with 15% improvements reported.\n"
                    "2. Small model RAG achieves 90% of large model performance.\n"
                    "3. Key gaps: limited language diversity, domain adaptation."
                )
            ),
        ])

        result = agent.process_message(
            "Search for recent RAG papers, compare them, and identify research gaps."
        )

        assert result["trace"][-1]["type"] == "final_answer"
        assert len(result["tools_called"]) >= 2
        assert "Cross-lingual" in result["response"] or "RAG" in result["response"]


# ═══════════════════════════════════════════════════════════════════
# Test Case 24: Multi-turn conversation continuity
# ═══════════════════════════════════════════════════════════════════

def test_multiturn_conversation():
    """Agent maintains context across multiple process_message calls."""
    mock, agent = _make_agent([
        # Turn 1
        react_response("First search.", "code_executor | code=print('turn1')"),
        react_response("Turn 1 done.", final_answer="I found 5 papers about RAG."),

        # Turn 2 (references previous context)
        react_response("I remember the previous search. Let me filter.",
                       final_answer="Among the 5 papers I found earlier, 3 focus on cross-lingual RAG."),
    ])

    # Turn 1
    r1 = agent.process_message("Search for RAG papers.")
    assert "5 papers" in r1["response"]
    assert "code_executor" in r1["tools_called"]

    # Turn 2 — the agent's memory should contain the history
    r2 = agent.process_message("Filter to cross-lingual only.")
    assert "cross-lingual" in r2["response"].lower() or "3" in r2["response"]
    assert r2["trace"][-1]["type"] == "final_answer"


# ═══════════════════════════════════════════════════════════════════
# Test Case 25: Error recovery — tool failure → retry with different tool
# ═══════════════════════════════════════════════════════════════════

def test_error_recovery_tool_failure():
    """When a tool fails, the agent tries an alternative approach."""
    mock, agent = _make_agent([
        # Step 1: Try a tool that will fail (bad path)
        react_response(
            "Let me read this paper.",
            "read_paper | path=/nonexistent/paper.pdf"
        ),
        # Step 2: Try an alternative approach after failure
        react_response(
            "The file wasn't found. Let me search for the paper online instead.",
            "search_papers | query=retrieval augmented generation cross-lingual | limit=3"
        ),
        # Step 3: Final answer
        react_response(
            "Found papers matching the topic despite file read failure.",
            final_answer="The local file was not found, but I searched online and found 3 relevant papers about cross-lingual RAG."
        ),
    ])

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(MOCK_SEARCH_API).encode()
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        result = agent.process_message("Read the paper at /nonexistent/paper.pdf and analyze it.")

    # Should recover and get a final answer
    assert result["trace"][-1]["type"] == "final_answer"
    # Should have at least one observation trace (the error from read_paper)
    observations = [t for t in result["trace"] if t["type"] == "observation"]
    assert len(observations) >= 1
    # The first observation should mention file not found
    assert "not found" in observations[0]["result"].lower() or "File" in observations[0]["result"]


# ═══════════════════════════════════════════════════════════════════
# Test Case 26: Document upload → analysis workflow
# ═══════════════════════════════════════════════════════════════════

def test_document_analysis_workflow(temp_paper_dir):
    """Upload a document, then the agent reads and analyzes it."""
    txt_path = os.path.join(temp_paper_dir, "sample_paper.txt")

    mock, agent = _make_agent([
        # Step 1: Read the uploaded paper
        react_response(
            "I'll read the uploaded paper to extract its content.",
            f"read_paper | path={txt_path}"
        ),
        # Step 2: Provide analysis
        react_response(
            "Successfully read the paper. Now I can summarize.",
            final_answer=(
                "The paper proposes CrossLing-RAG, a framework combining multilingual "
                "dense retrieval with language-adaptive generation, achieving 15% "
                "improvement on five low-resource languages."
            )
        ),
    ])

    doc_ctx = f"[Document: sample_paper.txt]\nPath: {txt_path}"
    result = agent.process_message(
        "Analyze the uploaded paper.",
        document_context=doc_ctx
    )

    assert result["trace"][-1]["type"] == "final_answer"
    assert "15%" in result["response"] or "CrossLing" in result["response"]
    # Document context should be in the conversation
    doc_messages = [m for m in agent.memory.messages
                    if "sample_paper.txt" in m.get("content", "")]
    assert len(doc_messages) >= 1


# ═══════════════════════════════════════════════════════════════════
# Test Case 27: Code execution → data analysis workflow
# ═══════════════════════════════════════════════════════════════════

def test_code_execution_data_analysis():
    """Agent runs code to analyze data and interprets results."""
    mock, agent = _make_agent([
        # Step 1: Calculate statistics
        react_response(
            "Let me calculate descriptive statistics.",
            "code_executor | code=import statistics\\ndata = [23, 45, 67, 32, 89, 54]\\nprint(f'Mean: {statistics.mean(data):.1f}')\\nprint(f'Stdev: {statistics.stdev(data):.1f}')"
        ),
        # Step 2: Final answer interpreting results
        react_response(
            "I have the statistics. Let me interpret them.",
            final_answer=(
                "The dataset has 6 values. The mean is approximately 51.7 and "
                "the standard deviation is approximately 23.6, indicating moderate variability."
            )
        ),
    ])

    result = agent.process_message(
        "Calculate basic statistics for data: [23, 45, 67, 32, 89, 54]"
    )

    assert result["trace"][-1]["type"] == "final_answer"
    # The observation from code execution should contain the output
    observations = [t for t in result["trace"] if t["type"] == "observation"]
    code_output = observations[0]["result"] if observations else ""
    assert "51" in code_output or "Mean" in code_output or "Code execution completed" in code_output


# ═══════════════════════════════════════════════════════════════════
# Test Case 28: Search → save → tag → search_by_tag workflow
# ═══════════════════════════════════════════════════════════════════

def test_paper_management_workflow():
    """Full pipeline: search papers → save to DB → tag → retrieve by tag."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(MOCK_SEARCH_API).encode()
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        mock, agent = _make_agent([
            # Step 1: Search
            react_response("Searching.", "search_papers | query=RAG | limit=3"),
            # Step 2: Save found papers to DB
            react_response(
                "Saving papers.",
                "sqlite_paper_db | operation=save | paper_id=p1 | title=RAG Method A | year=2024 | authors=Alice"
            ),
            # Step 3: Tag them
            react_response(
                "Adding tag.",
                "sqlite_paper_db | operation=tag | paper_id=p1 | tag=RAG"
            ),
            # Step 4: Verify by tag
            react_response(
                "Verifying.",
                "sqlite_paper_db | operation=search_by_tag | tag=RAG"
            ),
            # Step 5: Final answer
            react_response(
                "Papers found and tagged.",
                final_answer="Paper 'RAG Method A' has been saved and tagged as 'RAG'. Retrieved successfully."
            ),
        ])

        result = agent.process_message(
            "Find recent RAG papers, save them, and tag as 'RAG'."
        )

        assert result["trace"][-1]["type"] == "final_answer"
        # tools_called is deduplicated — search_papers + sqlite_paper_db = 2 unique tools
        # but the trace shows 5 action steps total
        assert "search_papers" in result["tools_called"]
        assert "sqlite_paper_db" in result["tools_called"]
        # Verify multiple steps were taken
        assert len(result["trace"]) >= 3

        # Cleanup
        reg = agent.tools
        reg.execute({"tool": "sqlite_paper_db",
                     "kwargs": {"operation": "delete", "paper_id": "p1"}})
