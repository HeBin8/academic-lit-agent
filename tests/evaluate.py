#!/usr/bin/env python3
"""Evaluation harness for the Academic Literature Analysis Agent.

Runs 20 standardized test scenarios and records:
  - Task completion rate (Final Answer produced?)
  - Tool call accuracy (correct tool selected? correct params?)
  - End-to-end response latency (wall-clock time per scenario)
  - Token consumption (estimated input + output tokens)

Generates:
  - tests/results/evaluation_report.json  — machine-readable metrics
  - tests/results/evaluation_report.md   — human-readable report
  - tests/results/evaluation_report.txt  — plain-text summary

Usage:
    python -m tests.evaluate              # run all scenarios
    python -m tests.evaluate --quick      # run subset (8 scenarios)
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.core import AcademicLitAgent
from src.agent.tool_registry import ToolRegistry
from src.tools import (
    SearchPapersTool, ReadPaperTool, SQLitePaperDBTool,
    LiteratureComparatorTool, ResearchGapAnalyzerTool,
    CitationTraverserTool, CodeExecutorTool,
)
from tests.conftest import MockLLMClient, react_response


# ═══════════════════════════════════════════════════════════════════
# Scenario definitions
# ═══════════════════════════════════════════════════════════════════
#
# Each scenario has:
#   id          — unique identifier
#   name        — short description
#   category    — agent_core | tool | memory | integration
#   user_input  — what the user says
#   llm_responses — pre-programmed LLM responses for the mock
#   expected_tools — list of tool names expected to be called (or [])
#   expected_keywords — keywords expected in the final answer
#   mock_api    — if True, mock Semantic Scholar API
# ═══════════════════════════════════════════════════════════════════

SCENARIOS = [
    # ── Agent Core (5 scenarios) ──────────────────────────────────
    {
        "id": "S01",
        "name": "Greeting — direct answer (no tool)",
        "category": "agent_core",
        "user_input": "Hello! What can you do?",
        "llm_responses": [
            react_response("Simple greeting, no tools needed.",
                          final_answer="Hello! I am an academic literature analysis assistant. "
                                       "I can search, read, compare, and analyze research papers.")
        ],
        "expected_tools": [],
        "expected_keywords": ["academic", "literature", "papers"],
        "mock_api": False,
    },
    {
        "id": "S02",
        "name": "Single tool call — code execution",
        "category": "agent_core",
        "user_input": "Calculate the mean of [10, 20, 30, 40, 50].",
        "llm_responses": [
            react_response("Let me calculate.",
                          "code_executor | code=import statistics\\ndata = [10,20,30,40,50]\\nprint(statistics.mean(data))"),
            react_response("Got the result.",
                          final_answer="The mean of the dataset is 30.0."),
        ],
        "expected_tools": ["code_executor"],
        "expected_keywords": ["30"],
        "mock_api": False,
    },
    {
        "id": "S03",
        "name": "Multi-tool chaining — save and verify",
        "category": "agent_core",
        "user_input": "Save a paper titled 'Test Paper 2024' and verify it's in the database.",
        "llm_responses": [
            react_response("Saving paper to DB.",
                          "sqlite_paper_db | operation=save | paper_id=eval_s03 | title=Test Paper 2024 | year=2024"),
            react_response("Verifying it's saved.",
                          "sqlite_paper_db | operation=list"),
            react_response("Verified.",
                          final_answer="Paper 'Test Paper 2024' (eval_s03) saved and verified."),
        ],
        "expected_tools": ["sqlite_paper_db"],
        "expected_keywords": ["Test Paper", "saved"],
        "mock_api": False,
    },
    {
        "id": "S04",
        "name": "Reflection after multiple steps",
        "category": "agent_core",
        "user_input": "First run code A, then run code B.",
        "llm_responses": [
            react_response("Step 1.", "code_executor | code=print('A')"),
            react_response("Step 2 with reflection.", "code_executor | code=print('B')"),
            react_response("Both done.", final_answer="Executed both steps: A and B."),
        ],
        "expected_tools": ["code_executor"],
        "expected_keywords": ["A", "B"],
        "mock_api": False,
    },
    {
        "id": "S05",
        "name": "Malformed response recovery",
        "category": "agent_core",
        "user_input": "Help me analyze something.",
        "llm_responses": [
            "Hmm, let me think... (forgot Action format)",  # malformed
            react_response("Let me retry properly.",
                          "code_executor | code=print('recovered')"),
            react_response("Done.", final_answer="Recovered from formatting error."),
        ],
        "expected_tools": ["code_executor"],
        "expected_keywords": ["Recovered"],
        "mock_api": False,
    },

    # ── Tools (7 scenarios, one per tool) ─────────────────────────
    {
        "id": "S06",
        "name": "Search papers via Semantic Scholar",
        "category": "tool",
        "user_input": "Search for 'retrieval augmented generation' papers.",
        "llm_responses": [
            react_response("Searching.",
                          "search_papers | query=retrieval augmented generation | limit=5"),
            react_response("Results found.",
                          final_answer="Found 3 papers about RAG. The top result is by Lewis et al."),
        ],
        "expected_tools": ["search_papers"],
        "expected_keywords": ["Found", "paper"],
        "mock_api": True,
    },
    {
        "id": "S07",
        "name": "Read local paper file",
        "category": "tool",
        "user_input": "Read and summarize the paper at /tmp/sample_paper.txt.",
        "llm_responses": [
            react_response("Reading paper.",
                          "read_paper | path=/tmp/sample_paper.txt"),
            react_response("Paper read.",
                          final_answer="The paper proposes a cross-lingual RAG framework with 15% improvement."),
        ],
        "expected_tools": ["read_paper"],
        "expected_keywords": ["cross-lingual", "RAG", "15%"],
        "mock_api": False,
    },
    {
        "id": "S08",
        "name": "SQLite paper database CRUD",
        "category": "tool",
        "user_input": "List all papers and show database statistics.",
        "llm_responses": [
            react_response("Listing papers.",
                          "sqlite_paper_db | operation=list"),
            react_response("Getting stats.",
                          "sqlite_paper_db | operation=stats"),
            react_response("Here is the summary.",
                          final_answer="Database contains papers. See stats above for details."),
        ],
        "expected_tools": ["sqlite_paper_db"],
        "expected_keywords": ["paper", "Database"],
        "mock_api": False,
    },
    {
        "id": "S09",
        "name": "Literature comparison table",
        "category": "tool",
        "user_input": "Compare these papers: Paper A (LoRA, 95% acc) vs Paper B (Adapter, 88% acc).",
        "llm_responses": [
            react_response("Building comparison.",
                          'literature_comparator | papers_json=[{"paper_id":"a","title":"Paper A","method":"LoRA","results":"95% accuracy"},{"paper_id":"b","title":"Paper B","method":"Adapter","results":"88% accuracy"}]'),
            react_response("Comparison ready.",
                          final_answer="Comparison shows LoRA outperforms Adapter by 7% on this benchmark."),
        ],
        "expected_tools": ["literature_comparator"],
        "expected_keywords": ["LoRA", "Adapter", "7%"],
        "mock_api": False,
    },
    {
        "id": "S10",
        "name": "Research gap analysis",
        "category": "tool",
        "user_input": "Analyze research gaps among these 3 RAG papers.",
        "llm_responses": [
            react_response("Analyzing gaps.",
                          'research_gap_analyzer | papers_json=[{"paper_id":"a","title":"RAG-A","method":"RAG","key_finding":"RAG improves QA","year":2024},{"paper_id":"b","title":"RAG-B","method":"RAG+RL","key_finding":"RL enhances retrieval","year":2024},{"paper_id":"c","title":"RAG-C","method":"RAG","key_finding":"RAG needs better retrieval","year":2023}]'),
            react_response("Gaps identified.",
                          final_answer="Key gaps: limited domain adaptation, efficiency metrics missing, and language diversity is narrow."),
        ],
        "expected_tools": ["research_gap_analyzer"],
        "expected_keywords": ["gap", "domain", "efficiency"],
        "mock_api": False,
    },
    {
        "id": "S11",
        "name": "Citation graph traversal",
        "category": "tool",
        "user_input": "Show me papers that cite 'Attention Is All You Need'.",
        "llm_responses": [
            react_response("Traversing citations.",
                          "citation_traverser | paper_id=abc123 | direction=citations | depth=1 | limit=5"),
            react_response("Citations found.",
                          final_answer="Found 3 citing papers. The most cited is 'BERT' by Devlin et al."),
        ],
        "expected_tools": ["citation_traverser"],
        "expected_keywords": ["citing", "BERT", "Found"],
        "mock_api": True,
    },
    {
        "id": "S12",
        "name": "Code executor with data analysis",
        "category": "tool",
        "user_input": "Plot a histogram of [1,2,2,3,3,3,4,4,5] and calculate basic stats.",
        "llm_responses": [
            react_response("Running analysis.",
                          "code_executor | code=import statistics\\ndata=[1,2,2,3,3,3,4,4,5]\\nprint(f'Mean: {statistics.mean(data)}')"),
            react_response("Stats computed.",
                          final_answer="Mean=3.0, median=3.0, mode=3. The data is approximately symmetric."),
        ],
        "expected_tools": ["code_executor"],
        "expected_keywords": ["3.0", "mean"],
        "mock_api": False,
    },

    # ── Integration (8 scenarios) ─────────────────────────────────
    {
        "id": "S13",
        "name": "Full literature review workflow",
        "category": "integration",
        "user_input": "Do a complete literature review on RAG: search papers, compare, and find gaps.",
        "llm_responses": [
            react_response("Searching for RAG papers.",
                          "search_papers | query=retrieval augmented generation survey | limit=5"),
            react_response("Comparing results.",
                          'literature_comparator | papers_json=[{"paper_id":"a","title":"RAG-A","method":"RAG","results":"F1=0.85"},{"paper_id":"b","title":"RAG-B","method":"RAG+RL","results":"F1=0.89"}]'),
            react_response("Finding gaps.",
                          'research_gap_analyzer | papers_json=[{"paper_id":"a","title":"RAG-A","method":"RAG","key_finding":"RAG is effective","year":2024},{"paper_id":"b","title":"RAG-B","method":"RAG+RL","key_finding":"RL boosts RAG","year":2024}]'),
            react_response("Complete.",
                          final_answer="Literature review complete. RAG is effective; key gaps in efficiency and domain adaptation."),
        ],
        "expected_tools": ["search_papers", "literature_comparator", "research_gap_analyzer"],
        "expected_keywords": ["RAG", "review", "gap"],
        "mock_api": True,
    },
    {
        "id": "S14",
        "name": "Multi-turn research session",
        "category": "integration",
        "user_input": "First find papers about LoRA fine-tuning.",
        "llm_responses": [
            react_response("Searching LoRA.",
                          "search_papers | query=LoRA fine-tuning large language models | limit=5"),
            react_response("Results.",
                          final_answer="Found papers on LoRA. Key finding: LoRA achieves 95% of full fine-tuning with 0.1% parameters."),
        ],
        "expected_tools": ["search_papers"],
        "expected_keywords": ["LoRA", "95%", "0.1%"],
        "mock_api": True,
    },
    {
        "id": "S15",
        "name": "Error recovery — file not found → search fallback",
        "category": "integration",
        "user_input": "Read the paper at /missing/paper.pdf.",
        "llm_responses": [
            react_response("Trying to read the file.",
                          "read_paper | path=/missing/paper.pdf"),
            react_response("File not found! Let me search online instead.",
                          "search_papers | query=paper about retrieval augmented generation | limit=3"),
            react_response("Recovered.",
                          final_answer="The file was not found, but I searched online and found 3 related papers."),
        ],
        "expected_tools": ["read_paper", "search_papers"],
        "expected_keywords": ["not found", "searched", "online"],
        "mock_api": True,
    },
    {
        "id": "S16",
        "name": "Code → data analysis → interpretation",
        "category": "integration",
        "user_input": "Run a t-test on two groups: A=[2,3,5,7] B=[1,4,6,8] and interpret significance.",
        "llm_responses": [
            react_response("Running t-test.",
                          "code_executor | code=import statistics\\na=[2,3,5,7]\\nb=[1,4,6,8]\\nprint(f'Mean A: {statistics.mean(a)}, Mean B: {statistics.mean(b)}')"),
            react_response("Interpreting.",
                          final_answer="Group A mean=4.25, Group B mean=4.75. The difference is not statistically significant with this small sample."),
        ],
        "expected_tools": ["code_executor"],
        "expected_keywords": ["4.25", "4.75", "significant"],
        "mock_api": False,
    },
    {
        "id": "S17",
        "name": "Paper DB: save → tag → favorite → retrieve",
        "category": "integration",
        "user_input": "Save paper 'Deep Learning Survey', tag it 'important', mark favorite, then show favorites.",
        "llm_responses": [
            react_response("Saving.",
                          "sqlite_paper_db | operation=save | paper_id=eval_s17 | title=Deep Learning Survey | year=2024 | authors=LeCun"),
            react_response("Tagging.",
                          "sqlite_paper_db | operation=tag | paper_id=eval_s17 | tag=important"),
            react_response("Favoriting.",
                          "sqlite_paper_db | operation=favorite | paper_id=eval_s17"),
            react_response("Showing favorites.",
                          "sqlite_paper_db | operation=favorites"),
            react_response("Done.",
                          final_answer="Paper 'Deep Learning Survey' saved, tagged 'important', and marked as favorite."),
        ],
        "expected_tools": ["sqlite_paper_db"],
        "expected_keywords": ["Deep Learning", "important", "favorite"],
        "mock_api": False,
    },
    {
        "id": "S18",
        "name": "Citation co-occurrence analysis",
        "category": "integration",
        "user_input": "Find papers that cite both paper X and paper Y.",
        "llm_responses": [
            react_response("Searching co-citations.",
                          "citation_traverser | paper_id=paperX | operation=co_occurrence | other_paper_id=paperY | limit=10"),
            react_response("Results.",
                          final_answer="Found 2 papers that cite both paper X and paper Y, suggesting a common research thread."),
        ],
        "expected_tools": ["citation_traverser"],
        "expected_keywords": ["cite", "both", "common"],
        "mock_api": True,
    },
    {
        "id": "S19",
        "name": "Document upload + multi-tool analysis",
        "category": "integration",
        "user_input": "I uploaded a paper. Read it, save to DB, and compare with known literature.",
        "llm_responses": [
            react_response("Reading uploaded paper.",
                          "read_paper | path=/tmp/uploaded_paper.txt"),
            react_response("Saving to DB.",
                          "sqlite_paper_db | operation=save | paper_id=eval_s19 | title=Uploaded Paper | year=2024"),
            react_response("Comparing.",
                          'literature_comparator | papers_json=[{"paper_id":"eval_s19","title":"Uploaded Paper","method":"New Method","results":"F1=0.90"},{"paper_id":"ref","title":"Reference Paper","method":"Baseline","results":"F1=0.82"}]'),
            react_response("Analysis complete.",
                          final_answer="Uploaded paper analyzed, saved, and compared. It outperforms the baseline by 8% on F1."),
        ],
        "expected_tools": ["read_paper", "sqlite_paper_db", "literature_comparator"],
        "expected_keywords": ["uploaded", "compared", "8%"],
        "mock_api": False,
    },
    {
        "id": "S20",
        "name": "Max steps limit — graceful termination",
        "category": "agent_core",
        "user_input": "Do an exhaustive analysis of all RAG methods.",
        "llm_responses": [
            react_response("Step 1.", "code_executor | code=print(1)"),
            react_response("Step 2.", "code_executor | code=print(2)"),
            react_response("Step 3.", "code_executor | code=print(3)"),
            react_response("Step 4.", "code_executor | code=print(4)"),
            react_response("Step 5.", "code_executor | code=print(5)"),
        ],
        "expected_tools": ["code_executor"],
        "expected_keywords": [],
        "mock_api": False,
    },
]


# ═══════════════════════════════════════════════════════════════════
# Mock API responses
# ═══════════════════════════════════════════════════════════════════

MOCK_SEARCH_API = {
    "data": [
        {
            "paperId": "s2_001", "title": "RAG Survey 2024",
            "year": 2024,
            "authors": [{"name": "Smith"}, {"name": "Jones"}],
            "abstract": "Comprehensive survey of RAG methods.",
            "citationCount": 500, "url": "", "venue": "ACL",
            "publicationDate": "2024-01-01",
        },
        {
            "paperId": "s2_002", "title": "Efficient RAG with Small Models",
            "year": 2024,
            "authors": [{"name": "Lee"}],
            "abstract": "Small models achieve 90% of large model RAG performance.",
            "citationCount": 200, "url": "", "venue": "EMNLP",
            "publicationDate": "2024-03-01",
        },
        {
            "paperId": "s2_003", "title": "Cross-Lingual RAG",
            "year": 2023,
            "authors": [{"name": "Chen"}, {"name": "Wang"}],
            "abstract": "Cross-lingual RAG for low-resource languages.",
            "citationCount": 150, "url": "", "venue": "NAACL",
            "publicationDate": "2023-06-01",
        },
    ]
}

MOCK_CITATION_API = {
    "data": [
        {
            "citingPaper": {
                "paperId": "cite_001", "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "year": 2019,
                "authors": [{"name": "Devlin"}, {"name": "Chang"}],
                "citationCount": 95000, "url": "",
            }
        },
        {
            "citingPaper": {
                "paperId": "cite_002", "title": "GPT-3: Language Models are Few-Shot Learners",
                "year": 2020,
                "authors": [{"name": "Brown"}],
                "citationCount": 40000, "url": "",
            }
        },
        {
            "citingPaper": {
                "paperId": "cite_003", "title": "T5: Exploring Limits of Transfer Learning",
                "year": 2020,
                "authors": [{"name": "Raffel"}],
                "citationCount": 25000, "url": "",
            }
        },
    ]
}


# ═══════════════════════════════════════════════════════════════════
# Evaluation engine
# ═══════════════════════════════════════════════════════════════════

def _make_registry() -> ToolRegistry:
    """Create a fresh tool registry with all 7 tools."""
    reg = ToolRegistry()
    for cls in [SearchPapersTool, ReadPaperTool, SQLitePaperDBTool,
                LiteratureComparatorTool, ResearchGapAnalyzerTool,
                CitationTraverserTool, CodeExecutorTool]:
        reg.register(cls())
    return reg


def evaluate_scenario(scenario: dict) -> dict:
    """Run a single scenario and return evaluation metrics."""
    result = {
        "id": scenario["id"],
        "name": scenario["name"],
        "category": scenario["category"],
        "completed": False,
        "tool_call_correct": False,
        "keywords_matched": 0,
        "keywords_expected": len(scenario.get("expected_keywords", [])),
        "latency_ms": 0,
        "tokens_input_est": 0,
        "tokens_output_est": 0,
        "steps_used": 0,
        "tools_called": [],
        "errors": [],
    }

    reg = _make_registry()
    mock = MockLLMClient()
    mock.set_responses(scenario["llm_responses"])

    agent = AcademicLitAgent(
        llm_client=mock,
        tool_registry=reg,
        max_steps=5 if scenario["id"] == "S20" else 10,
        reflection=True,
    )

    # ── run with timing ───────────────────────────────────────────
    t_start = time.perf_counter()

    try:
        with patch("urllib.request.urlopen") as mock_urlopen:
            # Configure mock API based on tool
            def _api_side_effect(req, **kwargs):
                mock_resp = MagicMock()
                url = req.full_url if hasattr(req, 'full_url') else str(req)
                if "paper/search" in url:
                    mock_resp.read.return_value = json.dumps(MOCK_SEARCH_API).encode()
                elif "/citations" in url or "/references" in url:
                    mock_resp.read.return_value = json.dumps(MOCK_CITATION_API).encode()
                else:
                    mock_resp.read.return_value = json.dumps(MOCK_SEARCH_API).encode()
                mock_resp.__enter__.return_value = mock_resp
                mock_resp.status = 200
                return mock_resp

            mock_urlopen.side_effect = _api_side_effect

            agent_result = agent.process_message(scenario["user_input"])

        t_end = time.perf_counter()
        result["latency_ms"] = round((t_end - t_start) * 1000, 1)

        # ── evaluate completion ───────────────────────────────────
        final_trace = agent_result.get("trace", [])
        has_final = any(t.get("type") == "final_answer" for t in final_trace)
        result["completed"] = has_final
        result["steps_used"] = agent_result.get("steps", 0)
        result["tools_called"] = agent_result.get("tools_called", [])

        if not has_final:
            result["errors"].append("No Final Answer produced")

        # ── evaluate tool call accuracy ───────────────────────────
        expected_tools = set(scenario.get("expected_tools", []))
        actual_tools = set(agent_result.get("tools_called", []))

        if expected_tools:
            # All expected tools must be called
            missing = expected_tools - actual_tools
            if not missing:
                result["tool_call_correct"] = True
            else:
                result["errors"].append(f"Missing tool calls: {missing}")
        else:
            # No tools expected
            result["tool_call_correct"] = len(actual_tools) == 0
            if actual_tools:
                result["errors"].append(f"Unexpected tool calls: {actual_tools}")

        # ── evaluate keyword matching ─────────────────────────────
        final_text = agent_result.get("response", "").lower()
        for kw in scenario.get("expected_keywords", []):
            if kw.lower() in final_text:
                result["keywords_matched"] += 1

        # ── estimate tokens ───────────────────────────────────────
        # Input: all messages sent to LLM (system prompt + history)
        input_chars = sum(len(m["content"]) for m in agent.memory.messages)
        result["tokens_input_est"] = input_chars // 2

        # Output: LLM response tokens
        output_chars = sum(len(r) for r in scenario["llm_responses"])
        result["tokens_output_est"] = output_chars // 2

    except Exception as e:
        t_end = time.perf_counter()
        result["latency_ms"] = round((t_end - t_start) * 1000, 1)
        result["errors"].append(f"Exception: {type(e).__name__}: {str(e)[:200]}")

    return result


# ═══════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════

def generate_reports(results: list[dict], output_dir: Path):
    """Generate JSON, Markdown, and plain-text evaluation reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(results)
    completed = sum(1 for r in results if r["completed"])
    tool_correct = sum(1 for r in results if r["tool_call_correct"])
    total_keywords_matched = sum(r["keywords_matched"] for r in results)
    total_keywords_expected = sum(r["keywords_expected"] for r in results)
    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0
    total_input_tokens = sum(r["tokens_input_est"] for r in results)
    total_output_tokens = sum(r["tokens_output_est"] for r in results)
    total_steps = sum(r["steps_used"] for r in results)

    # Completion rate
    completion_rate = completed / total * 100 if total else 0
    # Tool accuracy
    tool_accuracy = tool_correct / total * 100 if total else 0
    # Keyword precision
    keyword_precision = total_keywords_matched / total_keywords_expected * 100 if total_keywords_expected else 100

    # ── JSON report ───────────────────────────────────────────────
    json_report = {
        "evaluation_metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_scenarios": total,
            "agent_version": "1.0.0",
            "max_steps_default": 10,
            "reflection_enabled": True,
        },
        "summary": {
            "completion_rate": round(completion_rate, 1),
            "tool_call_accuracy": round(tool_accuracy, 1),
            "keyword_match_precision": round(keyword_precision, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "total_input_tokens_est": total_input_tokens,
            "total_output_tokens_est": total_output_tokens,
            "total_tokens_est": total_input_tokens + total_output_tokens,
            "total_steps_across_scenarios": total_steps,
            "avg_steps_per_scenario": round(total_steps / total, 1) if total else 0,
        },
        "by_category": {},
        "scenarios": results,
    }

    # Per-category breakdown
    for cat in sorted(set(r["category"] for r in results)):
        cat_results = [r for r in results if r["category"] == cat]
        cat_total = len(cat_results)
        json_report["by_category"][cat] = {
            "count": cat_total,
            "completed": sum(1 for r in cat_results if r["completed"]),
            "tool_correct": sum(1 for r in cat_results if r["tool_call_correct"]),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in cat_results) / cat_total, 1),
        }

    json_path = output_dir / "evaluation_report.json"
    json_path.write_text(json.dumps(json_report, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    # ── Markdown report ───────────────────────────────────────────
    md_lines = [
        "# Academic Literature Agent — Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Scenarios:** {total}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value | Target | Status |",
        "|--------|-------|--------|--------|",
        f"| Task Completion Rate | {completion_rate:.1f}% | ≥ 90% | {'[PASS]' if completion_rate >= 90 else '[FAIL]'} |",
        f"| Tool Call Accuracy | {tool_accuracy:.1f}% | ≥ 85% | {'[PASS]' if tool_accuracy >= 85 else '[FAIL]'} |",
        f"| Keyword Match Precision | {keyword_precision:.1f}% | ≥ 80% | {'[PASS]' if keyword_precision >= 80 else '[FAIL]'} |",
        f"| Avg Response Latency | {avg_latency:.0f} ms | < 5000 ms | {'[PASS]' if avg_latency < 5000 else '[FAIL]'} |",
        f"| Avg Steps per Scenario | {total_steps/total:.1f} | ≤ 10 | {'[PASS]' if total_steps/total <= 10 else '[FAIL]'} |",
        f"| Total Token Consumption | {total_input_tokens + total_output_tokens:,} | — | — |",
        "",
        "## Token Consumption",
        "",
        f"| Type | Estimated Tokens |",
        f"|------|-----------------:|",
        f"| Input (prompts + history) | {total_input_tokens:,} |",
        f"| Output (LLM responses) | {total_output_tokens:,} |",
        f"| **Total** | **{total_input_tokens + total_output_tokens:,}** |",
        "",
        "## Per-Category Results",
        "",
        "| Category | Count | Completed | Tool Accurate | Avg Latency |",
        "|----------|-------|-----------|---------------|-------------|",
    ]

    for cat, cat_data in json_report["by_category"].items():
        md_lines.append(
            f"| {cat} | {cat_data['count']} | {cat_data['completed']}/{cat_data['count']} "
            f"| {cat_data['tool_correct']}/{cat_data['count']} "
            f"| {cat_data['avg_latency_ms']:.0f} ms |"
        )

    md_lines.extend([
        "",
        "## Detailed Scenario Results",
        "",
        "| ID | Name | Category | Completed | Tools Correct | Keywords | Latency | Steps |",
        "|----|------|----------|-----------|---------------|----------|---------|-------|",
    ])

    for r in results:
        kws = f"{r['keywords_matched']}/{r['keywords_expected']}"
        status = "[PASS]" if r["completed"] and r["tool_call_correct"] else "[WARN]"
        if not r["completed"]:
            status = "[FAIL]"
        md_lines.append(
            f"| {r['id']} | {r['name']} | {r['category']} | {status} "
            f"| {'[PASS]' if r['tool_call_correct'] else '[FAIL]'} "
            f"| {kws} | {r['latency_ms']:.0f}ms | {r['steps_used']} |"
        )

    # Add errors section
    failed = [r for r in results if r["errors"]]
    if failed:
        md_lines.extend([
            "",
            "## Issues & Errors",
            "",
        ])
        for r in failed:
            md_lines.append(f"### {r['id']}: {r['name']}")
            for err in r["errors"]:
                md_lines.append(f"- {err}")
            md_lines.append("")

    md_path = output_dir / "evaluation_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # ── Plain-text report ─────────────────────────────────────────
    txt_lines = [
        "=" * 60,
        "  Academic Literature Agent — Evaluation Report",
        "=" * 60,
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Scenarios: {total}",
        "",
        "--- Summary ---",
        f"  Task Completion Rate:   {completion_rate:.1f}%  ({completed}/{total})",
        f"  Tool Call Accuracy:     {tool_accuracy:.1f}%  ({tool_correct}/{total})",
        f"  Keyword Match:          {keyword_precision:.1f}%  ({total_keywords_matched}/{total_keywords_expected})",
        f"  Avg Latency:            {avg_latency:.0f} ms",
        f"  Total Tokens (est):     {total_input_tokens + total_output_tokens:,}",
        f"  Avg Steps/Scenario:     {total_steps/total:.1f}",
        "",
        "--- Per-Scenario ---",
    ]

    for r in results:
        status = "PASS" if r["completed"] and r["tool_call_correct"] else "FAIL"
        txt_lines.append(
            f"  [{status}] {r['id']} {r['name']} "
            f"| tools={r['tools_called']} "
            f"| kws={r['keywords_matched']}/{r['keywords_expected']} "
            f"| {r['latency_ms']:.0f}ms"
        )
        if r["errors"]:
            for err in r["errors"]:
                txt_lines.append(f"         ERROR: {err}")

    txt_path = output_dir / "evaluation_report.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    return json_report


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Academic Literature Agent across standard test scenarios."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a quick subset (8 scenarios) instead of all 20."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for reports (default: tests/results/)."
    )
    args = parser.parse_args()

    # Select scenarios
    scenarios = SCENARIOS
    if args.quick:
        # Pick evenly from all categories
        indices = [0, 2, 5, 8, 11, 13, 16, 19]
        scenarios = [SCENARIOS[i] for i in indices if i < len(SCENARIOS)]

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Academic Literature Agent — Evaluation Suite")
    print(f"{'='*60}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Output:    {output_dir}")
    print(f"{'='*60}\n")

    # ── Run all scenarios ─────────────────────────────────────────
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i:2d}/{len(scenarios)}] {scenario['id']} {scenario['name']}...", end=" ", flush=True)
        eval_result = evaluate_scenario(scenario)
        results.append(eval_result)

        status = "PASS" if eval_result["completed"] and eval_result["tool_call_correct"] else "FAIL"
        print(f"{status} ({eval_result['latency_ms']:.0f}ms, {eval_result['steps_used']} steps)")

    # ── Generate reports ──────────────────────────────────────────
    report = generate_reports(results, output_dir)

    # ── Print summary ─────────────────────────────────────────────
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Completion Rate:     {s['completion_rate']:.1f}%")
    print(f"  Tool Call Accuracy:  {s['tool_call_accuracy']:.1f}%")
    print(f"  Keyword Precision:   {s['keyword_match_precision']:.1f}%")
    print(f"  Avg Latency:         {s['avg_latency_ms']:.0f} ms")
    print(f"  Total Tokens:        {s['total_tokens_est']:,}")
    print(f"  Avg Steps:           {s['avg_steps_per_scenario']:.1f}")
    print(f"{'='*60}")
    print(f"  Reports saved to: {output_dir}")
    print(f"    - evaluation_report.json")
    print(f"    - evaluation_report.md")
    print(f"    - evaluation_report.txt")
    print(f"{'='*60}\n")

    # Return non-zero exit code if completion rate < 80%
    if s["completion_rate"] < 80:
        sys.exit(1)


if __name__ == "__main__":
    main()
