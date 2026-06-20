"""Shared fixtures for the academic-lit-agent test suite."""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.llm_client import LLMClient
from src.agent.tool_registry import ToolRegistry, ToolSpec, BaseTool
from src.agent.core import AcademicLitAgent
from src.tools import (
    SearchPapersTool,
    ReadPaperTool,
    SQLitePaperDBTool,
    LiteratureComparatorTool,
    ResearchGapAnalyzerTool,
    CitationTraverserTool,
    CodeExecutorTool,
)


# ═══════════════════════════════════════════════════════════════════
# Mock LLM Client — returns pre-programmed responses from a queue
# ═══════════════════════════════════════════════════════════════════

class MockLLMClient:
    """LLM client that returns canned responses for deterministic testing."""

    def __init__(self, api_key="mock-key", base_url="http://mock", model_name="mock-model"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.responses: list[str] = []
        self.call_history: list[list[dict]] = []
        self._response_index = 0

    def set_responses(self, responses: list[str]):
        """Pre-load a sequence of responses to return on successive chat() calls."""
        self.responses = list(responses)
        self._response_index = 0

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: int = 4096) -> str:
        self.call_history.append(messages)
        if self._response_index < len(self.responses):
            reply = self.responses[self._response_index]
            self._response_index += 1
            return reply
        # Default fallback: return a Final Answer
        return (
            "Thought: I have no more prepared responses.\n"
            "Final Answer: The analysis is complete based on available information."
        )

    def system_message(self, text: str) -> dict:
        return {"role": "system", "content": text}

    def user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def assistant_message(self, text: str) -> dict:
        return {"role": "assistant", "content": text}

    def count_tokens(self, text: str) -> int:
        return len(text) // 2


# ═══════════════════════════════════════════════════════════════════
# Helper: build canned ReAct responses
# ═══════════════════════════════════════════════════════════════════

def react_response(thought: str, action: str | None = None,
                   final_answer: str | None = None) -> str:
    """Build a well-formed ReAct response string."""
    if final_answer is not None:
        return f"Thought: {thought}\nFinal Answer: {final_answer}"
    return f"Thought: {thought}\nAction: {action}"


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_llm():
    """Create a MockLLMClient with no pre-loaded responses."""
    return MockLLMClient()


@pytest.fixture
def tool_registry():
    """Create a ToolRegistry with all 7 production tools registered."""
    reg = ToolRegistry()
    reg.register(SearchPapersTool())
    reg.register(ReadPaperTool())
    reg.register(SQLitePaperDBTool())
    reg.register(LiteratureComparatorTool())
    reg.register(ResearchGapAnalyzerTool())
    reg.register(CitationTraverserTool())
    reg.register(CodeExecutorTool())
    return reg


@pytest.fixture
def agent(mock_llm, tool_registry):
    """Create an AcademicLitAgent with mock LLM and real tools."""
    return AcademicLitAgent(
        llm_client=mock_llm,
        tool_registry=tool_registry,
        max_steps=10,
        reflection=True,
    )


@pytest.fixture
def temp_paper_dir():
    """Create a temporary directory with sample paper files."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create a sample text paper
        txt_path = os.path.join(tmp, "sample_paper.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_PAPER_TEXT)
        # Create a sample markdown paper
        md_path = os.path.join(tmp, "sample_paper.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_PAPER_MD)
        yield tmp


@pytest.fixture
def sample_papers_json():
    """Return a JSON string of sample paper metadata for comparator/gap tools."""
    return json.dumps(SAMPLE_PAPERS)


# ═══════════════════════════════════════════════════════════════════
# Shared test data
# ═══════════════════════════════════════════════════════════════════

SAMPLE_PAPER_TEXT = """Abstract
This paper presents a novel approach to retrieval augmented generation for low-resource languages.
We propose a cross-lingual retrieval method that improves performance by 15%.

Introduction
Retrieval Augmented Generation (RAG) has become a standard paradigm for grounding
large language models in external knowledge. However, most RAG systems are designed
for English, leaving low-resource languages underserved.

Related Work
Previous work on cross-lingual retrieval includes dense retrieval models like mDPR
and knowledge distillation approaches.

Method
We propose CrossLing-RAG, a framework that combines multilingual dense retrieval
with language-adaptive generation. Our method uses a two-stage pipeline: first,
a multilingual retriever based on mContriever, and second, a generator fine-tuned
on language-specific instruction data.

Results
Experiments on five low-resource languages show 15% improvement over baseline.
F1 score improved from 0.62 to 0.71 on average across languages.

Conclusion
CrossLing-RAG demonstrates that cross-lingual retrieval augmented generation is
feasible for low-resource languages, opening new directions for multilingual NLP.

References
[1] Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020.
"""

SAMPLE_PAPER_MD = """# A Survey of Efficient Fine-Tuning Methods

## Abstract
We survey parameter-efficient fine-tuning (PEFT) methods for large language models,
comparing LoRA, prompt tuning, and adapter-based approaches.

## Introduction
Fine-tuning large language models is computationally expensive. Parameter-efficient
methods aim to reduce this cost while maintaining performance.

## Method
We evaluate three PEFT families: (1) LoRA-based methods including LoRA, QLoRA, AdaLoRA;
(2) prompt tuning methods including soft prompts and prefix tuning;
(3) adapter methods including bottleneck adapters and parallel adapters.

## Results
On the MMLU benchmark, LoRA achieves 95% of full fine-tuning performance with only
0.1% of trainable parameters. QLoRA further reduces memory by 4× with comparable accuracy.

## Conclusion
PEFT methods, particularly LoRA variants, offer an excellent cost-performance trade-off
for adapting LLMs to downstream tasks.

## References
[1] Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, ICLR 2022.
"""

SAMPLE_PAPERS = [
    {
        "paper_id": "abc123",
        "title": "CrossLing-RAG for Low-Resource Languages",
        "method": "CrossLing-RAG, multilingual dense retrieval",
        "dataset": "FLORES-200, XNLI",
        "metrics": "F1=0.71, accuracy=0.84",
        "results": "15% improvement over mDPR baseline on F1",
        "key_finding": "Cross-lingual retrieval improves low-resource RAG by 15%",
        "year": 2024,
    },
    {
        "paper_id": "def456",
        "title": "Efficient RAG with Small Language Models",
        "method": "Retrieval distillation, small model fine-tuning",
        "dataset": "Natural Questions, TriviaQA",
        "metrics": "EM=0.65, F1=0.72",
        "results": "Small model RAG reaches 90% of large model performance",
        "key_finding": "Small models with good retrieval match large models",
        "year": 2024,
    },
    {
        "paper_id": "ghi789",
        "title": "Cross-Lingual Dense Retrieval: A Comprehensive Survey",
        "method": "Cross-lingual dense retrieval, mDPR, mContriever",
        "dataset": "CLEF, MLQA, XOR-TyDi",
        "metrics": "MRR=0.58, recall@100=0.85",
        "results": "Dense models outperform sparse baselines by 20% in cross-lingual setting",
        "key_finding": "Dense cross-lingual retrieval benefits from pre-training scale",
        "year": 2023,
    },
]
