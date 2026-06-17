"""Main Streamlit app entry point."""

import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.agent.tool_registry import ToolRegistry
from src.models.model_manager import ModelManager
from src.tools import (
    CitationTraverserTool,
    CodeExecutorTool,
    LiteratureComparatorTool,
    ReadPaperTool,
    ResearchGapAnalyzerTool,
    SQLitePaperDBTool,
    SearchPapersTool,
)


TOOL_REGISTRY_VERSION = "2026-06-17.4"


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        SearchPapersTool(),
        ReadPaperTool(),
        SQLitePaperDBTool(),
        LiteratureComparatorTool(),
        ResearchGapAnalyzerTool(),
        CitationTraverserTool(),
        CodeExecutorTool(),
    ):
        registry.register(tool)
    return registry


def init_session_state() -> None:
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if (
        "tool_registry" not in st.session_state
        or st.session_state.get("tool_registry_version") != TOOL_REGISTRY_VERSION
    ):
        st.session_state.tool_registry = build_tool_registry()
        st.session_state.tool_registry_version = TOOL_REGISTRY_VERSION
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "uploaded_papers" not in st.session_state:
        st.session_state.uploaded_papers = []
    if "current_doc" not in st.session_state:
        st.session_state.current_doc = None
    if "current_doc_name" not in st.session_state:
        st.session_state.current_doc_name = ""
    if "current_model" not in st.session_state:
        models = st.session_state.model_manager.list_models()
        st.session_state.current_model = models[0].name if models else ""
    if "task_model" not in st.session_state:
        st.session_state.task_model = st.session_state.current_model
    if "message_model" not in st.session_state:
        st.session_state.message_model = st.session_state.current_model
    if "model_policy" not in st.session_state:
        st.session_state.model_policy = "允许单次切换"


st.set_page_config(page_title="学术文献分析助手", page_icon="📚", layout="wide")
init_session_state()

pg = st.navigation(
    [
        st.Page("pages/chat.py", title="文献工作台", icon=":material/forum:"),
        st.Page("pages/model_config.py", title="模型配置", icon=":material/tune:"),
    ],
    position="sidebar",
)

pg.run()
