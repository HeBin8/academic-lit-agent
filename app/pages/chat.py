# -*- coding: utf-8 -*-
"""Literature research workspace."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import html
import json
import os
import pathlib
import re
import sys
from datetime import datetime

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.agent.core import AcademicLitAgent
from src.agent.llm_client import LLMClient
from src.agent.tool_registry import ToolRegistry
from src.tools import (
    CitationTraverserTool,
    CodeExecutorTool,
    LiteratureComparatorTool,
    ReadPaperTool,
    ResearchGapAnalyzerTool,
    SQLitePaperDBTool,
    SearchPapersTool,
)
from src.tools.read_paper import PARSERS, _detect_format, _extract_sections


PAPER_DIR = pathlib.Path("data/papers")
TOOL_REGISTRY_VERSION = "2026-06-17.4"
NO_LLM = "无需 LLM"
AUTO_MODEL = "自动选择"


def ensure_tool_registry() -> None:
    if (
        "tool_registry" in st.session_state
        and st.session_state.get("tool_registry_version") == TOOL_REGISTRY_VERSION
    ):
        return
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
    st.session_state.tool_registry = registry
    st.session_state.tool_registry_version = TOOL_REGISTRY_VERSION


def active_model_names() -> list[str]:
    return [m.name for m in st.session_state.model_manager.list_models()]


def model_options(include_auto: bool = False, include_no_llm: bool = False) -> list[str]:
    options = active_model_names()
    if include_auto:
        options = [AUTO_MODEL] + options
    if include_no_llm:
        options = options + [NO_LLM]
    return options or ([NO_LLM] if include_no_llm else [])


def normalize_model_choice(choice: str | None) -> str:
    models = active_model_names()
    if choice in models:
        return choice
    if st.session_state.get("current_model") in models:
        return st.session_state.current_model
    return models[0] if models else ""


def model_index(options: list[str], value: str | None) -> int:
    if value in options:
        return options.index(value)
    return 0


def get_llm_client(model_name: str | None = None) -> LLMClient | None:
    name = normalize_model_choice(model_name)
    config = st.session_state.model_manager.get_model(name)
    if not config:
        return None
    return LLMClient(
        api_key=config.api_key or "",
        base_url=config.base_url or "",
        model_name=config.model_name or "",
    )


def get_agent(model_name: str | None = None) -> AcademicLitAgent | None:
    client = get_llm_client(model_name)
    if not client:
        return None
    return AcademicLitAgent(llm_client=client, tool_registry=st.session_state.tool_registry)


def page_css() -> None:
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            height: 0;
            min-height: 0;
            background: transparent;
        }
        section.main > div {
            padding-top: 0 !important;
        }
        .block-container {
            padding-top: 0.35rem;
            padding-bottom: 112px;
            max-width: 1480px;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #dfe5ee;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.75rem;
        }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #ebeff5;
            border-radius: 8px;
            padding: 10px 12px;
            box-shadow: 0 4px 18px rgba(30, 41, 59, 0.03);
        }
        div[data-testid="stMetric"] label {
            color: #697386;
            font-size: 12px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 21px;
            font-weight: 780;
        }
        .topbar-title h1 {
            margin: 0;
            font-size: 28px;
            line-height: 1.2;
            color: #192232;
        }
        .topbar-title p {
            margin: 5px 0 0;
            color: #697386;
            font-size: 14px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #17b26a;
            display: inline-block;
        }
        .primary-action {
            height: 38px;
            border-radius: 7px;
            background: #1f6feb;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            white-space: nowrap;
            margin-top: 28px;
        }
        .section-card {
            background: #ffffff;
            border: 1px solid #ebeff5;
            border-radius: 8px;
            box-shadow: 0 16px 40px rgba(30, 41, 59, 0.08);
            overflow: hidden;
        }
        div[data-testid="stForm"] {
            position: fixed;
            left: max(330px, calc((100vw - 1480px) / 2 + 300px + 80px));
            bottom: 16px;
            width: min(720px, calc((100vw - 360px) * 0.52));
            height: 74px !important;
            min-height: 74px !important;
            max-height: 74px !important;
            overflow: visible !important;
            z-index: 999;
            border: 1px solid #dfe3ea !important;
            border-radius: 28px !important;
            padding: 12px 14px !important;
            background: #ffffff !important;
            box-shadow: 0 12px 36px rgba(15, 23, 42, 0.13) !important;
        }
        div[data-testid="stForm"] [data-testid="column"] {
            display: flex;
            align-items: center;
        }
        div[data-testid="stForm"] [data-testid="stVerticalBlock"],
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
            gap: 0.35rem !important;
            min-height: 0 !important;
        }
        div[data-testid="stForm"] div[data-testid="stTextInput"] input {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
            font-size: 14px;
        }
        div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
            border-radius: 50% !important;
            min-width: 46px !important;
            width: 46px !important;
            height: 46px !important;
            padding: 0 !important;
        }
        div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
            border-radius: 50% !important;
            min-width: 38px !important;
            width: 38px !important;
            height: 38px !important;
            padding: 0 !important;
            font-size: 22px !important;
        }
        div[data-testid="stForm"] div[data-baseweb="select"] {
            border-radius: 20px !important;
            background: #f6f7f9 !important;
        }
        .scroll-paper {
            height: 640px;
            overflow-y: auto;
            background: #f0f2f6;
            border-radius: 8px;
            padding: 14px;
            border: 1px solid #e5eaf2;
        }
        .translation-scroll {
            height: 640px;
            overflow-y: auto;
            border: 1px solid #e5eaf2;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            line-height: 1.7;
        }
        .panel-head {
            height: 48px;
            border-bottom: 1px solid #ebeff5;
            padding: 0 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            background: #ffffff;
        }
        .panel-head h3 {
            margin: 0;
            font-size: 15px;
            color: #192232;
        }
        .panel-head span {
            color: #697386;
            font-size: 12px;
        }
        .tool-card {
            border: 1px solid #dfe5ee;
            border-radius: 8px;
            padding: 10px;
            background: #ffffff;
            min-height: 92px;
        }
        .tool-card strong {
            display: block;
            font-size: 13px;
            color: #192232;
            margin-bottom: 4px;
        }
        .tool-card p {
            min-height: 32px;
            margin: 0 0 8px;
            color: #697386;
            font-size: 12px;
            line-height: 1.35;
        }
        .message-card {
            border: 1px solid #ebeff5;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            background: #ffffff;
        }
        .message-card.user {
            background: #eef4ff;
            border-color: #c9ddff;
            margin-left: 9%;
        }
        .message-card.assistant {
            margin-right: 9%;
        }
        .message-meta {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #8a95a6;
            font-size: 11px;
            margin-bottom: 7px;
            flex-wrap: wrap;
        }
        .model-badge {
            border: 1px solid #cdd8ea;
            background: #f8fbff;
            border-radius: 999px;
            padding: 2px 7px;
            color: #2459a6;
            font-weight: 700;
            font-size: 11px;
        }
        .paper-chip {
            border: 1px solid #ebeff5;
            background: #f9fafc;
            border-radius: 8px;
            padding: 9px 10px;
            margin-bottom: 8px;
        }
        .paper-chip.active {
            border-color: #9bbcf5;
            background: #f5f9ff;
        }
        .paper-title {
            color: #192232;
            font-size: 13px;
            font-weight: 760;
            line-height: 1.3;
        }
        .paper-meta {
            margin-top: 5px;
            display: flex;
            justify-content: space-between;
            color: #8a95a6;
            font-size: 11px;
        }
        .tag {
            border-radius: 999px;
            padding: 3px 7px;
            background: #eef4ff;
            color: #2459a6;
            font-size: 11px;
            font-weight: 650;
            display: inline-block;
            margin: 6px 4px 0 0;
        }
        .tag.teal {
            background: #e9f7f4;
            color: #087968;
        }
        .tag.amber {
            background: #fff3dc;
            color: #915c00;
        }
        .reader-shell {
            background: #f0f2f6;
            border-radius: 8px;
            padding: 14px;
            min-height: 560px;
        }
        .inspector-card {
            border: 1px solid #ebeff5;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            background: #fbfcfe;
        }
        .inspector-card h4 {
            margin: 0 0 8px;
            font-size: 13px;
            color: #192232;
        }
        .kv-row {
            display: grid;
            grid-template-columns: 56px 1fr;
            gap: 8px;
            font-size: 12px;
            color: #697386;
            margin-bottom: 6px;
            line-height: 1.35;
        }
        .kv-row b {
            color: #344054;
        }
        .hint-box {
            border-left: 3px solid #af6b00;
            background: #fff8ea;
            padding: 7px 8px;
            border-radius: 5px;
            color: #6f4700;
            font-size: 12px;
            margin-bottom: 7px;
            line-height: 1.4;
        }
        .hint-box.teal {
            border-left-color: #118b7a;
            background: #eefaf7;
            color: #0f665b;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 7px;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            border-radius: 7px;
        }
        @media (max-width: 1100px) {
            div[data-testid="stForm"] {
                left: 16px;
                right: 16px;
                width: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_uploaded_file(uploaded) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    path = PAPER_DIR / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    record = {
        "name": uploaded.name,
        "path": str(path),
        "timestamp": datetime.now().strftime("%m-%d %H:%M"),
    }
    existing = next((p for p in st.session_state.uploaded_papers if p["name"] == uploaded.name), None)
    if existing:
        existing.update(record)
    else:
        st.session_state.uploaded_papers.append(record)
    st.session_state.current_doc = str(path)
    st.session_state.current_doc_name = uploaded.name
    st.session_state.doc_translation = ""


def parse_document(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return ""
    cache_key = f"parsed::{path}::{os.path.getmtime(path)}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    fmt = _detect_format(path)
    parser = PARSERS.get(fmt)
    if not parser:
        return f"[Unsupported format: {fmt}]"
    text = parser(path)
    st.session_state[cache_key] = text
    return text


def current_document_context(limit: int = 5000) -> str:
    path = st.session_state.get("current_doc")
    name = st.session_state.get("current_doc_name", "")
    text = parse_document(path)
    if not text or text.startswith("["):
        return ""
    return f"[Current document: {name}]\n{text[:limit]}"


def paper_record_from_path(path: str) -> dict:
    text = parse_document(path)
    sections = _extract_sections(text) if text and not text.startswith("[") else {}
    basename = os.path.basename(path)
    title = _clean(sections.get("title") or basename)
    method = _clean(sections.get("method") or _window(text, ["method", "approach", "framework", "model"]))
    results = _clean(sections.get("results") or _window(text, ["result", "experiment", "evaluation"]))
    return {
        "paper_id": basename,
        "title": title[:160] or basename,
        "method": method[:280],
        "dataset": _extract_mentions(text, ["dataset", "benchmark", "corpus"])[:180],
        "metrics": _extract_mentions(text, ["accuracy", "f1", "precision", "recall", "auc", "latency"])[:180],
        "results": results[:280],
        "key_finding": (results or method)[:280],
        "year": _guess_year(text),
    }


def uploaded_paper_records() -> list[dict]:
    return [
        paper_record_from_path(p["path"])
        for p in st.session_state.uploaded_papers
        if os.path.exists(p["path"])
    ]


def save_ui_session() -> None:
    if not st.session_state.messages:
        return
    session_id = st.session_state.get("current_session_id")
    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.current_session_id = session_id
    first_user = next((m["content"] for m in st.session_state.messages if m.get("role") == "user"), "新对话")
    st.session_state.sessions[session_id] = {
        "title": first_user[:24],
        "timestamp": datetime.now().strftime("%m-%d %H:%M"),
        "messages": list(st.session_state.messages),
    }


def run_agent_prompt(prompt: str, model_name: str | None = None) -> None:
    selected_model = normalize_model_choice(model_name)
    st.session_state.messages.append({"role": "user", "content": prompt, "model": selected_model})
    agent = get_agent(selected_model)
    if not agent:
        st.session_state.messages.append(
            {"role": "assistant", "content": "请先在“模型配置”页面添加并启用一个可用的大模型。", "model": selected_model}
        )
        save_ui_session()
        return
    try:
        result = agent.process_message(prompt, document_context=current_document_context())
        trace = result.get("trace") or result.get("推理过程") or []
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("response", "已完成，但没有生成可显示的回答。"),
                "trace": trace,
                "tools_called": result.get("tools_called", []),
                "model": selected_model,
            }
        )
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"运行时出现错误：`{type(exc).__name__}: {exc}`",
                "model": selected_model,
            }
        )
    save_ui_session()


def append_tool_result(title: str, content: str, model_name: str = NO_LLM, tool_name: str = "") -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"### {title}\n\n{content}",
            "model": model_name,
            "tools_called": [tool_name] if tool_name else [],
        }
    )
    save_ui_session()


def run_llm_structure_for_current_doc() -> None:
    path = st.session_state.get("current_doc")
    name = st.session_state.get("current_doc_name", "")
    text = parse_document(path)
    if not path or not text:
        st.session_state["last_structure_status"] = "请先选择论文。"
        return
    st.session_state[_doc_key(path, "llm_structure")] = llm_structure_document(text, name)
    st.session_state["last_structure_status"] = "结构化解析已生成。"


def run_full_translation_for_current_doc() -> None:
    """Callback: just flags that translation is pending. Main flow does the work."""
    st.session_state["_translate_pending"] = True


def _execute_full_translation(path: str, text: str) -> None:
    """Run the actual translation in the main render flow (not inside a callback).
    Uses st.status() for progress — safe in the main script execution context."""
    key = _doc_key(path, "full_translation")
    client = get_llm_client(st.session_state.get("current_model"))
    if not client:
        st.session_state[key] = "请先配置大模型。"
        st.session_state["last_translation_status"] = "翻译失败：未配置模型"
        return
    clean_text = _clean_for_translation(text)
    chunks = _split_text(clean_text, 2800)
    if not chunks:
        st.session_state[key] = "没有可翻译的文本。"
        st.session_state["last_translation_status"] = "没有可翻译的文本"
        return

    translated = []
    with st.status(f"正在翻译全文（共 {len(chunks)} 段）...", expanded=True) as status:
        for index, chunk in enumerate(chunks, 1):
            status.write(f"正在翻译第 {index}/{len(chunks)} 段...")
            prompt = (
                "将下面的学术论文正文完整翻译为中文。要求：\n"
                "- 保留术语、公式、引用编号和小节编号。\n"
                "- 不要总结，不要省略，不要添加解释。\n"
                "- 只输出译文。\n\n"
                f"第 {index}/{len(chunks)} 段：\n{chunk}"
            )
            try:
                translated.append(client.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.1, max_tokens=3500))
            except Exception as exc:
                translated.append(f"\n\n[第 {index} 段翻译失败：{exc}]\n\n{chunk}")
        status.update(label="全文翻译完成！", state="complete")

    st.session_state[key] = "\n\n".join(translated)
    st.session_state["last_translation_status"] = "全文翻译已完成。"


def render_trace(trace: list[dict]) -> None:
    for step in trace:
        kind = step.get("type")
        if kind == "action":
            st.caption(f"步骤 {step.get('step')}: 调用工具 {step.get('tool')}")
            st.json(step.get("params", {}))
        elif kind == "observation":
            st.caption(f"步骤 {step.get('step')}: 工具返回")
            st.code(step.get("result", "")[:1200])
        elif kind == "thought":
            st.caption(f"步骤 {step.get('step')}: 思考")
            st.write(step.get("content", "")[:800])
        elif kind == "final_answer":
            st.caption(f"步骤 {step.get('step')}: 最终回答")


def render_message(message: dict) -> None:
    role = message.get("role", "assistant")
    model = message.get("model") or st.session_state.get("current_model", "")
    tools_called = message.get("tools_called") or []
    meta = [f'<span class="model-badge">{html.escape(model or NO_LLM)}</span>']
    if tools_called:
        meta.append(f"调用 {'、'.join(html.escape(str(t)) for t in tools_called)}")
    elif role == "user":
        meta.append("本条消息")
    else:
        meta.append("模型回答")
    st.markdown(
        f'<div class="message-card {html.escape(role)}"><div class="message-meta">{"".join(meta)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(message.get("content", ""))
    trace = message.get("trace")
    if trace:
        with st.expander("查看推理过程", expanded=False):
            render_trace(trace)
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 论文资产")
        uploads = st.file_uploader(
            "上传 PDF / Word / Markdown",
            type=["pdf", "docx", "doc", "md", "txt"],
            accept_multiple_files=True,
            key="paper_uploads",
        )
        if uploads:
            # Only process files we haven't seen yet (avoids re-processing on rerun)
            if "_processed_uploads" not in st.session_state:
                st.session_state["_processed_uploads"] = set()
            new_count = 0
            for uploaded in uploads:
                if uploaded.name not in st.session_state["_processed_uploads"]:
                    add_uploaded_file(uploaded)
                    st.session_state["_processed_uploads"].add(uploaded.name)
                    new_count += 1
            if new_count:
                st.success(f"已加入 {new_count} 个文件")

        if st.session_state.uploaded_papers:
            for paper in list(reversed(st.session_state.uploaded_papers)):
                active = st.session_state.get("current_doc_name") == paper["name"]
                chip_class = "paper-chip active" if active else "paper-chip"
                st.markdown(
                    f"""
                    <div class="{chip_class}">
                      <div class="paper-title">{html.escape(paper["name"])}</div>
                      <div class="paper-meta"><span>{html.escape(paper["timestamp"])}</span><span>{'当前' if active else '可选择'}</span></div>
                      <span class="tag">论文</span>{'<span class="tag teal">已选择</span>' if active else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cols = st.columns([3, 1])
                if cols[0].button("打开", key=f"open_{paper['name']}", use_container_width=True):
                    st.session_state.current_doc = paper["path"]
                    st.session_state.current_doc_name = paper["name"]
                    st.rerun()
                if cols[1].button("×", key=f"delete_{paper['name']}", help="删除记录"):
                    st.session_state.uploaded_papers = [p for p in st.session_state.uploaded_papers if p["name"] != paper["name"]]
                    if active:
                        st.session_state.current_doc = None
                        st.session_state.current_doc_name = ""
                    st.rerun()
        else:
            st.caption("还没有上传论文。")

        st.divider()
        st.markdown("### 可用模型")
        models = active_model_names()
        if models:
            default_index = model_index(models, st.session_state.get("current_model"))
            st.session_state.current_model = st.selectbox("默认模型", models, index=default_index, key="global_model_select")
            if st.session_state.get("message_model") not in models:
                st.session_state.message_model = st.session_state.current_model
            if st.session_state.get("task_model") not in models:
                st.session_state.task_model = st.session_state.current_model
            for name in models:
                active = name == st.session_state.current_model
                st.markdown(
                    f"""
                    <div class="paper-chip {'active' if active else ''}">
                      <div class="paper-title"><span class="status-dot"></span> {html.escape(name)}</div>
                      <div class="paper-meta"><span>{'默认' if active else '在线'}</span><span>可用于任务</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("还没有启用模型。")

        st.divider()
        st.markdown("### 工作区")
        st.caption("当前项目：数字孪生综述")
        st.caption(f"默认模型：{st.session_state.get('current_model') or '未配置'}")
        st.caption(f"模型策略：{st.session_state.get('model_policy', '允许单次切换')}")

        st.divider()
        st.markdown("### 快捷入口")
        if st.button("模型配置", use_container_width=True):
            st.switch_page("pages/model_config.py")
        if st.button("论文库统计", use_container_width=True):
            out = st.session_state.tool_registry.execute({"tool": "sqlite_paper_db", "kwargs": {"operation": "stats"}})
            out += "\n\n" + st.session_state.tool_registry.execute({"tool": "sqlite_paper_db", "kwargs": {"operation": "list"}})
            append_tool_result("论文库统计", out, NO_LLM, "sqlite_paper_db")
        if st.button("导出报告", use_container_width=True):
            st.session_state.pop("_draft_msg_count", None)  # force refresh
            report = _get_or_generate_draft()
            append_tool_result("分析报告", report, NO_LLM, "")

        # Download button — always visible, auto-caches via _get_or_generate_draft
        st.download_button(
            label="下载文献综述草稿",
            data=_get_or_generate_draft(),
            file_name=f"文献综述草稿_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.divider()
        st.markdown("### 对话")
        if st.button("新建对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.rerun()
        for sid, session in reversed(list(st.session_state.sessions.items())):
            if st.button(session["title"], key=f"session_{sid}", use_container_width=True):
                st.session_state.messages = session["messages"]
                st.session_state.current_session_id = sid
                st.rerun()


def render_topbar() -> None:
    left, default_col, policy_col, save_col, action_col = st.columns([0.95, 0.4, 0.4, 0.28, 0.42], vertical_alignment="top")
    with left:
        st.markdown(
            """
            <div class="topbar-title">
              <h1>文献研究工作台</h1>
              <p>默认模型管理全局任务；单条消息、工具调用和长任务可以临时指定模型。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    models = active_model_names()
    with default_col:
        if models:
            idx = model_index(models, st.session_state.get("current_model"))
            st.session_state.current_model = st.selectbox("默认模型", models, index=idx, key="top_default_model")
        else:
            st.selectbox("默认模型", ["未配置"], disabled=True)
    with policy_col:
        policies = ["允许单次切换", "始终使用默认模型"]
        st.session_state.model_policy = st.selectbox(
            "任务策略",
            policies,
            index=model_index(policies, st.session_state.get("model_policy")),
            key="top_model_policy",
        )
    with save_col:
        if st.button("保存会话", use_container_width=True):
            save_ui_session()
            st.success("已保存")
    with action_col:
        if st.button("生成综述草稿", type="primary", use_container_width=True):
            st.session_state.pending_prompt = {
                "text": "请基于当前论文、分析流和已上传论文，生成一份文献综述草稿提纲，包含研究背景、方法对比、研究空白和可写作小节。",
                "model": normalize_model_choice(st.session_state.get("task_model")),
            }
            st.rerun()


def render_metrics() -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("当前论文", len(st.session_state.uploaded_papers), "已上传")
    metric_cols[1].metric("已调用工具", len(st.session_state.tool_registry.list_tools()), "全部可用")
    metric_cols[2].metric("当前消息", len(st.session_state.messages), "分析流")
    metric_cols[3].metric("本条模型", st.session_state.get("message_model") or "未配置", "可临时切换")


def render_tool_cards() -> None:
    st.markdown('<div class="panel-head"><h3>分析流</h3><span>每次工具调用都会记录使用的模型</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="padding: 12px 14px; background: #fbfcfe; border-bottom: 1px solid #ebeff5;">', unsafe_allow_html=True)
    cards = st.columns(4)
    labels = [
        ("搜索论文", "关键词、年份、领域和分页", "search_model", True, True),
        ("多论文对比", "方法、数据集、指标、结果", "compare_model", True, False),
        ("研究空白", "共识、矛盾、空白、趋势", "gap_model", True, False),
        ("引用图", "BFS / DFS 与共现关系", "citation_model", True, True),
    ]
    for col, (title, desc, key, include_auto, include_no_llm) in zip(cards, labels):
        with col:
            st.markdown(f'<div class="tool-card"><strong>{title}</strong><p>{desc}</p>', unsafe_allow_html=True)
            options = model_options(include_auto=include_auto, include_no_llm=include_no_llm)
            default = AUTO_MODEL if include_auto else st.session_state.get("task_model")
            if title == "引用图":
                default = NO_LLM
            st.selectbox("模型", options, index=model_index(options, st.session_state.get(key, default)), key=key, label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_tool_console() -> None:
    tools = st.session_state.tool_registry
    with st.expander("展开工具参数", expanded=False):
        tab_search, tab_analysis, tab_citation, tab_code = st.tabs(["搜索", "分析", "引用", "代码"])
        with tab_search:
            q = st.text_input("关键词", placeholder="retrieval augmented generation")
            cols = st.columns([1, 1, 1, 2])
            limit = cols[0].number_input("数量", 1, 50, 5)
            offset = cols[1].number_input("分页", 0, 500, 0, step=5)
            year = cols[2].text_input("年份", placeholder="2020-2026")
            field = cols[3].text_input("领域", placeholder="Computer Science")
            if st.button("搜索论文", use_container_width=True):
                if not q.strip():
                    st.warning("请输入关键词。")
                else:
                    out = tools.execute(
                        {
                            "tool": "search_papers",
                            "kwargs": {
                                "query": q,
                                "limit": limit,
                                "offset": offset,
                                "year": year,
                                "fields_of_study": field,
                            },
                        }
                    )
                    append_tool_result("论文搜索结果", out, st.session_state.get("search_model", AUTO_MODEL), "search_papers")
                    st.rerun()
        with tab_analysis:
            cols = st.columns(3)
            if cols[0].button("总结当前论文", use_container_width=True, disabled=not st.session_state.get("current_doc")):
                st.session_state.pending_prompt = {
                    "text": "请阅读当前论文，概括标题、研究问题、方法、实验结果和主要贡献。",
                    "model": normalize_model_choice(st.session_state.get("task_model")),
                }
                st.rerun()
            if cols[1].button("对比已上传论文", use_container_width=True, disabled=len(st.session_state.uploaded_papers) < 2):
                paths = [p["path"] for p in st.session_state.uploaded_papers]
                out = tools.execute(
                    {
                        "tool": "literature_comparator",
                        "kwargs": {"paper_paths_json": json.dumps(paths, ensure_ascii=False), "topic": "已上传论文对比"},
                    }
                )
                append_tool_result("多论文对比", out, st.session_state.get("compare_model", AUTO_MODEL), "literature_comparator")
                st.rerun()
            if cols[2].button("研究空白分析", use_container_width=True, disabled=len(st.session_state.uploaded_papers) < 2):
                records = uploaded_paper_records()
                out = tools.execute({"tool": "research_gap_analyzer", "kwargs": {"papers_json": json.dumps(records, ensure_ascii=False)}})
                append_tool_result("研究空白分析", out, st.session_state.get("gap_model", AUTO_MODEL), "research_gap_analyzer")
                st.rerun()
        with tab_citation:
            cid = st.text_input("论文 ID", placeholder="ARXIV:1706.03762")
            other = st.text_input("共现论文 ID", placeholder="可选")
            cols = st.columns(3)
            direction = cols[0].selectbox("方向", ["citations", "references"])
            depth = cols[1].number_input("深度", 1, 2, 1)
            citation_limit = cols[2].number_input("数量", 1, 50, 10, key="citation_limit")
            if st.button("运行引用分析", use_container_width=True):
                if not cid.strip():
                    st.warning("请输入论文 ID。")
                else:
                    kwargs = {"paper_id": cid, "direction": direction, "depth": depth, "limit": citation_limit}
                    if other:
                        kwargs.update({"operation": "co_occurrence", "other_paper_id": other})
                    out = tools.execute({"tool": "citation_traverser", "kwargs": kwargs})
                    append_tool_result("引用分析", out, st.session_state.get("citation_model", NO_LLM), "citation_traverser")
                    st.rerun()
        with tab_code:
            code = st.text_area("Python 代码", value="print('hello literature')", height=160)
            if st.button("执行代码", use_container_width=True):
                out = tools.execute({"tool": "code_executor", "kwargs": {"code": code, "timeout": 8}})
                append_tool_result("代码执行结果", out, NO_LLM, "code_executor")
                st.rerun()


def render_chat_feed() -> None:
    feed = st.container(height=322, border=False)
    with feed:
        if not st.session_state.messages:
            st.info("可以直接提问，也可以从工具参数里启动搜索、对比、研究空白或引用分析。")
        for message in st.session_state.messages:
            render_message(message)

    models = active_model_names()
    with st.form("message_form", clear_on_submit=True, border=False):
        cols = st.columns([0.06, 0.48, 0.28, 0.18])
        with cols[0]:
            st.markdown('<div style="height:38px;display:flex;align-items:center;justify-content:center;font-size:22px;color:#667085;">＋</div>', unsafe_allow_html=True)
        with cols[1]:
            prompt = st.text_input("消息", placeholder="继续提问，或输入 / 调用工具...", label_visibility="collapsed")
        with cols[2]:
            if models:
                current = st.session_state.get("message_model") or st.session_state.get("current_model")
                selected = st.selectbox(
                    "本条模型",
                    models,
                    index=model_index(models, current),
                    key="message_model_select",
                    label_visibility="collapsed",
                )
            else:
                selected = ""
                st.selectbox("本条模型", ["未配置"], disabled=True, label_visibility="collapsed")
        with cols[3]:
            submitted = st.form_submit_button("发送", type="primary", use_container_width=True)
        if submitted and prompt.strip():
            st.session_state.message_model = selected
            with st.spinner("正在思考并调用工具..."):
                run_agent_prompt(prompt.strip(), selected)
            st.rerun()


def render_document_preview(path: str | None, name: str) -> None:
    st.markdown(
        f'<div class="panel-head"><h3>阅读与结构化信息</h3><span>摘要模型：{html.escape(st.session_state.get("current_model") or "未配置")}</span></div>',
        unsafe_allow_html=True,
    )
    if not path or not os.path.exists(path):
        st.info("上传或选择论文后，这里会显示原文预览和解析内容。")
        return

    fmt = _detect_format(path)
    text = parse_document(path)
    toolbar = st.columns([1, 1, 1, 2])
    with toolbar[0]:
        if st.button("重新解析", use_container_width=True):
            for key in list(st.session_state.keys()):
                if str(key).startswith(f"parsed::{path}"):
                    del st.session_state[key]
            st.rerun()
    with toolbar[1]:
        st.button("LLM解析", use_container_width=True, key="llm_structure_top", on_click=run_llm_structure_for_current_doc)
    with toolbar[2]:
        st.button("翻译全文", use_container_width=True, key="translate_full_top", on_click=run_full_translation_for_current_doc)
    with toolbar[3]:
        st.caption(f"{name} · {fmt.upper()}")

    # ── execute pending translation in main flow (not inside callback) ──
    if st.session_state.pop("_translate_pending", False):
        _execute_full_translation(path, text)
        # No st.rerun() — let the script continue to render tabs below.
        # The translation was just stored in session_state, so the tab will see it.

    preview_tab, structure_tab, translation_tab, notes_tab = st.tabs(["原文", "结构化解析", "翻译", "笔记"])
    with preview_tab:
        if fmt == "pdf" and _has_lib("fitz"):
            render_pdf_page(path)
        elif fmt == "docx" and _has_lib("docx"):
            render_docx(path)
        elif fmt in {"md", "txt"}:
            st.text_area("文本预览", text[:10000], height=520, label_visibility="collapsed")
        else:
            st.warning("当前环境缺少该格式的预览依赖，已显示解析文本。")
            st.text_area("解析文本", text[:10000], height=520, label_visibility="collapsed")
    with structure_tab:
        sections = _extract_sections(text) if text and not text.startswith("[") else {}
        if not st.session_state.get(_doc_key(path, "llm_structure"), ""):
            st.button("调用 LLM 生成结构化解析", use_container_width=True, key="llm_structure_tab", on_click=run_llm_structure_for_current_doc)
        if st.session_state.get("last_structure_status"):
            st.caption(st.session_state["last_structure_status"])
        render_document_inspector(sections, text, path)
    with translation_tab:
        translation = st.session_state.get(_doc_key(path, "full_translation"), "")
        if translation:
            with st.container(height=640, border=True):
                st.markdown(translation)
        else:
            st.caption("点击“翻译全文”后显示完整译文。")
    with notes_tab:
        st.text_area("研究笔记", placeholder="记录综述线索、实验设置、可引用结论...", height=420)


def render_document_inspector(sections: dict, text: str, path: str) -> None:
    if text.startswith("["):
        st.warning(text)
        return
    llm_structure = st.session_state.get(_doc_key(path, "llm_structure"), "")
    if llm_structure:
        st.markdown("#### LLM 结构化解析")
        st.markdown(llm_structure)
        st.divider()
    else:
        st.info("点击上方“LLM解析”可调用当前模型生成结构化解析。下方为规则解析结果。")
    left, right = st.columns([1, 0.7])
    with left:
        for key, label in [
            ("title", "标题"),
            ("abstract", "摘要"),
            ("method", "方法"),
            ("results", "实验与结果"),
            ("conclusion", "结论"),
            ("references", "参考文献"),
        ]:
            value = sections.get(key, "").strip()
            if value:
                with st.expander(label, expanded=key in {"title", "abstract"}):
                    st.write(value[:3000])
    with right:
        st.markdown(
            """
            <div class="inspector-card">
              <h4>识别出的综述线索</h4>
              <div class="hint-box teal">共识：数字孪生可改善系统感知与反馈。</div>
              <div class="hint-box">空白：真实部署、跨域数据与成本评估不足。</div>
              <div class="hint-box">矛盾：效率提升与模型个性化之间存在取舍。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_pdf_page(path: str) -> None:
    import fitz

    cache_key = _doc_key(path, "pdf_pages")
    if cache_key not in st.session_state:
        with st.spinner(f"正在加载 PDF 页面..."):
            doc = fitz.open(path)
            try:
                pages = []
                total = len(doc)
                for page_index in range(total):
                    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(1.15, 1.15))
                    pages.append(pix.tobytes("png"))
                st.session_state[cache_key] = pages
            finally:
                doc.close()

    pages = st.session_state[cache_key]
    total_pages = len(pages)

    # Show first batch (3 pages), rest on demand — avoids blocking UI on large PDFs
    show_key = f"{cache_key}_shown"
    if show_key not in st.session_state:
        st.session_state[show_key] = min(3, total_pages)

    shown = st.session_state[show_key]
    with st.container(height=640, border=False):
        for index in range(shown):
            st.caption(f"第 {index + 1} 页 / 共 {total_pages} 页")
            st.image(pages[index], use_container_width=True)

        if shown < total_pages:
            remaining = total_pages - shown
            if st.button(f"加载更多页面（还有 {remaining} 页）", use_container_width=True, key=f"more_pages_{shown}"):
                st.session_state[show_key] = min(shown + 3, total_pages)
                st.rerun()


def render_docx(path: str) -> None:
    import docx

    doc = docx.Document(path)
    shown = 0
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            st.markdown(text)
            shown += 1
        if shown >= 80:
            st.caption("已显示前 80 段。")
            break


def translate_current_summary(text: str) -> None:
    client = get_llm_client(st.session_state.get("current_model"))
    if not client:
        st.session_state.doc_translation = "请先配置大模型。"
        return
    sections = _extract_sections(text)
    source = "\n\n".join(
        part
        for part in [sections.get("title", ""), sections.get("abstract", ""), sections.get("conclusion", "")]
        if part
    )
    if not source:
        source = text[:2500]
    try:
        st.session_state.doc_translation = client.chat(
            [
                {
                    "role": "user",
                    "content": "将下面的学术文本翻译为中文，保留术语、引用和公式，只输出译文。\n\n" + source[:3000],
                }
            ],
            temperature=0.1,
        )
    except Exception as exc:
        st.session_state.doc_translation = f"翻译失败：{exc}"


def llm_structure_document(text: str, name: str) -> str:
    client = get_llm_client(st.session_state.get("current_model"))
    if not client:
        return "请先配置大模型。"
    sections = _extract_sections(text)
    source = "\n\n".join(
        [
            f"文件名：{name}",
            "标题：\n" + sections.get("title", ""),
            "摘要：\n" + sections.get("abstract", ""),
            "方法：\n" + sections.get("method", ""),
            "实验与结果：\n" + sections.get("results", ""),
            "结论：\n" + sections.get("conclusion", ""),
            "原文片段：\n" + text[:4000],
        ]
    )
    prompt = (
        "请对下面论文内容做结构化解析，用中文输出 Markdown。必须包含：\n"
        "1. 标题\n2. 研究问题\n3. 核心方法\n4. 数据集/场景\n5. 评价指标\n"
        "6. 关键结果\n7. 主要贡献\n8. 局限性\n9. 可用于文献综述的写作线索。\n"
        "如果信息缺失，请写“原文未明确”。不要编造。\n\n"
        + source[:8000]
    )
    try:
        result = client.chat([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=1800)
        if result and result.strip():
            return result
        return _fallback_structure(sections)
    except Exception as exc:
        return f"LLM 结构化解析失败：{exc}\n\n{_fallback_structure(sections)}"


def _fallback_structure(sections: dict) -> str:
    return "\n\n".join(
        [
            "> LLM 未返回有效文本，以下为规则解析兜底结果。",
            "### 标题\n" + (sections.get("title", "原文未明确")[:800] or "原文未明确"),
            "### 研究问题\n" + _clean(sections.get("abstract", "原文未明确"))[:700],
            "### 核心方法\n" + _clean(sections.get("method", "原文未明确"))[:700],
            "### 关键结果\n" + _clean(sections.get("results", "原文未明确"))[:700],
            "### 局限性与综述线索\n原文未明确，需要结合更多论文继续分析。",
        ]
    )



def _split_text(text: str, chunk_size: int) -> list[str]:
    chunks = []
    current = []
    current_len = 0
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = [paragraph]
        if len(paragraph) > chunk_size:
            parts = [paragraph[i:i + chunk_size] for i in range(0, len(paragraph), chunk_size)]
        for part in parts:
            if current and current_len + len(part) > chunk_size:
                chunks.append("\n\n".join(current))
                current = [part]
                current_len = len(part)
            else:
                current.append(part)
                current_len += len(part)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _get_or_generate_draft() -> str:
    """Return cached draft, regenerating only when messages change."""
    msgs = st.session_state.messages
    msg_count = len(msgs)
    if st.session_state.get("_draft_msg_count") != msg_count:
        st.session_state["_cached_draft"] = _generate_report()
        st.session_state["_draft_msg_count"] = msg_count
    return st.session_state["_cached_draft"]


def _generate_report() -> str:
    """Generate a structured Markdown literature review draft from the current session."""
    msgs = st.session_state.messages
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [
        "# 文献综述草稿",
        "",
        f"> 自动生成于 {now} · 模型：{st.session_state.get('current_model', '未配置')}",
        "",
        "---",
        "",
    ]

    # ── 1. 研究问题 ──
    user_questions = [m.get("content", "") for m in msgs if m.get("role") == "user"]
    if user_questions:
        lines.append("## 1. 研究问题")
        for i, q in enumerate(user_questions, 1):
            lines.append(f"**Q{i}:** {q.strip()[:300]}")
        lines.append("")

    # ── 2. 文献搜索与筛选结果 ──
    search_results = [m for m in msgs if m.get("tools_called") and "search_papers" in m["tools_called"]]
    if search_results:
        lines.append("## 2. 文献检索结果")
        for i, sr in enumerate(search_results, 1):
            content = sr.get("content", "")
            # Strip the "### xxx" title since we add our own
            content = content.replace("### ", "#### ")
            lines.append(content[:5000])
        lines.append("")

    # ── 3. 文献对比分析 ──
    compare_results = [m for m in msgs if m.get("tools_called") and "literature_comparator" in m["tools_called"]]
    if compare_results:
        lines.append("## 3. 文献对比分析")
        for cr in compare_results:
            content = cr.get("content", "")
            lines.append(content[:5000])
        lines.append("")

    # ── 4. 研究空白分析 ──
    gap_results = [m for m in msgs if m.get("tools_called") and "research_gap_analyzer" in m["tools_called"]]
    if gap_results:
        lines.append("## 4. 研究空白与趋势")
        for gr in gap_results:
            content = gr.get("content", "")
            lines.append(content[:5000])
        lines.append("")

    # ── 5. 已上传论文的结构化解析 ──
    papers = st.session_state.get("uploaded_papers", [])
    if papers:
        lines.append("## 5. 已分析论文清单")
        for p in papers:
            path = p["path"]
            name = p["name"]
            lines.append(f"### {name}")
            # Try to get structured sections
            text = parse_document(path)
            if text and not text.startswith("["):
                sections = _extract_sections(text)
                for sec_key, sec_label in [
                    ("title", "标题"), ("abstract", "摘要"), ("method", "方法"),
                    ("results", "结果"), ("conclusion", "结论")
                ]:
                    sec_text = sections.get(sec_key, "").strip()
                    if sec_text and len(sec_text) > 10:
                        lines.append(f"**{sec_label}:** {sec_text[:800]}")
                        lines.append("")
        lines.append("")

    # ── 6. 翻译内容 ──
    for p in papers:
        path = p["path"]
        translation = st.session_state.get(_doc_key(path, "full_translation"), "")
        if translation and not translation.startswith("请先配置") and "翻译失败" not in translation:
            lines.append(f"## 6. 全文翻译：{p['name']}")
            lines.append(translation[:8000])
            lines.append("")
            break  # only include first translated paper

    # ── 7. 引文网络 ──
    citation_results = [m for m in msgs if m.get("tools_called") and "citation_traverser" in m["tools_called"]]
    if citation_results:
        lines.append("## 7. 引文网络分析")
        for ct in citation_results:
            lines.append(ct.get("content", "")[:5000])
        lines.append("")

    # ── 8. 其他分析 ──
    other_results = [m for m in msgs if m.get("tools_called") and
                     not any(t in m["tools_called"] for t in
                             ["search_papers", "literature_comparator", "research_gap_analyzer",
                              "citation_traverser"])]
    if other_results:
        lines.append("## 8. 补充分析")
        for ot in other_results:
            tools = ot.get("tools_called", [])
            lines.append(f"### 工具：{', '.join(tools)}")
            lines.append(ot.get("content", "")[:3000])
        lines.append("")

    # ── 数据库统计 ──
    try:
        stats = st.session_state.tool_registry.execute(
            {"tool": "sqlite_paper_db", "kwargs": {"operation": "stats"}})
        lines.append("## 附录：论文库统计")
        lines.append(stats)
        lines.append("")
    except Exception:
        pass

    lines.append("---")
    lines.append(f"*文献综述草稿由学术文献分析助手自动生成 · {now}*")
    return "\n".join(lines)


def _clean_for_translation(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []
    for line in lines:
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _doc_key(path: str, suffix: str) -> str:
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        stamp = 0
    return f"doc::{suffix}::{path}::{stamp}"


def _has_lib(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _window(text: str, needles: list[str], size: int = 900) -> str:
    lower = (text or "").lower()
    positions = [lower.find(n) for n in needles if lower.find(n) >= 0]
    if not positions:
        return (text or "")[:size]
    start = max(0, min(positions) - 120)
    return text[start:start + size]


def _extract_mentions(text: str, hints: list[str]) -> str:
    lower = (text or "").lower()
    found = [hint.upper() if hint in {"f1", "auc"} else hint for hint in hints if hint in lower]
    return ", ".join(dict.fromkeys(found))


def _guess_year(text: str) -> int:
    years = [int(y) for y in re.findall(r"\b(20[0-3][0-9]|19[8-9][0-9])\b", text or "")]
    return max(years) if years else 0


ensure_tool_registry()
page_css()
render_sidebar()
render_topbar()
render_metrics()

pending_prompt = st.session_state.pop("pending_prompt", None)
if pending_prompt:
    with st.spinner("正在分析..."):
        if isinstance(pending_prompt, dict):
            run_agent_prompt(pending_prompt["text"], pending_prompt.get("model"))
        else:
            run_agent_prompt(str(pending_prompt), st.session_state.get("task_model"))
    st.rerun()

chat_col, doc_col = st.columns([1.08, 0.92], gap="medium")
with chat_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_tool_cards()
    render_tool_console()
    render_chat_feed()
    st.markdown("</div>", unsafe_allow_html=True)

with doc_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_document_preview(st.session_state.get("current_doc"), st.session_state.get("current_doc_name", ""))
    st.markdown("</div>", unsafe_allow_html=True)
