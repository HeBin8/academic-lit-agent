"""Tool 2: Read PDF, Word (.docx), Markdown, and plain text papers."""

import os
import re
from src.agent.tool_registry import BaseTool


PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "papers")


def _get_fitz():
    try:
        import fitz
        return fitz
    except ImportError:
        return None


def _get_pypdf():
    try:
        import pypdf
        return pypdf
    except ImportError:
        return None


def _get_docx():
    try:
        import docx
        return docx
    except ImportError:
        return None


def _detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        "": "txt",
        ".txt": "txt",
        ".md": "md",
        ".markdown": "md",
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
    }.get(ext, "txt")


def _parse_pdf(path: str) -> str:
    fitz = _get_fitz()
    if fitz:
        doc = fitz.open(path)
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    pypdf = _get_pypdf()
    if pypdf:
        reader = pypdf.PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return "[PDF library not installed. Install pymupdf or pypdf.]"


def _parse_docx(path: str) -> str:
    docx = _get_docx()
    if not docx:
        return "[python-docx not installed. Install: pip install python-docx]"
    doc = docx.Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def _parse_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


PARSERS = {"pdf": _parse_pdf, "docx": _parse_docx, "md": _parse_text, "txt": _parse_text}


SECTION_PATTERNS = {
    "abstract": r"^(abstract|摘要)\b",
    "introduction": r"^(\d+\.?\s*)?(introduction|引言|背景)\b",
    "related_work": r"^(\d+\.?\s*)?(related work|related works|相关工作)\b",
    "method": r"^(\d+\.?\s*)?(method|methods|methodology|approach|framework|model|proposed|方法|模型|框架)\b",
    "results": r"^(\d+\.?\s*)?(experiment|experiments|result|results|evaluation|实验|结果|评估)\b",
    "conclusion": r"^(\d+\.?\s*)?(conclusion|discussion|limitations|总结|结论|讨论|展望)\b",
    "references": r"^(references|bibliography|参考文献)\b",
}


def _extract_sections(text: str) -> dict:
    sections = {
        "title": "",
        "abstract": "",
        "introduction": "",
        "related_work": "",
        "method": "",
        "results": "",
        "conclusion": "",
        "references": "",
    }
    if not text:
        return sections

    lines = [line.strip() for line in text.splitlines()]
    visible = [line for line in lines if line]
    sections["title"] = "\n".join(visible[:8])[:3000]

    current = "title"
    for line in visible:
        lower = line.lower()
        matched = None
        for key, pattern in SECTION_PATTERNS.items():
            if re.search(pattern, lower, re.I):
                matched = key
                break
        if matched:
            current = matched
            continue
        if current in sections:
            prev = sections.get(current, "")
            if current == "title" and len(prev) > 500:
                continue
            sections[current] = ((prev + "\n") if prev else "") + line
            sections[current] = sections[current][:3000]
    return sections


def translate_text(text: str, target_lang: str = "zh") -> str:
    return f"[Translation to {target_lang} requested. This is handled by the chat interface with the active LLM model.]"


class ReadPaperTool(BaseTool):
    name = "read_paper"
    description = (
        "Read and extract structured information from a local PDF, Word (.docx), "
        "Markdown, or text file. Returns title, abstract, method, results, "
        "conclusion, and references by section."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the document file on disk"},
            "output": {
                "type": "string",
                "enum": ["full", "sections"],
                "description": "full=entire text, sections=structured by section",
                "default": "sections",
            },
        },
        "required": ["path"],
    }

    def run(self, path: str = "", output: str = "sections") -> str:
        if not os.path.exists(path):
            alt = os.path.join(PAPER_DIR, os.path.basename(path))
            if os.path.exists(alt):
                path = alt
            else:
                return f"File not found: {path}"

        fmt = _detect_format(path)
        parser = PARSERS.get(fmt)
        if not parser:
            return f"Unsupported format: {fmt}"

        raw = parser(path)
        if raw.startswith("["):
            return raw

        lines = [f"File: {os.path.basename(path)}", f"Format: {fmt.upper()}", f"Size: {len(raw)} chars", ""]
        if output == "full":
            lines.append(raw[:8000])
            return "\n".join(lines)

        sections = _extract_sections(raw)
        labels = {
            "title": "Title",
            "abstract": "Abstract",
            "introduction": "Introduction",
            "related_work": "Related Work",
            "method": "Method / Approach",
            "results": "Experiments / Results",
            "conclusion": "Conclusion / Discussion",
            "references": "References",
        }
        for key, label in labels.items():
            value = sections.get(key, "").strip()
            if value:
                lines.append(f"[{label}]")
                lines.append(value[:1800])
                lines.append("")

        if len(lines) <= 4:
            lines.append("[No clear sections detected - showing raw content]")
            lines.append(raw[:3000])
        return "\n".join(lines)
