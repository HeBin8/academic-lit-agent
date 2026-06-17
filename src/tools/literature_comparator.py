"""Tool 4: Compare multiple papers and generate structured comparison tables."""

import json, os, re
from src.agent.tool_registry import BaseTool
from src.tools.read_paper import _detect_format, PARSERS, _extract_sections


# In-memory paper storage for comparator (data shared via agent context)
_comparison_cache: dict[str, dict] = {}


class LiteratureComparatorTool(BaseTool):
    name = "literature_comparator"
    description = "Compare multiple papers side-by-side on method, dataset, evaluation metrics, and key results. Provide paper data as JSON, or local file paths as JSON, and the tool will build a structured comparison table."
    parameters = {
        "type": "object",
        "properties": {
            "papers_json": {
                "type": "string",
                "description": "JSON array of paper objects. Each: {\"paper_id\":\"...\", \"title\":\"...\", \"method\":\"...\", \"dataset\":\"...\", \"metrics\":\"...\", \"results\":\"...\", \"key_finding\":\"...\"}",
            },
            "paper_paths_json": {
                "type": "string",
                "description": "Optional JSON array of local paper paths. Used when papers_json is not provided.",
                "default": "",
            },
            "topic": {
                "type": "string",
                "description": "Optional review topic to display in the comparison heading.",
                "default": "",
            },
            "dimension": {
                "type": "string",
                "enum": ["method", "dataset", "metrics", "all"],
                "description": "Which dimension to compare",
                "default": "all",
            },
        },
        "required": ["papers_json"],
    }

    def run(self, papers_json: str = "", paper_paths_json: str = "", topic: str = "", dimension: str = "all") -> str:
        global _comparison_cache
        papers = []
        if papers_json:
            try:
                papers = json.loads(papers_json) if isinstance(papers_json, str) else papers_json
            except json.JSONDecodeError:
                return "Error: papers_json must be valid JSON array"
        elif paper_paths_json:
            try:
                paths = json.loads(paper_paths_json) if isinstance(paper_paths_json, str) else paper_paths_json
            except json.JSONDecodeError:
                return "Error: paper_paths_json must be valid JSON array"
            papers = [_paper_from_path(path) for path in paths]

        if not papers or not isinstance(papers, list):
            return "Error: provide papers_json or paper_paths_json as a non-empty JSON array"

        # Cache paper data
        for p in papers:
            pid = p.get("paper_id", p.get("title", f"paper_{len(_comparison_cache)}"))
            _comparison_cache[pid] = p

        dims = ["method", "dataset", "metrics", "results", "key_finding"]
        if dimension != "all":
            dims = [dimension]

        # Build Markdown table header
        header_cols = ["Paper"] + [d.capitalize() for d in dims]
        header = "| " + " | ".join(header_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(header_cols)) + " |"

        rows = []
        for i, p in enumerate(papers):
            name = p.get("title", p.get("paper_id", f"Paper {i+1}"))[:40]
            vals = [name]
            for d in dims:
                val = (p.get(d, "") or "")[:30].replace("|", "&#124;")
                vals.append(val)
            rows.append("| " + " | ".join(vals) + " |")

        result = [
            "## Literature Comparison Table",
            f"Topic: {topic}" if topic else "",
            f"Comparing {len(papers)} papers across: {', '.join(dims)}",
            "",
            header,
            sep,
        ]
        result.extend(rows)
        result.extend([
            "",
            "### Common Patterns",
        ])

        # Simple pattern detection
        methods = [p.get("method", "").lower() for p in papers if p.get("method")]
        if methods:
            from collections import Counter
            words = []
            for m in methods:
                words.extend(m.split()[:10])
            common = Counter(words).most_common(5)
            if common:
                result.append(f"Frequent method keywords: {', '.join(f'{w}({c})' for w,c in common)}")

        datasets = set(p.get("dataset", "") for p in papers if p.get("dataset"))
        if datasets:
            result.append(f"Datasets used: {', '.join(datasets)}")

        return "\n".join(result)


def _paper_from_path(path: str) -> dict:
    if not os.path.exists(path):
        return {"paper_id": path, "title": os.path.basename(path), "method": "File not found", "dataset": "", "metrics": "", "results": ""}
    fmt = _detect_format(path)
    parser = PARSERS.get(fmt)
    if not parser:
        return {"paper_id": path, "title": os.path.basename(path), "method": f"Unsupported format: {fmt}", "dataset": "", "metrics": "", "results": ""}
    raw = parser(path)
    sections = _extract_sections(raw) if raw and not raw.startswith("[") else {}
    title = sections.get("title") or os.path.basename(path)
    method_text = sections.get("method") or _window(raw, ["method", "approach", "framework", "model"])
    result_text = sections.get("results") or _window(raw, ["result", "experiment", "evaluation"])
    return {
        "paper_id": os.path.basename(path),
        "title": _clean(title)[:120],
        "method": _clean(method_text)[:220],
        "dataset": ", ".join(_extract_terms(raw, ["dataset", "benchmark", "corpus"]))[:220],
        "metrics": ", ".join(_extract_terms(raw, ["accuracy", "f1", "precision", "recall", "auc", "mae", "rmse", "latency"]))[:220],
        "results": _clean(result_text)[:220],
        "key_finding": _clean(result_text or method_text)[:220],
    }


def _window(text: str, needles: list[str], size: int = 900) -> str:
    lower = text.lower()
    positions = [lower.find(n) for n in needles if lower.find(n) >= 0]
    if not positions:
        return text[:size]
    start = max(0, min(positions) - 120)
    return text[start:start + size]


def _extract_terms(text: str, hints: list[str]) -> list[str]:
    found = []
    for hint in hints:
        if re.search(rf"\b{re.escape(hint)}\b", text, re.I):
            found.append(hint.upper() if hint in {"f1", "auc", "mae", "rmse"} else hint)
    for pattern in [r"\b[A-Z][A-Za-z0-9_-]{2,}(?:-[A-Za-z0-9]+)?\b", r"\b[A-Z]{2,}\b"]:
        for item in re.findall(pattern, text[:8000]):
            if item not in found and len(found) < 12:
                found.append(item)
    return found[:12]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

