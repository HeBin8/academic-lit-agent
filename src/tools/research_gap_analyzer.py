"""Tool 5: Analyze research gaps from a collection of papers."""

import json
import re
from collections import Counter
from src.agent.tool_registry import BaseTool


class ResearchGapAnalyzerTool(BaseTool):
    name = "research_gap_analyzer"
    description = (
        "Analyze research gaps from a set of papers. Provide paper data as JSON "
        "with fields: paper_id, title, method, dataset, metrics, results, "
        "key_finding, and year. Returns consensus points, contradictions, "
        "research gaps, and emerging trends."
    )
    parameters = {
        "type": "object",
        "properties": {
            "papers_json": {
                "type": "string",
                "description": "JSON array: each with paper_id, title, method, key_finding, results, dataset, year",
            },
        },
        "required": ["papers_json"],
    }

    def run(self, papers_json: str = "") -> str:
        try:
            papers = json.loads(papers_json) if isinstance(papers_json, str) else papers_json
        except json.JSONDecodeError:
            return "Error: papers_json must be valid JSON"
        if not papers or not isinstance(papers, list):
            return "Error: must provide a non-empty array of papers"

        result = ["# Research Gap Analysis", ""]
        findings = [str(p.get("key_finding", "")) for p in papers if p.get("key_finding")]
        methods = [str(p.get("method", "")).strip().lower() for p in papers if p.get("method")]
        datasets = [str(p.get("dataset", "")).strip() for p in papers if p.get("dataset")]
        years = sorted({_safe_int(p.get("year")) for p in papers if _safe_int(p.get("year"))})

        keyword_counts = _keyword_counts(findings)
        min_count = max(2, len(papers) // 3)
        common_kw = Counter({k: v for k, v in keyword_counts.items() if v >= min_count})

        result.append("## Consensus")
        if common_kw:
            result.append("Frequent concepts across papers:")
            for kw, count in common_kw.most_common(10):
                result.append(f"- {kw}: appears in {count}/{len(papers)} papers")
        else:
            result.append("- No strong repeated finding terms were detected.")
        if datasets:
            common_datasets = Counter(datasets).most_common(5)
            result.append("- Common datasets: " + ", ".join(f"{d} ({c})" for d, c in common_datasets))
        result.append("")

        result.append("## Contradictions")
        contradictions = _find_contradictions(papers)
        if contradictions:
            for method, p1, r1, p2, r2 in contradictions[:8]:
                result.append(f"- Same method '{method}' but different results:")
                result.append(f"  [{p1.get('paper_id','?')}] {r1}")
                result.append(f"  [{p2.get('paper_id','?')}] {r2}")
        else:
            result.append("- No direct contradiction was detected from the supplied structured fields.")
        result.append("")

        result.append("## Identified Research Gaps")
        for gap in _identify_gaps(papers, methods, common_kw, years):
            result.append(f"- {gap}")
        result.append("")

        result.append("## Trends")
        if years:
            by_year = Counter(_safe_int(p.get("year")) for p in papers if _safe_int(p.get("year")))
            result.append("- Publication distribution: " + ", ".join(f"{y}: {by_year[y]}" for y in sorted(by_year)))
            recent_cutoff = max(years) - 1
            recent_methods = Counter(
                (p.get("method") or "unknown").lower()
                for p in papers
                if _safe_int(p.get("year")) >= recent_cutoff
            )
            if recent_methods:
                result.append("- Recent method signals: " + ", ".join(f"{m} ({c})" for m, c in recent_methods.most_common(5)))
        else:
            result.append("- No year metadata was provided, so temporal trends cannot be estimated.")

        return "\n".join(result)


def _keyword_counts(texts: list[str]) -> Counter:
    stop_words = {
        "that", "this", "with", "from", "their", "which", "based", "using",
        "paper", "study", "result", "results", "method", "model", "approach",
        "performance", "proposed", "shows", "showed", "large",
    }
    counts = Counter()
    for text in texts:
        seen = set()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower()):
            if len(word) > 4 and word not in stop_words:
                seen.add(word)
        counts.update(seen)
    return counts


def _find_contradictions(papers: list[dict]) -> list[tuple]:
    contradictions = []
    same_method = [(p, str(p.get("method", "")).strip()) for p in papers if p.get("method")]
    for i, (p1, m1) in enumerate(same_method):
        for j, (p2, m2) in enumerate(same_method):
            if i >= j or m1.lower() != m2.lower():
                continue
            r1 = str(p1.get("results", "") or "")[:120]
            r2 = str(p2.get("results", "") or "")[:120]
            if r1 and r2 and r1 != r2:
                contradictions.append((m1, p1, r1, p2, r2))
    return contradictions


def _identify_gaps(papers: list[dict], methods: list[str], common_kw: Counter, years: list[int]) -> list[str]:
    unique_methods = set(methods)
    all_text = " ".join(
        str(p.get("key_finding", "")) + " " +
        str(p.get("method", "")) + " " +
        str(p.get("dataset", "")) + " " +
        str(p.get("results", ""))
        for p in papers
    ).lower()

    gaps = []
    if len(unique_methods) <= 2 and len(papers) >= 3:
        gaps.append("Limited methodological diversity: most papers use similar approaches.")
    if common_kw:
        top = [k for k, _ in common_kw.most_common(3)]
        gaps.append(f"Research is clustered around {', '.join(top)}; adjacent directions may be underexplored.")
    if not any(w in all_text for w in ["real-world", "real world", "deployment", "production", "practical"]):
        gaps.append("Real-world deployment validation is weak or missing.")
    if sum(1 for w in ["english", "chinese", "multilingual", "cross-lingual", "low-resource", "language"] if w in all_text) <= 1:
        gaps.append("Language diversity is limited; multilingual and low-resource settings need more coverage.")
    if sum(1 for w in ["biomedical", "scientific", "legal", "medical", "finance", "education"] if w in all_text) <= 1:
        gaps.append("Domain-specific adaptation remains underexplored.")
    if any(w in all_text for w in ["efficient", "lightweight", "small"]) and not any(
        w in all_text for w in ["latency", "throughput", "inference time", "speed"]
    ):
        gaps.append("Efficiency claims need runtime, latency, or throughput measurements.")
    if years and len(years) >= 2 and max(years) - min(years) >= 3:
        gaps.append("The timeline spans multiple years; compare older and recent assumptions explicitly.")
    if not gaps:
        gaps.append("The supplied metadata is too thin for confident gap detection; extract richer method/result fields.")
    return gaps


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
