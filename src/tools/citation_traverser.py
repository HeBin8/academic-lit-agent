"""Tool 6: Citation graph traversal - BFS/DFS through citation chains."""

import json, urllib.request, urllib.parse, time
from collections import Counter
from src.agent.tool_registry import BaseTool


BASE = "https://api.semanticscholar.org/graph/v1/paper"


def _fetch_edges(paper_id: str, edge_type: str, limit: int = 20) -> list[dict]:
    """Fetch citations or references for a paper."""
    fields = ["paperId", "title", "year", "authors", "citationCount", "url"]
    params = {"limit": min(limit, 50), "offset": 0, "fields": ",".join(fields)}
    url = f"{BASE}/{urllib.parse.quote(paper_id, safe='')}/{edge_type}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AcademicLitAgent/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                edges = data.get("data", [])
                result = []
                for e in edges:
                    target = e.get("citingPaper") or e.get("citedPaper") or {}
                    if target:
                        result.append(target)
                return result
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            return [{"error": f"API error {e.code}"}]
        except Exception:
            time.sleep(1)
    return []


class CitationTraverserTool(BaseTool):
    name = "citation_traverser"
    description = "Traverse the citation graph from a starting paper. Can fetch citations (papers that cite it) or references (papers it cites). Supports BFS and DFS traversal. Also supports co-occurrence analysis to find papers that cite two given papers."
    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string", "description": "Semantic Scholar paper ID or external ID (e.g. ARXIV:1706.03762)"},
            "direction": {"type": "string", "enum": ["citations", "references"], "description": "Direction: citations=papers that cite this, references=papers this cites", "default": "citations"},
            "limit": {"type": "integer", "description": "Max results (1-50)", "default": 10},
            "depth": {"type": "integer", "description": "Traversal depth. 1 = direct neighbors only, 2 = one hop further", "default": 1},
            "mode": {"type": "string", "enum": ["bfs", "dfs"], "description": "Search strategy: bfs=breadth-first, dfs=depth-first", "default": "bfs"},
            "operation": {"type": "string", "enum": ["traverse", "co_occurrence"], "description": "traverse=follow citation chain, co_occurrence=find papers that cite both paper_id and another_paper_id", "default": "traverse"},
            "other_paper_id": {"type": "string", "description": "Second paper ID for co_occurrence analysis", "default": ""},
        },
        "required": ["paper_id"],
    }

    def run(self, paper_id: str = "", direction: str = "citations",
            limit: int = 10, depth: int = 1, mode: str = "bfs",
            operation: str = "traverse", other_paper_id: str = "") -> str:
        depth = int(depth)
        limit = min(int(limit), 50)

        if operation == "co_occurrence":
            if not other_paper_id:
                return "Error: other_paper_id is required for co_occurrence"
            first = _fetch_edges(paper_id, "citations", limit)
            second = _fetch_edges(other_paper_id, "citations", limit)
            if first and "error" in first[0]:
                return first[0]["error"]
            if second and "error" in second[0]:
                return second[0]["error"]
            first_by_id = {p.get("paperId"): p for p in first if p.get("paperId")}
            second_ids = {p.get("paperId") for p in second if p.get("paperId")}
            shared = [first_by_id[pid] for pid in first_by_id.keys() & second_ids]
            lines = [
                f"# Citation Co-occurrence",
                f"Papers citing both {paper_id} and {other_paper_id}: {len(shared)}",
                "",
            ]
            if not shared:
                lines.append("No shared citing papers found in the fetched window. Increase limit for a broader check.")
                return "\n".join(lines)
            for i, paper in enumerate(sorted(shared, key=lambda p: p.get("citationCount", 0), reverse=True), 1):
                authors = ", ".join(a["name"] for a in (paper.get("authors") or [])[:3])
                if len(paper.get("authors") or []) > 3:
                    authors += " et al."
                lines.append(
                    f"{i}. [{paper.get('paperId','')}] {paper.get('title','')} "
                    f"({paper.get('year','')}) citations: {paper.get('citationCount',0)}"
                )
                if authors:
                    lines.append(f"   Authors: {authors}")
            return "\n".join(lines)

        visited = set()

        result = [f"# Citation Traversal: {direction} from {paper_id}", ""]
        queue = [(paper_id, 0)]

        while queue and len(visited) < limit * depth:
            current_id, current_depth = queue.pop(0) if mode != "dfs" else queue.pop()
            if current_id in visited or current_depth >= depth:
                continue
            visited.add(current_id)

            edges = _fetch_edges(current_id, direction, limit)
            result.append(f"### Depth {current_depth + 1}")
            for i, edge in enumerate(edges, 1):
                if "error" in edge:
                    result.append(f"  {edge['error']}")
                    continue
                authors = ", ".join(a["name"] for a in (edge.get("authors") or [])[:3])
                if len(edge.get("authors") or []) > 3:
                    authors += " et al."
                result.append(
                    f"  {i}. [{edge.get('paperId','')}] {edge.get('title','')} "
                    f"({edge.get('year','')}) citations: {edge.get('citationCount',0)}"
                )
                result.append(f"     Authors: {authors}")
                if current_depth + 1 < depth:
                    queue.append((edge.get("paperId", ""), current_depth + 1))
            result.append("")

        result.append(f"Traversed {len(visited)} unique papers.")
        if len(visited) == 1 and depth > 1:
            result.append("(Limited connectivity - API may have returned fewer results than requested)")
        return "\n".join(result)

