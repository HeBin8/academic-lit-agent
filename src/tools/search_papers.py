"""Tool 1: Search academic papers via Semantic Scholar."""

import json, urllib.request, urllib.parse, time
from src.agent.tool_registry import BaseTool, ToolSpec


BASE = "https://api.semanticscholar.org/graph/v1/paper/search"


def _api_search(query: str, limit: int = 10, offset: int = 0,
                year: str = "", fields_of_study: str = "") -> list[dict]:
    fields = ["paperId", "title", "year", "authors", "abstract",
              "citationCount", "url", "venue", "publicationDate"]
    if fields_of_study:
        fields.append("fieldsOfStudy")
    params = {"query": query, "limit": min(limit, 50), "offset": offset,
              "fields": ",".join(fields)}
    if year:
        params["year"] = year
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AcademicLitAgent/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("data", [])
                if fields_of_study:
                    results = [r for r in results if fields_of_study.lower()
                               in str(r.get("fieldsOfStudy", [])).lower()]
                return results
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            return [{"error": f"API error {e.code}: {e.read().decode()[:200]}"}]
        except Exception:
            time.sleep(1)
    return [{"error": "API request failed after 3 retries"}]


class SearchPapersTool(BaseTool):
    name = "search_papers"
    description = "Search academic papers by keyword query. Returns a list of paper IDs, titles, years, authors, citation counts, and abstracts."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (e.g. 'retrieval augmented generation low resource')"},
            "limit": {"type": "integer", "description": "Max results (1-50)", "default": 10},
            "offset": {"type": "integer", "description": "Pagination offset", "default": 0},
            "year": {"type": "string", "description": "Year filter (e.g. '2020-2024')", "default": ""},
            "fields_of_study": {"type": "string", "description": "Filter by field (e.g. Computer Science, Medicine)", "default": ""},
        },
        "required": ["query"],
    }

    def run(self, query: str = "", limit: int = 10, offset: int = 0, year: str = "",
            fields_of_study: str = "") -> str:
        results = _api_search(query, int(limit), int(offset), year=year, fields_of_study=fields_of_study)
        if not results:
            return "No papers found."
        if isinstance(results[0], dict) and results[0].get("error"):
            return results[0]["error"]
        lines = [f"Found {len(results)} papers (offset={offset}):"]
        for i, r in enumerate(results, 1):
            authors = ", ".join(a["name"] for a in (r.get("authors") or [])[:3])
            if len(r.get("authors") or []) > 3:
                authors += " et al."
            lines.append(
                f"{i}. [{r.get('paperId','')}] {r.get('title','')} "
                f"({r.get('year','')}) - citations: {r.get('citationCount',0)}"
            )
            lines.append(f"   Authors: {authors}")
            abstract = (r.get("abstract") or "")[:200]
            if abstract:
                lines.append(f"   Abstract: {abstract}...")
            lines.append("")
        return "\n".join(lines)

