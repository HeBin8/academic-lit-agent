---
name: semantic-scholar-tools
description: Search and retrieve academic papers, citations, references, and BibTeX from the Semantic Scholar Academic Graph API. Use when the task involves literature search, paper metadata retrieval, citation analysis, building a bibliography, or any academic paper data pipeline. Scripts handle rate limiting (429) with exponential backoff retry.
---

# Semantic Scholar API Tools

Four CLI scripts in `scripts/` covering the core Semantic Scholar API surface. All output JSON to stdout; all accept `--fields` to select return fields and use `urllib.request` (stdlib only, no pip dependencies).

See [references/api_reference.md](references/api_reference.md) for endpoint details, available fields, and rate limits.

## Quick Start

```bash
python scripts/search_papers.py "transformer attention mechanism" --limit 5
python scripts/get_paper.py ARXIV:1706.03762
python scripts/get_citations.py ARXIV:1706.03762 --type citations --limit 10
python scripts/export_bibtex.py ARXIV:1706.03762 649def34f8be52 > refs.bib
```

## Scripts

### search_papers.py

Search by keyword. Supports pagination (`--offset`), Open Access filter (`--open-access`), and JSONL output (`--jsonl`).

```bash
python scripts/search_papers.py "low-resource neural machine translation" --limit 20 --fields paperId,title,year,citationCount,abstract
```

### get_paper.py

Fetch full metadata for one paper. Accepts S2 ID, DOI, ARXIV, or CorpusId.

```bash
python scripts/get_paper.py DOI:10.1038/nature14539 --fields title,abstract,tldr,authors,citationCount
```

### get_citations.py

List papers that cite (`--type citations`) or are cited by (`--type references`) a given paper.

```bash
python scripts/get_citations.py ARXIV:2005.14165 --type citations --limit 50 --offset 100
```

### export_bibtex.py

Batch-fetch papers and emit BibTeX entries. Takes paper IDs as arguments or reads from stdin (`--from-stdin`). Useful for building `.bib` files.

```bash
# From a file of IDs
cat paper_ids.txt | python scripts/export_bibtex.py --from-stdin > bibliography.bib

# Direct IDs
python scripts/export_bibtex.py DOI:10.xxx ARXIV:2101.xxx > refs.bib
```

## Usage in Agent Tools

When building a LangChain/LangGraph tool, wrap these scripts with `subprocess.run`:

```python
import subprocess, json

def search_papers(query: str, limit: int = 10) -> list[dict]:
    result = subprocess.run(
        ["python", "scripts/search_papers.py", query, "--limit", str(limit), "--fields", "paperId,title,year,authors,abstract,citationCount,url"],
        capture_output=True, text=True, cwd="/path/to/skill"
    )
    return json.loads(result.stdout)
```

For rate-limited environments, the scripts retry 429s automatically with 1s/2s/4s backoff.
