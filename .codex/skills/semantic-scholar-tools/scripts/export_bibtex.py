#!/usr/bin/env python3
"""Export paper metadata to BibTeX format via Semantic Scholar batch API."""
import argparse, json, sys, urllib.request, time

BASE = 'https://api.semanticscholar.org/graph/v1/paper'

def fetch_papers(paper_ids):
    """Batch-fetch paper details from Semantic Scholar."""
    payload = json.dumps({'ids': paper_ids}).encode()
    req = urllib.request.Request(
        f'{BASE}/batch?fields=paperId,externalIds,title,year,authors,venue,journal,url',
        data=payload,
        headers={'Content-Type': 'application/json', 'User-Agent': 'AcademicLitAgent/1.0'},
        method='POST'
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            time.sleep(1)
    return []

def paper_to_bibtex(p):
    """Convert a paper dict to a BibTeX entry."""
    ext = p.get('externalIds', {}) or {}
    citekey = (ext.get('DOI') or p.get('paperId', 'unknown'))[:40].replace('/', '_')
    etype = 'article'
    if any(k for k in ext if 'arxiv' in k.lower()):
        etype = 'misc'
    authors = ' and '.join(a['name'] for a in (p.get('authors') or [])[:10])
    title = (p.get('title') or 'Untitled').replace('{', r'\{').replace('}', r'\}')
    year = p.get('year', '')
    venue = ''
    if isinstance(p.get('venue'), dict):
        venue = p['venue'].get('name', '')
    elif isinstance(p.get('journal'), dict):
        venue = p['journal'].get('name', '')
    doi = ext.get('DOI', '')
    url = p.get('url', '')
    lines = [f'@{etype}{{{citekey},']
    lines.append(f'  title = {{{title}}},')
    if authors:
        lines.append(f'  author = {{{authors}}},')
    if year:
        lines.append(f'  year = {{{year}}},')
    if venue:
        lines.append(f'  journal = {{{venue}}},')
    if doi:
        lines.append(f'  doi = {{{doi}}},')
    if url:
        lines.append(f'  url = {{{url}}},')
    lines.append('}')
    return '\n'.join(lines)

def main():
    p = argparse.ArgumentParser(description='Export papers to BibTeX')
    p.add_argument('paper_ids', nargs='*', help='Paper IDs to export')
    p.add_argument('--from-stdin', action='store_true', help='Read paper IDs from stdin')
    args = p.parse_args()
    ids = args.paper_ids
    if args.from_stdin:
        raw = sys.stdin.read().strip()
        try:
            ids = json.loads(raw) if raw.startswith('[') else [l.strip() for l in raw.splitlines() if l.strip()]
        except json.JSONDecodeError:
            ids = [l.strip() for l in raw.splitlines() if l.strip()]
    if not ids:
        print('% No paper IDs provided', file=sys.stderr)
        sys.exit(1)
    papers = fetch_papers(ids)
    for paper in papers:
        print(paper_to_bibtex(paper))
        print()

if __name__ == '__main__':
    main()
