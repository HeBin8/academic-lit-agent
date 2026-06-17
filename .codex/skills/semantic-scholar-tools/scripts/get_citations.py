#!/usr/bin/env python3
"""Get citations and/or references for a given paper."""
import argparse, json, sys, urllib.request, urllib.parse, time

BASE = 'https://api.semanticscholar.org/graph/v1/paper'

def get_edges(paper_id, edge_type, limit=20, offset=0, fields=None):
    if fields is None:
        fields = ['paperId','title','year','authors','citationCount','url','abstract']
    params = {'limit': limit, 'offset': offset, 'fields': ','.join(fields)}
    url = f'{BASE}/{urllib.parse.quote(paper_id, safe="")}/{edge_type}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'AcademicLitAgent/1.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                edges = data.get('data', [])
                result = []
                for e in edges:
                    citing_or_cited = e.get('citingPaper') or e.get('citedPaper') or {}
                    result.append(citing_or_cited)
                return result
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            time.sleep(1)
    return []

def main():
    p = argparse.ArgumentParser(description='Get citations or references from Semantic Scholar')
    p.add_argument('paper_id')
    p.add_argument('--type', choices=['citations','references'], default='citations')
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--fields', default='paperId,title,year,authors,citationCount,url,abstract')
    args = p.parse_args()
    fields = [f.strip() for f in args.fields.split(',')]
    results = get_edges(args.paper_id, args.type, args.limit, args.offset, fields)
    json.dump(results, sys.stdout, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
