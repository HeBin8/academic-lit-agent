#!/usr/bin/env python3
"""Search Semantic Scholar for papers by keyword query."""
import argparse, json, sys, urllib.request, urllib.parse, time

BASE = 'https://api.semanticscholar.org/graph/v1/paper/search'

def search(query, limit=10, offset=0, fields=None, open_access=False):
    if fields is None:
        fields = ['paperId','title','year','authors','abstract','citationCount','url']
    params = {'query': query, 'limit': limit, 'offset': offset,
              'fields': ','.join(fields)}
    url = f'{BASE}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'AcademicLitAgent/1.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                papers = data.get('data', [])
                if open_access:
                    papers = [p for p in papers if p.get('isOpenAccess')]
                return papers
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            time.sleep(1)
    return []

def main():
    p = argparse.ArgumentParser(description='Search Semantic Scholar papers')
    p.add_argument('query', help='Search query string')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--fields', default='paperId,title,year,authors,abstract,citationCount,url')
    p.add_argument('--open-access', action='store_true')
    p.add_argument('--jsonl', action='store_true', help='Output one JSON per line')
    args = p.parse_args()
    fields = [f.strip() for f in args.fields.split(',')]
    results = search(args.query, args.limit, args.offset, fields, args.open_access)
    if args.jsonl:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
    else:
        json.dump(results, sys.stdout, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
