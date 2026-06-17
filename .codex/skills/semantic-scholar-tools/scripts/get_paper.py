#!/usr/bin/env python3
"""Get detailed paper metadata by Semantic Scholar paper ID (or external ID like ARXIV:...)."""
import argparse, json, sys, urllib.request, urllib.parse, time

BASE = 'https://api.semanticscholar.org/graph/v1/paper'

def get_paper(paper_id, fields=None):
    if fields is None:
        fields = ['paperId','externalIds','title','abstract','year','authors',
                  'citationCount','referenceCount','url','venue','tldr',
                  'openAccessPdf','fieldsOfStudy']
    params = {'fields': ','.join(fields)}
    url = f'{BASE}/{urllib.parse.quote(paper_id, safe="")}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'AcademicLitAgent/1.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            if e.code == 404:
                return None
            raise
        except Exception:
            time.sleep(1)
    return None

def main():
    p = argparse.ArgumentParser(description='Get paper details from Semantic Scholar')
    p.add_argument('paper_id', help='Semantic Scholar ID or external ID (e.g. ARXIV:2101.00001)')
    p.add_argument('--fields', default='paperId,externalIds,title,abstract,year,authors,citationCount,referenceCount,url,venue,tldr,openAccessPdf,fieldsOfStudy')
    args = p.parse_args()
    fields = [f.strip() for f in args.fields.split(',')]
    result = get_paper(args.paper_id, fields)
    if result is None:
        print(json.dumps({'error': 'paper not found'}, ensure_ascii=False))
        sys.exit(1)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
