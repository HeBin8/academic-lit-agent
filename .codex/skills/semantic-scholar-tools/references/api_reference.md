# Semantic Scholar API Reference

Base URL: `https://api.semanticscholar.org/graph/v1`

## Endpoints

### Paper Search
```
GET /paper/search?query=<str>&limit=<int>&offset=<int>&fields=<csv>
```
Returns `{"total": N, "offset": M, "next": N, "data": [...]}`

### Paper Details
```
GET /paper/{paper_id}?fields=<csv>
```
`paper_id` can be: S2 ID (hash), `DOI:10.xxx`, `ARXIV:2101.00001`, `CorpusId:123`, etc.

### Citations
```
GET /paper/{paper_id}/citations?limit=<int>&offset=<int>&fields=<csv>
```
Returns `{"data": [{"citingPaper": {...}, "contexts": [...]}, ...]}`

### References
```
GET /paper/{paper_id}/references?limit=<int>&offset=<int>&fields=<csv>
```
Returns `{"data": [{"citedPaper": {...}, "contexts": [...]}, ...]}`

### Batch
```
POST /paper/batch?fields=<csv>
Body: {"ids": ["id1", "id2", ...]}
```
Returns array of paper objects. Max 500 IDs per request.

### Bulk Search
```
GET /paper/search/bulk?query=<str>&fields=<csv>&year=<range>&fieldsOfStudy=<str>
```
Pagination via `token` parameter in response.

## Available Fields
- `paperId`, `externalIds` (DOI, ArXiv, MAG, CorpusId, PubMed), `url`
- `title`, `abstract`, `tldr` (auto-generated summary)
- `year`, `publicationDate`, `publicationTypes`
- `authors` (name, authorId, externalIds, url)
- `venue`, `journal` (name, volume, pages)
- `citationCount`, `influentialCitationCount`, `referenceCount`
- `isOpenAccess`, `openAccessPdf` (url, status)
- `fieldsOfStudy`, `s2FieldsOfStudy`
- `embedding` (vector, requires special tier)

## Rate Limits
- Without API key: 100 requests per 5 minutes
- With API key (header `x-api-key`): 100 requests per second
- Retry on 429 with exponential backoff (1s, 2s, 4s)

## Common Queries
- Exact phrase: `"neural machine translation"`
- Boolean: `+transformers +attention -vision`
- Year range: `&year=2020-2024`
- Field filter: `&fieldsOfStudy=Computer Science`
- Open access only: `&isOpenAccess=true` (bulk endpoint only)
