"""Tool 3: SQLite paper database - store, deduplicate, query, and tag papers."""

import json
import hashlib
import os
import pathlib
import sqlite3
from datetime import datetime
from src.agent.tool_registry import BaseTool


DB_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "papers.db"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            year INTEGER,
            authors TEXT,
            abstract TEXT,
            venue TEXT,
            citation_count INTEGER,
            url TEXT,
            tags TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            favorite INTEGER DEFAULT 0,
            file_path TEXT DEFAULT '',
            source_hash TEXT DEFAULT '',
            page_count INTEGER DEFAULT 0,
            status TEXT DEFAULT '',
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            paper_id TEXT,
            tag TEXT,
            PRIMARY KEY (paper_id, tag)
        )
    """)
    _ensure_column(conn, "papers", "favorite", "INTEGER DEFAULT 0")
    _ensure_column(conn, "papers", "notes", "TEXT DEFAULT ''")
    _ensure_column(conn, "papers", "file_path", "TEXT DEFAULT ''")
    _ensure_column(conn, "papers", "source_hash", "TEXT DEFAULT ''")
    _ensure_column(conn, "papers", "page_count", "INTEGER DEFAULT 0")
    _ensure_column(conn, "papers", "status", "TEXT DEFAULT ''")
    return conn


class SQLitePaperDBTool(BaseTool):
    name = "sqlite_paper_db"
    description = "Store, retrieve, deduplicate, tag, favorite, and manage paper metadata in a local SQLite database shared across sessions."
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "save", "search", "list", "list_json", "get", "delete",
                    "tag", "untag", "set_tags", "favorite", "unfavorite",
                    "favorites", "search_by_tag", "dedupe", "stats"
                ],
                "description": "Database operation to perform",
            },
            "paper_id": {"type": "string", "description": "Semantic Scholar paper ID or local paper identifier", "default": ""},
            "title": {"type": "string", "description": "Paper title", "default": ""},
            "authors": {"type": "string", "description": "Author names", "default": ""},
            "year": {"type": "string", "description": "Publication year", "default": ""},
            "abstract": {"type": "string", "description": "Paper abstract", "default": ""},
            "tag": {"type": "string", "description": "Tag name for tag/search_by_tag operations", "default": ""},
            "tags": {"type": "string", "description": "Comma-separated tags for set_tags/save operations", "default": ""},
            "query": {"type": "string", "description": "Text search in title/abstract", "default": ""},
            "file_path": {"type": "string", "description": "Local paper file path", "default": ""},
            "source_hash": {"type": "string", "description": "SHA-256 hash for local-file deduplication", "default": ""},
            "page_count": {"type": "string", "description": "Parsed page count", "default": ""},
            "status": {"type": "string", "description": "UI status such as parsed/recent", "default": ""},
        },
        "required": ["operation"],
    }

    def run(self, operation: str = "", paper_id: str = "", title: str = "",
            authors: str = "", year: str = "", abstract: str = "",
            tag: str = "", tags: str = "", query: str = "", file_path: str = "",
            source_hash: str = "", page_count: str = "", status: str = "", **kwargs) -> str:
        conn = _get_conn()
        try:
            if operation == "save":
                paper_id = paper_id or _local_id(source_hash, title, file_path)
                if not paper_id:
                    return "Error: paper_id, source_hash, title, or file_path required for save"
                duplicate_id = _find_duplicate(conn, paper_id, source_hash, title)
                if duplicate_id and duplicate_id != paper_id:
                    paper_id = duplicate_id
                conn.execute("""
                    INSERT INTO papers
                    (paper_id, title, year, authors, abstract, citation_count, url, tags, favorite,
                     file_path, source_hash, page_count, status, saved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(paper_id) DO UPDATE SET
                        title = COALESCE(NULLIF(excluded.title, ''), papers.title),
                        year = CASE WHEN excluded.year > 0 THEN excluded.year ELSE papers.year END,
                        authors = COALESCE(NULLIF(excluded.authors, ''), papers.authors),
                        abstract = COALESCE(NULLIF(excluded.abstract, ''), papers.abstract),
                        citation_count = CASE WHEN excluded.citation_count > 0 THEN excluded.citation_count ELSE papers.citation_count END,
                        url = COALESCE(NULLIF(excluded.url, ''), papers.url),
                        tags = COALESCE(NULLIF(excluded.tags, ''), papers.tags),
                        favorite = CASE WHEN excluded.favorite > 0 THEN excluded.favorite ELSE papers.favorite END,
                        file_path = COALESCE(NULLIF(excluded.file_path, ''), papers.file_path),
                        source_hash = COALESCE(NULLIF(excluded.source_hash, ''), papers.source_hash),
                        page_count = CASE WHEN excluded.page_count > 0 THEN excluded.page_count ELSE papers.page_count END,
                        status = COALESCE(NULLIF(excluded.status, ''), papers.status),
                        saved_at = CURRENT_TIMESTAMP
                """, (
                    paper_id, title, _int(year), authors, abstract, _int(kwargs.get("citation_count", 0)),
                    kwargs.get("url", ""), _normalize_tags(tags or tag), _int(kwargs.get("favorite", 0)),
                    file_path, source_hash, _int(page_count), status
                ))
                if tags or tag:
                    _replace_tags(conn, paper_id, _split_tags(tags or tag))
                conn.commit()
                return json.dumps({"status": "saved", "paper_id": paper_id, "title": title or paper_id}, ensure_ascii=False)

            elif operation == "search":
                if not query:
                    return "Error: query required for search"
                cur = conn.execute(
                    "SELECT paper_id, title, year, authors, tags, favorite FROM papers WHERE title LIKE ? OR abstract LIKE ? LIMIT 20",
                    (f"%{query}%", f"%{query}%")
                )
                rows = cur.fetchall()
                if not rows:
                    return f"No papers matching '{query}' in database."
                lines = [f"Papers matching '{query}':"]
                for r in rows:
                    fav = " ★" if r[5] else ""
                    lines.append(f"  [{r[0]}] {r[1]} ({r[2]}) tags: {r[4]}{fav}")
                return "\n".join(lines)

            elif operation == "list":
                cur = conn.execute("""
                    SELECT paper_id, title, year, citation_count, favorite, tags, page_count, status
                    FROM papers ORDER BY favorite DESC, saved_at DESC LIMIT 30
                """)
                rows = cur.fetchall()
                if not rows:
                    return "No papers in database yet."
                lines = ["Papers in database:"]
                for i, r in enumerate(rows, 1):
                    fav = "favorite" if r[4] else ""
                    lines.append(f"  {i}. [{r[0]}] {r[1]} ({r[2]}) pages: {r[6]} tags: {r[5]} {fav} {r[7]}")
                return "\n".join(lines)

            elif operation == "list_json":
                cur = conn.execute("""
                    SELECT paper_id, title, year, authors, abstract, venue, citation_count, url,
                           tags, notes, favorite, file_path, source_hash, page_count, status, saved_at
                    FROM papers ORDER BY favorite DESC, saved_at DESC LIMIT 100
                """)
                return json.dumps([_record(row) for row in cur.fetchall()], ensure_ascii=False)

            elif operation == "get":
                if not paper_id:
                    return "Error: paper_id required"
                cur = conn.execute("""
                    SELECT paper_id, title, year, authors, abstract, venue, citation_count, url,
                           tags, notes, favorite, file_path, source_hash, page_count, status, saved_at
                    FROM papers WHERE paper_id = ?
                """, (paper_id,))
                row = cur.fetchone()
                return json.dumps(_record(row) if row else {}, ensure_ascii=False)

            elif operation == "tag":
                if not paper_id or not tag:
                    return "Error: paper_id and tag required"
                conn.execute("INSERT OR IGNORE INTO tags VALUES (?, ?)", (paper_id, tag))
                _sync_tags_column(conn, paper_id)
                conn.commit()
                return f"Tagged {paper_id} with '{tag}'"

            elif operation == "untag":
                if not paper_id or not tag:
                    return "Error: paper_id and tag required"
                conn.execute("DELETE FROM tags WHERE paper_id = ? AND tag = ?", (paper_id, tag))
                _sync_tags_column(conn, paper_id)
                conn.commit()
                return f"Removed tag '{tag}' from {paper_id}"

            elif operation == "set_tags":
                if not paper_id:
                    return "Error: paper_id required"
                _replace_tags(conn, paper_id, _split_tags(tags or tag))
                conn.commit()
                return f"Updated tags for {paper_id}"

            elif operation == "favorite":
                if not paper_id:
                    return "Error: paper_id required"
                conn.execute("UPDATE papers SET favorite = 1 WHERE paper_id = ?", (paper_id,))
                conn.commit()
                return f"Marked favorite: {paper_id}"

            elif operation == "unfavorite":
                if not paper_id:
                    return "Error: paper_id required"
                conn.execute("UPDATE papers SET favorite = 0 WHERE paper_id = ?", (paper_id,))
                conn.commit()
                return f"Removed favorite: {paper_id}"

            elif operation == "favorites":
                cur = conn.execute("SELECT paper_id, title, year, authors FROM papers WHERE favorite = 1 ORDER BY saved_at DESC LIMIT 30")
                rows = cur.fetchall()
                if not rows:
                    return "No favorite papers yet."
                lines = ["Favorite papers:"]
                for r in rows:
                    lines.append(f"  [{r[0]}] {r[1]} ({r[2]}) - {r[3]}")
                return "\n".join(lines)

            elif operation == "search_by_tag":
                if not tag:
                    return "Error: tag required"
                cur = conn.execute("""
                    SELECT p.paper_id, p.title, p.year, p.authors
                    FROM papers p JOIN tags t ON p.paper_id = t.paper_id
                    WHERE t.tag = ? LIMIT 20
                """, (tag,))
                rows = cur.fetchall()
                if not rows:
                    return f"No papers tagged '{tag}'."
                lines = [f"Papers tagged '{tag}':"]
                for r in rows:
                    lines.append(f"  [{r[0]}] {r[1]} ({r[2]})")
                return "\n".join(lines)

            elif operation == "stats":
                cur = conn.execute("SELECT COUNT(*), COALESCE(SUM(citation_count),0), COALESCE(SUM(favorite),0) FROM papers")
                count, total_cites, favorites = cur.fetchone()
                return f"Database: {count} papers, {favorites} favorites, {total_cites} total citations"

            elif operation == "dedupe":
                removed = _dedupe(conn)
                conn.commit()
                return f"Deduplicated paper database. Removed {removed} duplicate record(s)."

            elif operation == "delete":
                if not paper_id:
                    return "Error: paper_id required"
                conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
                conn.execute("DELETE FROM tags WHERE paper_id = ?", (paper_id,))
                conn.commit()
                return f"Deleted paper: {paper_id}"

            return f"Unknown operation: {operation}"
        finally:
            conn.close()


def _int(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _split_tags(tags: str) -> list[str]:
    return [t.strip() for t in str(tags or "").replace("，", ",").split(",") if t.strip()]


def _normalize_tags(tags: str) -> str:
    return ",".join(dict.fromkeys(_split_tags(tags)))


def _replace_tags(conn, paper_id: str, tags: list[str]) -> None:
    conn.execute("DELETE FROM tags WHERE paper_id = ?", (paper_id,))
    for tag in dict.fromkeys(tags):
        conn.execute("INSERT OR IGNORE INTO tags VALUES (?, ?)", (paper_id, tag))
    _sync_tags_column(conn, paper_id)


def _sync_tags_column(conn, paper_id: str) -> None:
    rows = conn.execute("SELECT tag FROM tags WHERE paper_id = ? ORDER BY tag", (paper_id,)).fetchall()
    conn.execute("UPDATE papers SET tags = ? WHERE paper_id = ?", (",".join(row[0] for row in rows), paper_id))


def _local_id(source_hash: str, title: str, file_path: str) -> str:
    if source_hash:
        return f"local:{source_hash[:16]}"
    seed = title or os.path.basename(file_path or "")
    if not seed:
        return ""
    return "local:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _find_duplicate(conn, paper_id: str, source_hash: str, title: str) -> str:
    if source_hash:
        row = conn.execute("SELECT paper_id FROM papers WHERE source_hash = ? LIMIT 1", (source_hash,)).fetchone()
        if row:
            return row[0]
    if title:
        row = conn.execute("SELECT paper_id FROM papers WHERE lower(title) = lower(?) LIMIT 1", (title,)).fetchone()
        if row:
            return row[0]
    return paper_id


def _record(row) -> dict:
    if not row:
        return {}
    return {
        "paper_id": row[0],
        "title": row[1] or row[0],
        "year": row[2] or 0,
        "authors": row[3] or "",
        "abstract": row[4] or "",
        "venue": row[5] or "",
        "citation_count": row[6] or 0,
        "url": row[7] or "",
        "tags": _split_tags(row[8] or ""),
        "notes": row[9] or "",
        "favorite": bool(row[10]),
        "file_path": row[11] or "",
        "source_hash": row[12] or "",
        "page_count": row[13] or 0,
        "status": row[14] or "",
        "saved_at": row[15] or "",
    }


def _dedupe(conn) -> int:
    removed = 0
    duplicate_sets = []
    for column in ("source_hash", "lower(title)"):
        rows = conn.execute(f"""
            SELECT {column}, GROUP_CONCAT(paper_id), COUNT(*)
            FROM papers
            WHERE COALESCE({column}, '') != ''
            GROUP BY {column}
            HAVING COUNT(*) > 1
        """).fetchall()
        duplicate_sets.extend(rows)
    seen = set()
    for _, ids_csv, _ in duplicate_sets:
        ids = [pid for pid in ids_csv.split(",") if pid]
        if len(ids) < 2:
            continue
        if tuple(ids) in seen:
            continue
        seen.add(tuple(ids))
        keep = conn.execute(
            f"SELECT paper_id FROM papers WHERE paper_id IN ({','.join('?' for _ in ids)}) ORDER BY favorite DESC, saved_at DESC LIMIT 1",
            ids,
        ).fetchone()[0]
        for pid in ids:
            if pid == keep:
                continue
            for tag_row in conn.execute("SELECT tag FROM tags WHERE paper_id = ?", (pid,)).fetchall():
                conn.execute("INSERT OR IGNORE INTO tags VALUES (?, ?)", (keep, tag_row[0]))
            conn.execute("DELETE FROM papers WHERE paper_id = ?", (pid,))
            conn.execute("DELETE FROM tags WHERE paper_id = ?", (pid,))
            removed += 1
        _sync_tags_column(conn, keep)
    return removed
