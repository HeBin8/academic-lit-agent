"""Tool 3: SQLite paper database - store, deduplicate, query, and tag papers."""

import sqlite3, json, os, pathlib
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
    return conn


class SQLitePaperDBTool(BaseTool):
    name = "sqlite_paper_db"
    description = "Store, retrieve, tag, and manage paper metadata in a local SQLite database. Supports save, search_by_tag, list_all, delete, and tag operations."
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["save", "search", "list", "delete", "tag", "favorite", "favorites", "search_by_tag", "stats"],
                "description": "Database operation to perform",
            },
            "paper_id": {"type": "string", "description": "Semantic Scholar paper ID", "default": ""},
            "title": {"type": "string", "description": "Paper title", "default": ""},
            "authors": {"type": "string", "description": "Author names", "default": ""},
            "year": {"type": "string", "description": "Publication year", "default": ""},
            "abstract": {"type": "string", "description": "Paper abstract", "default": ""},
            "tag": {"type": "string", "description": "Tag name for tag/search_by_tag operations", "default": ""},
            "query": {"type": "string", "description": "Text search in title/abstract", "default": ""},
        },
        "required": ["operation"],
    }

    def run(self, operation: str = "", paper_id: str = "", title: str = "",
            authors: str = "", year: str = "", abstract: str = "",
            tag: str = "", query: str = "", **kwargs) -> str:
        conn = _get_conn()
        try:
            if operation == "save":
                if not paper_id:
                    return "Error: paper_id required for save"
                conn.execute("""
                    INSERT OR REPLACE INTO papers
                    (paper_id, title, year, authors, abstract, citation_count, url, saved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (paper_id, title, _int(year), authors, abstract, _int(kwargs.get("citation_count", 0)), kwargs.get("url", "")))
                conn.commit()
                return f"Saved paper: {title or paper_id}"

            elif operation == "search":
                if not query:
                    return "Error: query required for search"
                cur = conn.execute(
                    "SELECT paper_id, title, year, authors, tags FROM papers WHERE title LIKE ? OR abstract LIKE ? LIMIT 20",
                    (f"%{query}%", f"%{query}%")
                )
                rows = cur.fetchall()
                if not rows:
                    return f"No papers matching '{query}' in database."
                lines = [f"Papers matching '{query}':"]
                for r in rows:
                    lines.append(f"  [{r[0]}] {r[1]} ({r[2]}) tags: {r[4]}")
                return "\n".join(lines)

            elif operation == "list":
                cur = conn.execute("SELECT paper_id, title, year, citation_count, favorite FROM papers ORDER BY favorite DESC, citation_count DESC LIMIT 30")
                rows = cur.fetchall()
                if not rows:
                    return "No papers in database yet."
                lines = ["Papers in database (sorted by citations):"]
                for i, r in enumerate(rows, 1):
                    fav = "favorite" if r[4] else ""
                    lines.append(f"  {i}. [{r[0]}] {r[1]} ({r[2]}) - citations: {r[3]} {fav}")
                return "\n".join(lines)

            elif operation == "tag":
                if not paper_id or not tag:
                    return "Error: paper_id and tag required"
                conn.execute("INSERT OR IGNORE INTO tags VALUES (?, ?)", (paper_id, tag))
                conn.execute("UPDATE papers SET tags = COALESCE(tags || ',', '') || ? WHERE paper_id = ?", (tag, paper_id))
                conn.commit()
                return f"Tagged {paper_id} with '{tag}'"

            elif operation == "favorite":
                if not paper_id:
                    return "Error: paper_id required"
                conn.execute("UPDATE papers SET favorite = 1 WHERE paper_id = ?", (paper_id,))
                conn.commit()
                return f"Marked favorite: {paper_id}"

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
                cur = conn.execute("SELECT COUNT(*), COALESCE(SUM(citation_count),0) FROM papers")
                count, total_cites = cur.fetchone()
                return f"Database: {count} papers, {total_cites} total citations"

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
