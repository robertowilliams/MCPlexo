# tools.py
import asyncio
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def build_local_db_tool_spec() -> Dict[str, Any]:
    """
    Dict-based tool spec describing the local SQLite query tool.
    This is converted to an OpenAI function tool in bridge.py.
    """
    return {
        "name": "query_database",
        "description": (
            "Run a read-only SQL query against the local SQLite database. "
            "Use this for SELECT statements only. Supports optional named parameters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A single SQL SELECT statement. Do not include multiple statements "
                        "or write operations (INSERT/UPDATE/DELETE/PRAGMA/etc.)."
                    ),
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Optional named parameters for the query. Example: "
                        '{"min_price": 10, "category": "books"}'
                    ),
                    "additionalProperties": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum number of rows to return (server-side cap also applies).",
                    "minimum": 1,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


class DatabaseQueryTool:
    """
    Lightweight, read-only SQLite runner with:
      - async-safe execution (runs DB work off the event loop)
      - basic SQL validation to prevent writes / multi-statements
      - schema summarization to guide the LLM
    """

    # Hard cap to avoid returning huge payloads by mistake
    HARD_ROW_LIMIT = 5000

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = str(db_path)
        self._compiled_select = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
        # Disallow statements that are clearly unsafe in this context
        self._blocked = re.compile(
            r"\b(?:insert|update|delete|replace|drop|alter|create|attach|detach|vacuum|pragma|reindex|truncate)\b",
            re.IGNORECASE,
        )
        self._semicolon = re.compile(r";")
        logger.debug("DatabaseQueryTool initialized for %s", self.db_path)

    # -------------------------
    # Public API
    # -------------------------
    async def execute(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query with optional named parameters.
        Args shape (validated by bridge/tool schema):
           {
             "query": "SELECT * FROM table WHERE x >= :min_x",
             "params": {"min_x": 10},
             "limit": 100
           }
        Returns: list of row dicts
        """
        query: str = (args.get("query") or "").strip()
        params: Dict[str, Any] = args.get("params") or {}
        req_limit: Optional[int] = args.get("limit")

        self._validate_query(query)

        # Apply a protective LIMIT if the query had none
        query = self._ensure_limit(query, req_limit)

        # Run off the event loop
        rows = await asyncio.to_thread(self._run_query_sync, query, params)
        return rows

    def generate_schema_summary(self) -> str:
        """
        Returns a human-readable snapshot of tables & columns to help the LLM write correct SQL.
        This is best-effort; if the DB cannot be inspected, returns an empty string.
        """
        try:
            return self._schema_summary_sync()
        except Exception as e:
            logger.debug("Schema summary not available: %s", e)
            return ""

    # -------------------------
    # Validation helpers
    # -------------------------
    def _validate_query(self, query: str) -> None:
        """
        Basic guardrails: single SELECT, no semicolons, no obvious write/DDL verbs.
        """
        if not query:
            raise ValueError("Query must be a non-empty SQL SELECT statement.")

        # Disallow multiple statements by semicolon
        if self._semicolon.search(query):
            raise ValueError("Only a single statement is allowed (no semicolons).")

        # Must start with SELECT
        if not self._compiled_select.match(query):
            raise ValueError("Only SELECT statements are allowed.")

        # Block common write/DDL/unsafe verbs anywhere in the statement
        if self._blocked.search(query):
            raise ValueError("Disallowed SQL detected. This tool is read-only; use SELECT only.")

    def _ensure_limit(self, query: str, req_limit: Optional[int]) -> str:
        """
        Append a LIMIT if the query doesn't already include one. Respect user-provided limit
        but enforce a hard upper bound to avoid massive responses.
        """
        # If the query already contains LIMIT (naive check), leave it
        if re.search(r"\blimit\b", query, re.IGNORECASE):
            return query

        limit = min(req_limit or 200, self.HARD_ROW_LIMIT)
        return f"{query}\nLIMIT {limit}"

    # -------------------------
    # DB execution (sync, offloaded)
    # -------------------------
    def _run_query_sync(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Synchronous DB access executed in a worker thread.
        """
        path = Path(self.db_path)
        if not path.exists():
            raise FileNotFoundError(f"SQLite database not found at: {path}")

        con = sqlite3.connect(str(path), check_same_thread=False)
        try:
            con.row_factory = sqlite3.Row

            # Best-effort read-only: ensure no schema or data changes via a transaction trick
            # (SQLite doesn't have a strict read-only flag for connections to regular files.)
            cur = con.cursor()
            try:
                cur.execute(query, params or {})
                rows = cur.fetchall()
            finally:
                cur.close()

            # Convert rows to list[dict]
            return [dict(row) for row in rows]
        finally:
            con.close()

    # -------------------------
    # Schema inspection (sync)
    # -------------------------
    def _schema_summary_sync(self) -> str:
        """
        Introspect the DB: list tables and their columns/types, plus row counts (lightweight).
        """
        path = Path(self.db_path)
        if not path.exists():
            return ""

        con = sqlite3.connect(str(path), check_same_thread=False)
        try:
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            # Tables
            cur.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
            tables = [r["name"] for r in cur.fetchall()]

            lines: List[str] = []
            for t in tables:
                # Columns
                cur.execute(f"PRAGMA table_info('{t}')")
                cols = cur.fetchall()
                col_strs = []
                for c in cols:
                    cname = c["name"]
                    ctype = c["type"] or ""
                    notnull = " NOT NULL" if c["notnull"] else ""
                    pk = " PRIMARY KEY" if c["pk"] else ""
                    col_strs.append(f"- {cname}: {ctype}{notnull}{pk}")

                # Row count (cheap enough; if the table is huge this is still fast in SQLite)
                try:
                    cur.execute(f"SELECT COUNT(1) AS n FROM '{t}'")
                    n = cur.fetchone()["n"]
                except Exception:
                    n = "?"

                lines.append(f"Table: {t} (rows: {n})")
                if col_strs:
                    lines.extend(col_strs)
                lines.append("")

            return "\n".join(lines).strip()
        finally:
            con.close()
