"""DuckDB-backed full-text search index for business-rule chunks.

Uses DuckDB's built-in FTS extension (BM25) to rank chunks by relevance.
The index lives in a dedicated .duckdb file separate from the main data DB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from retrieval.parser import RuleChunk

# Table name used for both the DuckDB table and the derived FTS function.
_TABLE = "chunks"


@dataclass(frozen=True)
class SearchResult:
    """A ranked match returned by a search query."""

    chunk: RuleChunk
    score: float


class RulesIndex:
    """Build and query a BM25 full-text search index over RuleChunks.

    The index is persisted to a DuckDB file so it can be built once and
    reused across runs. Call build() to (re)populate it, and search() to
    query it.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[RuleChunk], *, overwrite: bool = False) -> int:
        """Populate the index from a list of chunks.

        Args:
            chunks: Parsed rule chunks to index.
            overwrite: If True, drop and recreate an existing index.

        Returns:
            Number of chunks indexed.

        Raises:
            FileExistsError: If the index already exists and overwrite=False.
        """
        if self._db_path.exists() and not overwrite:
            raise FileExistsError(
                f"Index already exists at {self._db_path}. "
                "Pass overwrite=True to rebuild."
            )

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if self._db_path.exists():
            self._db_path.unlink()

        conn = duckdb.connect(str(self._db_path))
        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
            self._create_table(conn)
            self._insert_chunks(conn, chunks)
            self._create_fts_index(conn)
        finally:
            conn.close()

        return len(chunks)

    def _create_table(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(f"""
            CREATE TABLE {_TABLE} (
                chunk_id      VARCHAR PRIMARY KEY,
                source        VARCHAR NOT NULL,
                doc_title     VARCHAR NOT NULL,
                section_title VARCHAR NOT NULL,
                content       TEXT    NOT NULL
            )
        """)

    def _insert_chunks(
        self, conn: duckdb.DuckDBPyConnection, chunks: list[RuleChunk]
    ) -> None:
        rows = [
            (c.chunk_id, c.source, c.doc_title, c.section_title, c.content)
            for c in chunks
        ]
        conn.executemany(
            f"INSERT INTO {_TABLE} VALUES (?, ?, ?, ?, ?)", rows
        )

    def _create_fts_index(self, conn: duckdb.DuckDBPyConnection) -> None:
        # Index content + section_title so queries match both rule text and headings.
        conn.execute(f"""
            PRAGMA create_fts_index(
                '{_TABLE}', 'chunk_id',
                'content', 'section_title', 'doc_title',
                stemmer='porter',
                stopwords='english',
                lower=1,
                overwrite=1
            )
        """)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the top-k chunks ranked by BM25 relevance.

        Args:
            query: Natural-language or keyword search string.
            k: Maximum number of results to return.

        Returns:
            List of SearchResult ordered by descending relevance score.
        """
        conn = duckdb.connect(str(self._db_path), read_only=True)
        try:
            conn.execute("LOAD fts")
            rows = conn.execute(
                f"""
                SELECT
                    fts_main_{_TABLE}.match_bm25(chunk_id, ?) AS score,
                    chunk_id, source, doc_title, section_title, content
                FROM {_TABLE}
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT ?
                """,
                [query, k],
            ).fetchall()
        finally:
            conn.close()

        return [
            SearchResult(
                chunk=RuleChunk(
                    chunk_id=row[1],
                    source=row[2],
                    doc_title=row[3],
                    section_title=row[4],
                    content=row[5],
                ),
                score=row[0],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_built(self) -> bool:
        """Return True if the index file exists and contains data."""
        if not self._db_path.exists():
            return False
        try:
            conn = duckdb.connect(str(self._db_path), read_only=True)
            count = conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()
            conn.close()
            return count is not None and count[0] > 0
        except Exception:
            return False

    def chunk_count(self) -> int:
        """Return the number of indexed chunks (0 if not built)."""
        if not self.is_built():
            return 0
        conn = duckdb.connect(str(self._db_path), read_only=True)
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
