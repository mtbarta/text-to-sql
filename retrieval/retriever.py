"""High-level retriever interface over the business-rules index.

The Retriever protocol is the extension point: the RulesRetriever provided here
uses DuckDB BM25, but any class that implements retrieve() satisfies the protocol
— a vector retriever, a hybrid re-ranker, etc. can be dropped in without touching
the agent or evaluation code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from retrieval.index import RulesIndex, SearchResult
from retrieval.parser import MarkdownParser

# Default locations relative to the project root.
_PROJECT_ROOT = Path(__file__).parent.parent
GUIDES_DIR = _PROJECT_ROOT / "evaluation" / "data" / "guides"
INDEX_PATH = _PROJECT_ROOT / "retrieval" / "rules.duckdb"


@runtime_checkable
class Retriever(Protocol):
    """Retrieve relevant RuleChunks for a natural-language query."""

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]: ...


class RulesRetriever:
    """BM25 retriever over the business-rules guides.

    Builds the index on first use if it does not exist, so callers need
    only construct the retriever and call retrieve().
    """

    def __init__(
        self,
        guides_dir: Path = GUIDES_DIR,
        index_path: Path = INDEX_PATH,
    ) -> None:
        self._guides_dir = guides_dir
        self._index = RulesIndex(index_path)

    def ensure_index(self) -> None:
        """Build the index if it has not been built yet."""
        if not self._index.is_built():
            parser = MarkdownParser()
            chunks = parser.parse_directory(self._guides_dir)
            self._index.build(chunks)

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the top-k rule chunks most relevant to query."""
        self.ensure_index()
        return self._index.search(query, k=k)

    def format_context(self, query: str, k: int = 5) -> str:
        """Return retrieved chunks formatted as a prompt-ready context block.

        Convenience method for injecting rules directly into an agent prompt.
        """
        results = self.retrieve(query, k=k)
        if not results:
            return ""
        sections = [r.chunk.format_for_prompt() for r in results]
        return "\n\n---\n\n".join(sections)

    @property
    def chunk_count(self) -> int:
        return self._index.chunk_count()
