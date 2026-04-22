"""Parse business-rule markdown guides into searchable chunks.

Each guide file is split at H2 boundaries. The H1 heading becomes the
document title shared across all chunks from that file; each H2 block
(heading + body) becomes one RuleChunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuleChunk:
    """A single section from a business-rules guide, ready for indexing."""

    chunk_id: str       # "{source}#{section_slug}"  — stable, unique key
    source: str         # stem of the originating file, e.g. "financial_operations"
    doc_title: str      # H1 heading of the guide
    section_title: str  # H2 heading of this chunk
    content: str        # full section text (heading + body), suitable for injection

    def format_for_prompt(self) -> str:
        """Return a compact, prompt-ready representation."""
        return f"### {self.doc_title} — {self.section_title}\n{self.content}"


def _slugify(text: str) -> str:
    """Convert a heading to a lowercase, hyphen-separated identifier."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class MarkdownParser:
    """Parse a directory (or individual file) of Markdown guides into RuleChunks.

    Chunking strategy: one chunk per H2 section. Any content before the first H2
    (e.g. an introduction paragraph after the H1) is collected into a synthetic
    "Overview" section rather than silently dropped.
    """

    def parse_file(self, path: Path) -> list[RuleChunk]:
        """Parse a single markdown file into one RuleChunk per H2 section."""
        source = path.stem
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        doc_title = source.replace("_", " ").title()
        chunks: list[RuleChunk] = []

        current_section: str | None = None
        current_body: list[str] = []

        for line in lines:
            h1 = re.match(r"^#\s+(.+)$", line)
            h2 = re.match(r"^##\s+(.+)$", line)

            if h1:
                doc_title = h1.group(1).strip()
                continue

            if h2:
                if current_section is not None:
                    chunks.append(
                        self._make_chunk(source, doc_title, current_section, current_body)
                    )
                elif current_body:
                    # Pre-H2 content — treat as an "Overview" section
                    chunks.append(
                        self._make_chunk(source, doc_title, "Overview", current_body)
                    )
                current_section = h2.group(1).strip()
                current_body = []
                continue

            current_body.append(line)

        # Flush the final section
        if current_section is not None:
            chunks.append(
                self._make_chunk(source, doc_title, current_section, current_body)
            )

        return chunks

    def parse_directory(self, directory: Path) -> list[RuleChunk]:
        """Parse all *.md files in a directory, sorted by filename."""
        chunks: list[RuleChunk] = []
        for path in sorted(directory.glob("*.md")):
            chunks.extend(self.parse_file(path))
        return chunks

    def _make_chunk(
        self,
        source: str,
        doc_title: str,
        section_title: str,
        body_lines: list[str],
    ) -> RuleChunk:
        body = "\n".join(body_lines).strip()
        content = f"## {section_title}\n\n{body}" if body else f"## {section_title}"
        return RuleChunk(
            chunk_id=f"{source}#{_slugify(section_title)}",
            source=source,
            doc_title=doc_title,
            section_title=section_title,
            content=content,
        )
