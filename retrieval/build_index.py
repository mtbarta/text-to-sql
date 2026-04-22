"""CLI for building (or rebuilding) the business-rules search index.

Usage:
    uv run build-index                 # build from default guides directory
    uv run build-index --rebuild       # force rebuild even if index exists
    uv run build-index --query "performing loans"  # build then run a test query
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from retrieval.index import RulesIndex
from retrieval.parser import MarkdownParser
from retrieval.retriever import GUIDES_DIR, INDEX_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the business-rules BM25 search index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--guides-dir",
        type=Path,
        default=GUIDES_DIR,
        help=f"Directory containing markdown guide files (default: {GUIDES_DIR})",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=INDEX_PATH,
        help=f"Output path for the DuckDB index file (default: {INDEX_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if the index already exists.",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="After building, run a test search query and display results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    guides_dir: Path = args.guides_dir
    index_path: Path = args.index_path

    if not guides_dir.exists():
        console.print(f"[red]Guides directory not found: {guides_dir}[/red]")
        raise SystemExit(1)

    # ── Parse ─────────────────────────────────────────────────────────────────
    console.print(f"[dim]Parsing guides from {guides_dir} ...[/dim]")
    t0 = time.monotonic()
    parser = MarkdownParser()
    chunks = parser.parse_directory(guides_dir)
    parse_time = time.monotonic() - t0

    sources = {c.source for c in chunks}
    console.print(
        f"  Parsed [bold]{len(chunks)}[/bold] chunks "
        f"from [bold]{len(sources)}[/bold] files "
        f"in {parse_time:.2f}s"
    )

    # ── Index ─────────────────────────────────────────────────────────────────
    index = RulesIndex(index_path)

    if index.is_built() and not args.rebuild:
        console.print(
            f"[yellow]Index already exists at {index_path} "
            f"({index.chunk_count()} chunks). "
            "Pass --rebuild to overwrite.[/yellow]"
        )
    else:
        console.print(f"[dim]Building FTS index at {index_path} ...[/dim]")
        t1 = time.monotonic()
        indexed = index.build(chunks, overwrite=args.rebuild)
        index_time = time.monotonic() - t1
        console.print(
            f"  Indexed [bold green]{indexed}[/bold green] chunks "
            f"in {index_time:.2f}s"
        )

    # ── Optional test query ───────────────────────────────────────────────────
    if args.query:
        console.print(f'\n[dim]Test query: "{args.query}"[/dim]')
        results = index.search(args.query, k=5)

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold", show_lines=True)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Source", width=28)
        table.add_column("Section")

        for r in results:
            table.add_row(
                f"{r.score:.4f}",
                r.chunk.source,
                r.chunk.section_title,
            )
        console.print(table)


if __name__ == "__main__":
    main()
