#!/usr/bin/env python3
"""Post-hoc analysis of agent evaluation traces.

Reads trace JSON files produced by the --verbose evaluate run and generates a
human-readable report of failures, behavioral patterns, and per-case details.

Usage:
    uv run analyze logs/run_20260420_123456/
    uv run analyze logs/run_20260420_123456/ --show-queries

The Analyzer protocol is designed for extension: swap in an LLMJudgeAnalyzer
alongside or instead of the built-in RuleBasedAnalyzer without changing the
rendering or CLI layer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from rich.console import Console
from rich.table import Table

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

FAILURE_CATEGORIES = [
    "wrong_schema",      # queried the wrong schema or table
    "null_handling",     # spurious IS NULL guards or wrong NULL semantics
    "wrong_groupby",     # wrong GROUP BY keys
    "wrong_column",      # wrong column names or aliases in SELECT
    "wrong_filter",      # missing or incorrect WHERE conditions
    "wrong_aggregation", # wrong aggregation function or scope
    "over_engineering",  # unnecessary CTEs/subqueries introducing semantic differences
    "other",
]

# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class ToolEvent:
    """A single tool call extracted from the agent event stream."""

    name: str
    result_preview: str  # truncated for display


@dataclass(frozen=True)
class TraceRecord:
    """Raw deserialized content of a single trace file."""

    trace_id: str
    timestamp: str
    duration_seconds: float | None
    prompt: str
    gold_query: str
    events: list[dict[str, Any]]
    passed: bool
    submitted_query: str | None
    failure_type: str  # mirrors FailureType.name
    error: str | None

    @classmethod
    def from_file(cls, path: Path) -> TraceRecord:
        with open(path) as f:
            data = json.load(f)
        return cls(
            trace_id=data["trace_id"],
            timestamp=data["timestamp"],
            duration_seconds=data.get("duration_seconds"),
            prompt=data["case"]["prompt"],
            gold_query=data["case"]["gold_query"],
            events=data["events"],
            passed=data["result"]["passed"],
            submitted_query=data["result"]["submitted_query"],
            failure_type=data["result"]["failure_type"],
            error=data["result"]["error"],
        )

    def tool_events(self) -> list[ToolEvent]:
        """Return ordered tool calls from the event stream."""
        return [
            ToolEvent(
                name=e["data"].get("name", "unknown"),
                result_preview=str(e["data"].get("result", ""))[:120],
            )
            for e in self.events
            if e["type"] == "TOOL_EXECUTION_END"
        ]

    def iteration_count(self) -> int:
        return sum(1 for e in self.events if e["type"] == "ITERATION_START")


# =============================================================================
# Insights
# =============================================================================


class InsightCode(StrEnum):
    """Canonical codes for behavioral insights.

    String values are stable identifiers suitable for grouping, filtering, or
    passing as labels to an LLM judge.
    """

    NO_SCHEMA_DISCOVERY = "no_schema_discovery"
    NO_TABLE_INSPECTION = "no_table_inspection"
    NO_QUERY_TESTING = "no_query_testing"
    REPEATED_QUERY_ERRORS = "repeated_query_errors"
    MAX_ITERATIONS_EXHAUSTED = "max_iterations_exhausted"
    SINGLE_QUERY_NO_ITERATION = "single_query_no_iteration"


@dataclass(frozen=True)
class Insight:
    code: InsightCode
    detail: str


# =============================================================================
# Analyzer protocol
# =============================================================================


@dataclass
class TraceAnalysis:
    """Enriched analysis of a single trace, produced by an Analyzer."""

    trace: TraceRecord
    tool_sequence: list[str]
    iterations: int
    insights: list[Insight] = field(default_factory=list)
    # Extra annotations from richer analyzers (e.g. LLM judge verdicts)
    annotations: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Analyzer(Protocol):
    """Analyze a single trace and return enriched findings."""

    def analyze(self, trace: TraceRecord) -> TraceAnalysis: ...


# =============================================================================
# Rule-based analyzer
# =============================================================================


class RuleBasedAnalyzer:
    """Heuristic analysis of agent behavior from event traces.

    Each rule is a private method that receives the trace context and appends
    to the insight list — easy to add, remove, or reorder individual checks.
    """

    def analyze(self, trace: TraceRecord) -> TraceAnalysis:
        tool_events = trace.tool_events()
        tool_names = [e.name for e in tool_events]
        iterations = trace.iteration_count()
        insights: list[Insight] = []

        self._check_schema_discovery(tool_names, insights)
        self._check_table_inspection(tool_names, insights)
        self._check_query_testing(tool_names, trace, insights)
        self._check_repeated_errors(tool_events, insights)
        self._check_max_iterations(trace, iterations, insights)
        self._check_single_query(tool_names, trace, insights)

        return TraceAnalysis(
            trace=trace,
            tool_sequence=tool_names,
            iterations=iterations,
            insights=insights,
        )

    def _check_schema_discovery(
        self, tool_names: list[str], insights: list[Insight]
    ) -> None:
        discovery_tools = {"list_schemas", "list_tables"}
        if not discovery_tools.intersection(tool_names):
            insights.append(Insight(
                InsightCode.NO_SCHEMA_DISCOVERY,
                "Agent never called list_schemas or list_tables.",
            ))

    def _check_table_inspection(
        self, tool_names: list[str], insights: list[Insight]
    ) -> None:
        if "describe_table" not in tool_names:
            insights.append(Insight(
                InsightCode.NO_TABLE_INSPECTION,
                "Agent never called describe_table.",
            ))

    def _check_query_testing(
        self,
        tool_names: list[str],
        trace: TraceRecord,
        insights: list[Insight],
    ) -> None:
        if "run_query" not in tool_names and trace.submitted_query:
            insights.append(Insight(
                InsightCode.NO_QUERY_TESTING,
                "Agent submitted without testing the query via run_query.",
            ))

    def _check_repeated_errors(
        self, tool_events: list[ToolEvent], insights: list[Insight]
    ) -> None:
        error_count = sum(
            1
            for e in tool_events
            if e.name == "run_query" and e.result_preview.startswith("Query error")
        )
        if error_count >= 2:
            insights.append(Insight(
                InsightCode.REPEATED_QUERY_ERRORS,
                f"Agent hit {error_count} query errors during exploration.",
            ))

    def _check_max_iterations(
        self, trace: TraceRecord, iterations: int, insights: list[Insight]
    ) -> None:
        if trace.failure_type == "NO_SUBMISSION" and iterations >= 28:
            insights.append(Insight(
                InsightCode.MAX_ITERATIONS_EXHAUSTED,
                f"Agent exhausted max iterations ({iterations}) without submitting.",
            ))

    def _check_single_query(
        self,
        tool_names: list[str],
        trace: TraceRecord,
        insights: list[Insight],
    ) -> None:
        if (
            tool_names.count("run_query") == 1
            and tool_names[-1] == "submit_answer"
            and tool_names[-2] == "run_query"
        ):
            insights.append(Insight(
                InsightCode.SINGLE_QUERY_NO_ITERATION,
                "Agent ran exactly one query then submitted immediately.",
            ))


# =============================================================================
# LLM judge analyzer
# =============================================================================


class LLMJudgeAnalyzer:
    """Wraps RuleBasedAnalyzer and adds LLM-based failure categorization for MISMATCHes.

    Uses a fast model via OpenRouter to classify why the submitted SQL
    produced different results than the gold query. Only makes API calls for
    MISMATCH failures that have a submitted query.
    """

    def __init__(self, api_key: str, model: str = "openai/gpt-4.1-mini") -> None:
        self._base = RuleBasedAnalyzer()
        self._api_key = api_key
        self._model = model
        self._client = httpx.Client(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def analyze(self, trace: TraceRecord) -> TraceAnalysis:
        analysis = self._base.analyze(trace)
        if trace.failure_type == "MISMATCH" and trace.submitted_query:
            verdict = self._judge(trace)
            analysis.annotations["llm_category"] = verdict["category"]
            analysis.annotations["llm_reason"] = verdict["reason"]
        return analysis

    def _judge(self, trace: TraceRecord) -> dict[str, str]:
        categories_list = "\n".join(f"- {c}" for c in FAILURE_CATEGORIES)
        prompt = (
            f"A SQL agent was given this question:\n{trace.prompt}\n\n"
            f"The correct (gold) SQL query was:\n{trace.gold_query}\n\n"
            f"The agent submitted this SQL query:\n{trace.submitted_query}\n\n"
            "The submitted query produced different results than the gold query.\n"
            "Categorize the PRIMARY reason for the mismatch. Choose exactly one:\n"
            f"{categories_list}\n\n"
            'Respond with JSON only: {"category": "<category>", "reason": "<one sentence>"}'
        )
        try:
            response = self._client.post(
                OPENROUTER_API_URL,
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(content)
            if result.get("category") not in FAILURE_CATEGORIES:
                result["category"] = "other"
            return result
        except Exception as e:
            return {"category": "other", "reason": f"Judge error: {e}"}


# =============================================================================
# Report rendering
# =============================================================================


def render_report(
    analyses: list[TraceAnalysis],
    console: Console,
    *,
    show_queries: bool,
) -> None:
    """Render a human-readable analysis report."""
    failures = [a for a in analyses if not a.trace.passed]
    total = len(analyses)
    passed = total - len(failures)

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold]Failure Analysis Report[/bold]")
    pass_rate = passed / total if total else 0.0
    style = "green" if pass_rate >= 0.8 else "yellow" if pass_rate >= 0.5 else "red"
    console.print(
        f"  Total: [bold]{total}[/bold]  "
        f"Passed: [green]{passed}[/green]  "
        f"Failed: [red]{len(failures)}[/red]  "
        f"Pass rate: [bold {style}]{pass_rate:.1%}[/bold {style}]"
    )
    console.print()

    if not failures:
        console.print("[green]All cases passed — nothing to analyze.[/green]")
        return

    # ── Failure type distribution ─────────────────────────────────────────────
    type_counts: Counter[str] = Counter(a.trace.failure_type for a in failures)
    type_table = Table(
        title="Failure Type Distribution",
        show_header=True,
        header_style="bold",
        show_lines=False,
    )
    type_table.add_column("Failure Type", style="red")
    type_table.add_column("Count", justify="right")
    type_table.add_column("% of failures", justify="right")
    for ft, count in type_counts.most_common():
        pct = count / len(failures) * 100
        type_table.add_row(ft, str(count), f"{pct:.0f}%")
    console.print(type_table)
    console.print()

    # ── Behavioral insight distribution ───────────────────────────────────────
    insight_counts: Counter[str] = Counter(
        i.code.value for a in failures for i in a.insights
    )
    if insight_counts:
        insight_table = Table(
            title="Behavioral Insights (failures only)",
            show_header=True,
            header_style="bold",
            show_lines=False,
        )
        insight_table.add_column("Insight Code", style="yellow")
        insight_table.add_column("Count", justify="right")
        insight_table.add_column("% of failures", justify="right")
        for code, count in insight_counts.most_common():
            pct = count / len(failures) * 100
            insight_table.add_row(code, str(count), f"{pct:.0f}%")
        console.print(insight_table)
        console.print()

    # ── LLM judge category distribution (when available) ─────────────────────
    judged = [a for a in failures if "llm_category" in a.annotations]
    if judged:
        category_counts: Counter[str] = Counter(
            a.annotations["llm_category"] for a in judged
        )
        judge_table = Table(
            title=f"LLM Judge Category Distribution ({len(judged)} MISMATCH cases)",
            show_header=True,
            header_style="bold",
            show_lines=False,
        )
        judge_table.add_column("Category", style="cyan")
        judge_table.add_column("Count", justify="right")
        judge_table.add_column("% of judged", justify="right")
        for cat, count in category_counts.most_common():
            pct = count / len(judged) * 100
            judge_table.add_row(cat, str(count), f"{pct:.0f}%")
        console.print(judge_table)
        console.print()

    # ── Per-failure details grouped by type ───────────────────────────────────
    by_type: dict[str, list[TraceAnalysis]] = {}
    for a in failures:
        by_type.setdefault(a.trace.failure_type, []).append(a)

    for failure_type, group in sorted(by_type.items()):
        console.rule(f"[red]{failure_type}[/red]  ({len(group)} cases)")
        for idx, analysis in enumerate(group, 1):
            _render_failure(console, idx, analysis, show_queries=show_queries)


def _render_failure(
    console: Console,
    idx: int,
    analysis: TraceAnalysis,
    *,
    show_queries: bool,
) -> None:
    trace = analysis.trace
    prompt = trace.prompt[:120] + "…" if len(trace.prompt) > 120 else trace.prompt
    duration = f"{trace.duration_seconds:.1f}s" if trace.duration_seconds else "?"
    tool_chain = " → ".join(analysis.tool_sequence) if analysis.tool_sequence else "none"

    console.print(f"\n  [bold]{idx}.[/bold] {prompt}")
    console.print(
        f"     [dim]iter={analysis.iterations}  dur={duration}  "
        f"tools: {tool_chain}[/dim]"
    )

    if trace.error:
        console.print(f"     [red]error:[/red] {trace.error}")

    for insight in analysis.insights:
        console.print(f"     [yellow]⚠[/yellow]  {insight.detail}")

    if show_queries:
        submitted = trace.submitted_query or "(none)"
        console.print(f"     [cyan]submitted:[/cyan] {submitted[:300]}")
        console.print(f"     [green]gold:     [/green] {trace.gold_query[:300]}")

    # Render any extra annotations (e.g. from an LLM judge)
    for key, value in analysis.annotations.items():
        console.print(f"     [magenta]{key}:[/magenta] {value}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze agent evaluation traces.\n\n"
            "Point at a directory produced by: uv run evaluate --verbose ...\n\n"
            "Examples:\n"
            "  uv run analyze logs/run_20260420_123456/\n"
            "  uv run analyze logs/run_20260420_123456/ --show-queries\n"
            "  uv run analyze logs/run_20260420_123456/ --llm-judge --api-key KEY"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Path to trace log directory (e.g. logs/run_20260420_123456/).",
    )
    parser.add_argument(
        "--show-queries",
        action="store_true",
        help="Include submitted and gold queries in per-failure details.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use an LLM to categorize MISMATCH failures (requires --api-key).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (required when --llm-judge is set).",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4.1-mini",
        help="Model to use for LLM judging (default: openai/gpt-4.1-mini).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_analysis"),
        help="Directory to save the analysis report (default: experiment_analysis/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    log_dir: Path = args.log_dir
    if not log_dir.exists():
        console.print(f"[red]Directory not found: {log_dir}[/red]")
        raise SystemExit(1)

    trace_files = sorted(log_dir.glob("*.json"))
    if not trace_files:
        console.print(f"[red]No trace files (*.json) found in {log_dir}[/red]")
        raise SystemExit(1)

    console.print(f"[dim]Loading {len(trace_files)} traces from {log_dir}[/dim]")

    traces: list[TraceRecord] = []
    for path in trace_files:
        try:
            traces.append(TraceRecord.from_file(path))
        except Exception as e:
            console.print(f"[yellow]Skipping {path.name}: {e}[/yellow]")

    if not traces:
        console.print("[yellow]No valid traces to analyze.[/yellow]")
        return

    if args.llm_judge:
        if not args.api_key:
            console.print("[red]--api-key is required when --llm-judge is set.[/red]")
            raise SystemExit(1)
        analyzer: Analyzer = LLMJudgeAnalyzer(
            api_key=args.api_key,
            model=args.judge_model,
        )
        console.print(
            f"[dim]LLM judge enabled (model: {args.judge_model})[/dim]"
        )
    else:
        analyzer = RuleBasedAnalyzer()

    analyses = [analyzer.analyze(t) for t in traces]

    render_report(analyses, console, show_queries=args.show_queries)

    # Derive the run name: if log_dir is a split subdir (e.g. .../run_XYZ/evals_easy),
    # name the file after the run directory; otherwise use log_dir itself.
    parent = log_dir.parent
    run_name = parent.name if parent.name.startswith("run_") else log_dir.name

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / run_name
    with open(output_path, "w") as f:
        file_console = Console(file=f, no_color=True, highlight=False, width=100)
        render_report(analyses, file_console, show_queries=True)
    console.print(f"[dim]Report saved to: {output_path}[/dim]")


if __name__ == "__main__":
    main()
