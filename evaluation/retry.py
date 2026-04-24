#!/usr/bin/env python3
"""Re-run previously failed evaluation cases against the current agent.

Reads trace files from a prior evaluate run, filters to failures, and
re-evaluates them so you can measure the impact of agent changes without
running the full eval set.

Usage:
    uv run retry logs/run_20240101_120000 --api-key YOUR_API_KEY
    uv run retry logs/run_20240101_120000 --api-key YOUR_API_KEY --concurrency 4
    uv run retry logs/run_20240101_120000 --api-key YOUR_API_KEY --failure-types MISMATCH SQL_ERROR
    uv run retry logs/run_20240101_120000 --api-key YOUR_API_KEY --prompts "top 10 airlines" "average delay"
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live

from evaluation.analyze import TraceRecord
from evaluation.evaluate import (
    EvalCase,
    EvalConfig,
    EvalResult,
    EvalSplitResults,
    FailureType,
    _run_single_eval_worker,
    create_status_table,
    create_tools,
    print_summary,
    run_single_eval,
)
from framework.agent import Agent
from framework.llm import OpenRouterConfig


def load_failed_cases(
    log_dir: Path,
    failure_types: set[str] | None = None,
    prompt_filters: list[str] | None = None,
) -> dict[str, list[EvalCase]]:
    """Load failed cases grouped by split from a previous run's log directory.

    Args:
        log_dir: Root of a prior evaluate run (e.g. logs/run_20240101_120000).
        failure_types: If given, only include cases whose failure_type name is
            in this set. If None, include all failures.
        prompt_filters: If given, only include cases whose prompt contains at
            least one of these substrings (case-insensitive).

    Returns:
        Mapping of split name → list of EvalCase objects to retry.
    """
    cases_by_split: dict[str, list[EvalCase]] = {}

    subdirs = sorted(p for p in log_dir.iterdir() if p.is_dir())
    if not subdirs:
        subdirs = [log_dir]

    for split_dir in subdirs:
        split_name = split_dir.name
        cases: list[EvalCase] = []
        for trace_file in sorted(split_dir.glob("*.json")):
            try:
                record = TraceRecord.from_file(trace_file)
            except Exception:
                continue
            if record.passed:
                continue
            if failure_types is not None and record.failure_type not in failure_types:
                continue
            if prompt_filters is not None:
                prompt_lower = record.prompt.lower()
                if not any(f.lower() in prompt_lower for f in prompt_filters):
                    continue
            cases.append(EvalCase(prompt=record.prompt, gold_query=record.gold_query))

        if cases:
            cases_by_split[split_name] = cases

    return cases_by_split


def retry_split(
    tools: dict,
    split_name: str,
    cases: list[EvalCase],
    console: Console,
    api_key: str,
    concurrency: int = 1,
    log_dir: Path | None = None,
    verbose: bool = False,
) -> EvalSplitResults:
    """Re-evaluate a list of cases for one split."""
    split_results = EvalSplitResults(name=split_name)

    split_log_dir: Path | None = None
    if log_dir is not None:
        split_log_dir = log_dir / split_name
        split_log_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Logging traces to: {split_log_dir}[/dim]")

    console.print(f"\n[bold cyan]Retrying: {split_name}[/bold cyan]")
    console.print(f"[dim]{len(cases)} failed case(s) to retry[/dim]")
    console.print(f"[dim]Running with concurrency: {concurrency}[/dim]\n")

    if concurrency == 1:
        llm_config = OpenRouterConfig(api_key=api_key)
        agent = Agent(config=llm_config, tools=tools)
        eval_config = EvalConfig(verbose=verbose, log_dir=split_log_dir)

        with Live(
            create_status_table(split_name, [], len(cases)),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            for case in cases:
                agent.reset_conversation()
                result = run_single_eval(agent, case, eval_config)
                split_results.results.append(result)
                live.update(
                    create_status_table(split_name, split_results.results, len(cases))
                )
    else:
        results_by_index: dict[int, EvalResult] = {}
        results_lock = threading.Lock()

        with Live(
            create_status_table(split_name, [], len(cases)),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        _run_single_eval_worker,
                        case,
                        idx,
                        tools,
                        api_key,
                        split_log_dir,
                        verbose,
                    ): idx
                    for idx, case in enumerate(cases)
                }

                for future in as_completed(futures):
                    try:
                        case_index, result = future.result()
                    except Exception as e:
                        case_index = futures[future]
                        result = EvalResult(
                            case=cases[case_index],
                            submitted_query=None,
                            passed=False,
                            error=f"Worker exception: {e!s}",
                            failure_type=FailureType.EXCEPTION,
                        )

                    with results_lock:
                        results_by_index[case_index] = result
                        ordered = [
                            results_by_index[i]
                            for i in range(len(cases))
                            if i in results_by_index
                        ]
                        live.update(
                            create_status_table(split_name, ordered, len(cases))
                        )

        split_results.results = [results_by_index[i] for i in range(len(cases))]

    return split_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run previously failed evaluation cases against the current agent"
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Directory of a previous evaluate run (e.g. logs/run_20240101_120000)",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenRouter API key",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel evaluations to run (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--failure-types",
        nargs="+",
        choices=[ft.name for ft in FailureType if ft != FailureType.NONE],
        default=None,
        metavar="TYPE",
        help=(
            "Only retry cases with these failure types "
            f"(choices: {', '.join(ft.name for ft in FailureType if ft != FailureType.NONE)}). "
            "Default: retry all failures."
        ),
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        metavar="SUBSTRING",
        help=(
            "Only retry cases whose prompt contains at least one of these substrings "
            "(case-insensitive). Useful for targeting a single question or domain."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    console.print("[bold]SQL Agent — Retry Failed Cases[/bold]")

    if not args.log_dir.is_dir():
        console.print(f"[red]Error: {args.log_dir} is not a directory[/red]")
        raise SystemExit(1)

    failure_types = set(args.failure_types) if args.failure_types else None
    cases_by_split = load_failed_cases(
        args.log_dir,
        failure_types=failure_types,
        prompt_filters=args.prompts,
    )

    if not cases_by_split:
        console.print("[green]No matching failed cases found — nothing to retry.[/green]")
        return

    total_cases = sum(len(c) for c in cases_by_split.values())
    filters: list[str] = []
    if failure_types:
        filters.append(f"types: {', '.join(sorted(failure_types))}")
    if args.prompts:
        filters.append(f"prompts: {', '.join(repr(p) for p in args.prompts)}")
    filter_desc = f" ({'; '.join(filters)})" if filters else ""
    console.print(
        f"[dim]Found {total_cases} failed case(s) across "
        f"{len(cases_by_split)} split(s){filter_desc}[/dim]"
    )

    tools = create_tools()
    console.print(f"[dim]Agent tools: {', '.join(tools.keys())}[/dim]")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"retry_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Saving traces to: {log_dir}[/dim]")

    all_results: list[EvalSplitResults] = []
    for split_name, cases in sorted(cases_by_split.items()):
        split_results = retry_split(
            tools=tools,
            split_name=split_name,
            cases=cases,
            console=console,
            api_key=args.api_key,
            concurrency=args.concurrency,
            log_dir=log_dir,
            verbose=args.verbose,
        )
        all_results.append(split_results)

    print_summary(all_results, console, verbose=args.verbose)
