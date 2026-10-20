"""Command-line demo of the five onboarding analyses.

uv run python -m opspilot.analytics
uv run python -m opspilot.analytics --segment Enterprise --period-start 2026-04-01
uv run python -m opspilot.analytics --json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from opspilot.analytics.engine import DEFAULT_DATA_DIR, OnboardingAnalytics, run_all
from opspilot.analytics.models import (
    AnalysisResult,
    CompletionResult,
    DelayTrendResult,
    Filters,
    StageAbandonmentResult,
    SupportTicketResult,
    TimeToOnboardResult,
)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _num(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.1f}{suffix}"


def _header(number: int, result: AnalysisResult) -> list[str]:
    return [
        "",
        f"{number}. {result.question}",
        f"   Definition: {result.definition}",
    ]


def _completion(r: CompletionResult) -> list[str]:
    lines = [
        (
            f"   {_pct(r.value)} of {r.sample_size} eligible customers completed onboarding "
            f"({r.completed} completed, {r.abandoned} abandoned, {r.in_progress} in progress; "
            f"{r.excluded_insufficient_follow_up} too recent to judge)."
        ),
    ]
    lines += [
        f"     {b.segment or 'Unknown':<11} {_pct(b.completion_rate):>6}  "
        f"(n={b.eligible_customers})"
        for b in r.by_segment
    ]
    return lines


def _time_to_onboard(r: TimeToOnboardResult) -> list[str]:
    lines = [f"   Overall median: {_num(r.value, ' days')} (n={r.sample_size})"]
    lines.append(
        f"     {'Segment':<11} {'Median':>8} {'Target':>7} {'Done':>5} {'Censored':>9}"
        f" {'Completed-only median':>22}"
    )
    lines += [
        f"     {b.segment or 'Unknown':<11} {_num(b.median_days):>8} "
        f"{b.target_days or '-':>7} {b.completed:>5} {b.censored:>9} "
        f"{_num(b.completed_only_median_days):>22}"
        for b in r.by_segment
    ]
    return lines


def _stage_abandonment(r: StageAbandonmentResult) -> list[str]:
    lines = [
        (
            f"   Highest: {r.highest_abandonment_stage or 'n/a'} at {_pct(r.value)} "
            f"(stage entries with >= {r.min_follow_up_days} days follow-up)"
        )
    ]
    lines += [
        f"     {s.stage_order}. {s.stage:<26} {_pct(s.abandonment_rate):>6}  "
        f"({s.abandoned}/{s.eligible_entries}; {s.entered - s.eligible_entries} too recent)"
        for s in r.stages
    ]
    return lines


def _delay_trend(r: DelayTrendResult) -> list[str]:
    lines = [f"   Overall: {_pct(r.value)} delayed (n={r.sample_size} active journeys)"]
    lines += [
        f"     {p.period_start} cohort {_pct(p.delay_rate):>6}  "
        f"({p.delayed}/{p.eligible} active journeys; {p.disengaged} disengaged; "
        f"{p.entries - p.eligible - p.disengaged} not yet judgeable)"
        for p in r.periods
    ]
    return lines


def _support(r: SupportTicketResult) -> list[str]:
    scope = "technical tickets" if r.categories else "all tickets"
    lines = [
        (
            f"   Delayed customers file {_num(r.value, 'x')} the {scope} per onboarding day "
            "expected at their segment's on-target rate."
        )
    ]
    for row in r.rates:
        label = "delayed" if row.delayed else "on target"
        lines.append(
            f"     {row.segment:<11} {label:<9} "
            f"{_num(row.tickets_per_30_onboarding_days):>5} per 30 days  "
            f"({row.tickets} tickets / {row.onboarding_days:.0f} days, "
            f"{row.customers} customers)"
        )
    return lines


FORMATTERS = {
    "completion": _completion,
    "time_to_onboard": _time_to_onboard,
    "stage_abandonment": _stage_abandonment,
    "delay_trend": _delay_trend,
    "support_tickets": _support,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Relay onboarding analytics demo.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--segment", action="append", dest="segments")
    parser.add_argument("--period-start", type=date.fromisoformat)
    parser.add_argument("--period-end", type=date.fromisoformat)
    parser.add_argument("--json", action="store_true", help="print raw JSON results")
    args = parser.parse_args(argv)

    analytics = OnboardingAnalytics(args.data_dir)
    filters = Filters(
        segments=args.segments,
        period_start=args.period_start,
        period_end=args.period_end,
    )
    results = run_all(analytics, filters)

    if args.json:
        print(
            json.dumps(
                {k: r.model_dump(mode="json") for k, r in results.items()}, indent=2
            )
        )
        return

    print(
        f"Relay Technologies onboarding analytics (data as of {analytics.as_of:%Y-%m-%d})"
    )
    print(f"Filters: {filters.model_dump(exclude_none=True) or 'none'}")
    for number, (key, result) in enumerate(results.items(), start=1):
        lines = _header(number, result) + FORMATTERS[key](result)
        lines += [f"   Note: {note}" for note in result.notes]
        print("\n".join(lines))


if __name__ == "__main__":
    main()
