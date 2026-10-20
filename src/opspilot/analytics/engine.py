"""Deterministic onboarding analytics over the Relay Parquet datasets.

The engine reads only permitted business data: the four Parquet tables and
``dataset_metadata.json`` in the analysis directory. It never imports the data
generator and never reads evaluation data (ground truth).

All definitions follow docs/synthetic_data.md:

- A journey is *completed* when go_live has a stage_completed event,
  *abandoned* when a stage_abandoned event exists, otherwise *in progress*.
- In-progress journeys are right-censored, never counted as abandoned.
- A customer's onboarding window runs from signup_at to go-live (completed),
  to the last activity before abandonment (abandoned), or to as_of (in progress).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from opspilot.analytics.models import (
    CompletionBreakdown,
    CompletionResult,
    DelayPeriod,
    DelayTrendResult,
    Filters,
    ReportingWindow,
    SegmentRateRatio,
    StageAbandonmentResult,
    StageAbandonmentRow,
    SupportRateRow,
    SupportTicketResult,
    TimeToOnboardBreakdown,
    TimeToOnboardResult,
)
from opspilot.analytics.stats import kaplan_meier_median

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"
TABLES = ("customers", "customer_attributes", "onboarding_events", "support_tickets")
METADATA_FILE = "dataset_metadata.json"
FILTER_COLUMNS = {
    "segments": "segment",
    "plan_tiers": "plan_tier",
    "regions": "region",
    "acquisition_channels": "acquisition_channel",
}
GRANULARITY_MONTHS = {"month": 1, "quarter": 3}

_JOURNEYS_SQL = """
CREATE VIEW journeys AS
WITH activity AS (
    SELECT
        customer_id,
        min(event_timestamp) FILTER (
            WHERE stage = 'go_live' AND event_type = 'stage_completed'
        ) AS completed_at,
        min(event_timestamp) FILTER (WHERE event_type = 'stage_abandoned') AS abandoned_at,
        min(stage) FILTER (WHERE event_type = 'stage_abandoned') AS abandoned_stage,
        max(event_timestamp) FILTER (WHERE event_type <> 'stage_abandoned')
            AS last_activity_at
    FROM onboarding_events, snapshot
    WHERE event_timestamp <= snapshot.as_of
    GROUP BY customer_id
),
base AS (
    SELECT
        c.customer_id,
        c.signup_at,
        a.segment,
        a.plan_tier,
        a.region,
        a.acquisition_channel,
        act.completed_at,
        act.abandoned_at,
        act.abandoned_stage,
        act.last_activity_at,
        CASE
            WHEN act.completed_at IS NOT NULL THEN 'completed'
            WHEN act.abandoned_at IS NOT NULL THEN 'abandoned'
            ELSE 'in_progress'
        END AS status,
        t.target_days,
        s.as_of
    FROM customers AS c
    CROSS JOIN snapshot AS s
    LEFT JOIN customer_attributes AS a ON a.customer_id = c.customer_id
    LEFT JOIN activity AS act ON act.customer_id = c.customer_id
    LEFT JOIN segment_targets AS t ON t.segment = a.segment
    WHERE c.signup_at IS NOT NULL AND c.signup_at <= s.as_of
)
SELECT
    *,
    CASE status
        WHEN 'completed' THEN completed_at
        WHEN 'abandoned' THEN coalesce(last_activity_at, signup_at)
        ELSE as_of
    END AS onboarding_end_at,
    date_diff('second', signup_at, onboarding_end_at) / 86400.0 AS onboarding_days
FROM base
"""

_STAGE_ENTRIES_SQL = """
CREATE VIEW stage_entries AS
SELECT * FROM (
    SELECT
        e.customer_id,
        e.stage,
        min(e.stage_order) AS stage_order,
        min(e.event_timestamp) FILTER (WHERE e.event_type = 'stage_started') AS started_at,
        min(e.event_timestamp) FILTER (WHERE e.event_type = 'stage_completed')
            AS completed_at,
        min(e.event_timestamp) FILTER (WHERE e.event_type = 'stage_abandoned')
            AS abandoned_at,
        max(e.event_timestamp) FILTER (WHERE e.event_type <> 'stage_abandoned')
            AS last_activity_at
    FROM onboarding_events AS e, snapshot AS s
    WHERE e.event_timestamp <= s.as_of
    GROUP BY e.customer_id, e.stage
)
WHERE started_at IS NOT NULL
"""


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _permitted_data_dir(data_dir: Path) -> Path:
    data_dir = Path(data_dir).resolve()
    if "evaluation" in data_dir.parts:
        raise PermissionError(
            "Analytics may only read business data, not evaluation data."
        )
    required = [f"{table}.parquet" for table in TABLES] + [METADATA_FILE]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. "
            "Run `uv run python -m opspilot.generate_data` first."
        )
    return data_dir


class OnboardingAnalytics:
    """Onboarding analyses over the Parquet files in ``data_dir``."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = _permitted_data_dir(Path(data_dir))
        metadata = json.loads((self.data_dir / METADATA_FILE).read_text())
        self.as_of: datetime = pd.Timestamp(metadata["as_of"]).to_pydatetime()
        self.stages: list[str] = [
            s["name"] for s in sorted(metadata["stages"], key=lambda s: s["order"])
        ]
        self.target_days: dict[str, int] = metadata["target_onboarding_days"]
        self.inactivity_days: int = metadata["abandonment_inactivity_days"]
        definitions = metadata["definitions"]
        self.ticket_categories: list[str] = list(definitions["ticket_categories"])
        self.technical_categories: list[str] = definitions[
            "technical_ticket_categories"
        ]

        self.con = duckdb.connect()  # in-memory; views read the Parquet files directly
        for table in TABLES:
            path = (self.data_dir / f"{table}.parquet").as_posix().replace("'", "''")
            self.con.execute(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')"
            )
        self.con.execute("CREATE TABLE snapshot (as_of TIMESTAMP)")
        self.con.execute("INSERT INTO snapshot VALUES (?)", [self.as_of])
        self.con.execute(
            "CREATE TABLE segment_targets (segment VARCHAR, target_days INTEGER)"
        )
        self.con.executemany(
            "INSERT INTO segment_targets VALUES (?, ?)", list(self.target_days.items())
        )
        self.con.execute(_JOURNEYS_SQL)
        self.con.execute(_STAGE_ENTRIES_SQL)

    # --- helpers -------------------------------------------------------------------

    def _query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def _where(self, filters: Filters, period_column: str) -> tuple[str, list]:
        """SQL predicate over alias ``j`` (journeys) plus the period column."""
        clauses, params = [], []
        for field, column in FILTER_COLUMNS.items():
            values = getattr(filters, field)
            if values is None:
                continue
            known = set(
                self._query(
                    f"SELECT DISTINCT {column} FROM customer_attributes "
                    f"WHERE {column} IS NOT NULL"
                )[column]
            )
            unknown = sorted(set(values) - known)
            if unknown:
                raise ValueError(
                    f"Unknown {column} value(s) {unknown}; known: {sorted(known)}"
                )
            if not values:
                clauses.append("FALSE")
                continue
            clauses.append(f"j.{column} IN ({', '.join('?' * len(values))})")
            params += values
        if filters.period_start:
            clauses.append(f"{period_column} >= ?")
            params.append(datetime.combine(filters.period_start, datetime.min.time()))
        if filters.period_end:
            clauses.append(f"{period_column} < ?")
            params.append(datetime.combine(filters.period_end, datetime.min.time()))
        return " AND ".join(clauses) or "TRUE", params

    def _window(self, filters: Filters, basis: str) -> ReportingWindow:
        return ReportingWindow(
            basis=basis,
            start=filters.period_start,
            end=filters.period_end,
            as_of=self.as_of,
        )

    # --- 1. Completion -------------------------------------------------------------

    def completion_rate(
        self, filters: Filters | None = None, min_follow_up_days: int = 120
    ) -> CompletionResult:
        """Share of customers who completed onboarding.

        Only customers who signed up at least ``min_follow_up_days`` before as_of
        are eligible, so recent signups don't drag the rate down or get
        mislabelled as failures.
        """
        filters = filters or Filters()
        where, params = self._where(filters, "j.signup_at")
        df = self._query(
            f"""
            SELECT j.segment, j.status,
                   j.signup_at <= j.as_of - to_days(?) AS eligible,
                   count(*) AS n
            FROM journeys AS j
            WHERE {where}
            GROUP BY ALL
            """,
            [min_follow_up_days, *params],
        )
        eligible = df[df["eligible"]]
        counts = eligible.groupby("status")["n"].sum()
        completed, abandoned, in_progress = (
            int(counts.get(s, 0)) for s in ("completed", "abandoned", "in_progress")
        )
        total = completed + abandoned + in_progress

        by_segment = []
        for segment, group in eligible.groupby("segment", dropna=False, sort=True):
            seg_counts = group.groupby("status")["n"].sum()
            seg_total = int(seg_counts.sum())
            by_segment.append(
                CompletionBreakdown(
                    segment=None if pd.isna(segment) else segment,
                    eligible_customers=seg_total,
                    completed=int(seg_counts.get("completed", 0)),
                    abandoned=int(seg_counts.get("abandoned", 0)),
                    in_progress=int(seg_counts.get("in_progress", 0)),
                    completion_rate=_rate(seg_counts.get("completed", 0), seg_total),
                )
            )

        notes = []
        if total == 0:
            notes.append("No eligible customers match the filters.")
        if in_progress:
            notes.append(
                f"{in_progress} eligible customers are still onboarding; the rate is a "
                "lower bound."
            )
        return CompletionResult(
            question="What percentage of customers complete onboarding?",
            metric="onboarding_completion_rate",
            definition=(
                "Customers with a completed go_live stage / eligible customers. "
                f"Eligible: signed up at least {min_follow_up_days} days before as_of."
            ),
            unit="proportion",
            value=_rate(completed, total),
            sample_size=total,
            window=self._window(filters, "signup_at"),
            filters=filters,
            notes=notes,
            min_follow_up_days=min_follow_up_days,
            completed=completed,
            abandoned=abandoned,
            in_progress=in_progress,
            excluded_insufficient_follow_up=int(df.loc[~df["eligible"], "n"].sum()),
            by_segment=by_segment,
        )

    # --- 2. Time to onboard --------------------------------------------------------

    def time_to_onboard(self, filters: Filters | None = None) -> TimeToOnboardResult:
        """Median days from signup to go-live, by segment (Kaplan-Meier).

        In-progress journeys are censored at as_of, abandoned journeys at their
        last activity. The median therefore describes customers who keep
        onboarding. The completed-only median is reported for reference; it is
        biased low because slow journeys have not finished yet.
        """
        filters = filters or Filters()
        where, params = self._where(filters, "j.signup_at")
        df = self._query(
            f"""
            SELECT j.segment, j.status = 'completed' AS completed, j.onboarding_days
            FROM journeys AS j
            WHERE {where}
            """,
            params,
        )

        def summarise(
            group: pd.DataFrame, segment: str | None
        ) -> TimeToOnboardBreakdown:
            done = group.loc[group["completed"], "onboarding_days"]
            return TimeToOnboardBreakdown(
                segment=segment,
                customers=len(group),
                completed=len(done),
                censored=len(group) - len(done),
                median_days=kaplan_meier_median(
                    group["onboarding_days"].tolist(), group["completed"].tolist()
                ),
                completed_only_median_days=float(done.median()) if len(done) else None,
                target_days=self.target_days.get(segment) if segment else None,
            )

        by_segment = [
            summarise(group, None if pd.isna(segment) else segment)
            for segment, group in df.groupby("segment", dropna=False, sort=True)
        ]
        overall = summarise(df, None)

        notes = []
        if df.empty:
            notes.append("No customers match the filters.")
        elif overall.median_days is None:
            notes.append(
                "Fewer than half of the customers have gone live; median not reached."
            )
        notes += [
            f"Median not yet reached for segment {row.segment}."
            for row in by_segment
            if row.customers and row.median_days is None
        ]
        return TimeToOnboardResult(
            question="What is the median time to onboard, by customer segment?",
            metric="median_time_to_go_live",
            definition=(
                "Kaplan-Meier median of days from signup_at to go_live completion. "
                "In-progress journeys censored at as_of, abandoned journeys at last "
                "activity."
            ),
            unit="days",
            value=overall.median_days,
            sample_size=len(df),
            window=self._window(filters, "signup_at"),
            filters=filters,
            notes=notes,
            by_segment=by_segment,
        )

    # --- 3. Stage abandonment ------------------------------------------------------

    def stage_abandonment(
        self, filters: Filters | None = None, min_follow_up_days: int = 60
    ) -> StageAbandonmentResult:
        """Abandonment rate per stage among stage entries with enough follow-up.

        A stage entry is eligible when the stage started at least
        ``min_follow_up_days`` before as_of. The period filter applies to the
        stage start date.
        """
        filters = filters or Filters()
        where, params = self._where(filters, "s.started_at")
        df = self._query(
            f"""
            SELECT
                s.stage,
                count(*) AS entered,
                count(*) FILTER (WHERE eligible) AS eligible_entries,
                count(*) FILTER (WHERE eligible AND s.abandoned_at IS NOT NULL) AS abandoned,
                count(*) FILTER (WHERE eligible AND s.completed_at IS NOT NULL) AS completed
            FROM (
                SELECT s.*, s.started_at <= j.as_of - to_days(?) AS eligible
                FROM stage_entries AS s
                JOIN journeys AS j ON j.customer_id = s.customer_id
                WHERE {where}
            ) AS s
            GROUP BY s.stage
            """,
            [min_follow_up_days, *params],
        ).set_index("stage")

        rows = []
        for order, stage in enumerate(self.stages, start=1):
            r = df.loc[stage] if stage in df.index else None
            entered, eligible, abandoned, completed = (
                (0, 0, 0, 0)
                if r is None
                else (
                    int(r["entered"]),
                    int(r["eligible_entries"]),
                    int(r["abandoned"]),
                    int(r["completed"]),
                )
            )
            rows.append(
                StageAbandonmentRow(
                    stage=stage,
                    stage_order=order,
                    entered=entered,
                    eligible_entries=eligible,
                    abandoned=abandoned,
                    completed=completed,
                    still_in_stage=eligible - abandoned - completed,
                    abandonment_rate=_rate(abandoned, eligible),
                )
            )

        rated = [r for r in rows if r.abandonment_rate is not None]
        worst = max(rated, key=lambda r: r.abandonment_rate, default=None)
        eligible_total = sum(r.eligible_entries for r in rows)
        notes = []
        if eligible_total == 0:
            notes.append("No stage entries with enough follow-up match the filters.")
        if min_follow_up_days < self.inactivity_days:
            notes.append(
                f"Follow-up is shorter than the {self.inactivity_days}-day inactivity "
                "rule, so abandonment is understated."
            )
        return StageAbandonmentResult(
            question="Which onboarding stages experience the most abandonment?",
            metric="stage_abandonment_rate",
            definition=(
                "Stage entries with a stage_abandoned event / eligible stage entries. "
                f"Eligible: stage started at least {min_follow_up_days} days before as_of."
            ),
            unit="proportion",
            value=worst.abandonment_rate if worst else None,
            sample_size=eligible_total,
            window=self._window(filters, "stage started_at"),
            filters=filters,
            notes=notes,
            min_follow_up_days=min_follow_up_days,
            highest_abandonment_stage=worst.stage if worst else None,
            stages=rows,
        )

    # --- 4. Delays over time -------------------------------------------------------

    def delay_trend(
        self,
        filters: Filters | None = None,
        granularity: Literal["month", "quarter"] = "month",
        stage: str | None = None,
        threshold_days: int | None = None,
    ) -> DelayTrendResult:
        """Share of still-active journeys that are delayed, by signup cohort.

        Each journey is judged at its checkpoint: start + threshold. For the whole
        onboarding (``stage=None``) start is signup and the threshold defaults to
        the segment's target time-to-value; for a single stage, start is the
        stage start and ``threshold_days`` is required. At the checkpoint a
        journey is:

        - on time: done (go-live / stage completed) by the checkpoint;
        - delayed: not done, but still active after the checkpoint;
        - disengaged: not done and no activity after the checkpoint. This is
          abandonment, not delay, so it is reported separately.

        A journey is judgeable only once the abandonment inactivity window after
        its checkpoint has passed, so recent cohorts are not judged early.
        Delay rate = delayed / (on time + delayed).
        """
        filters = filters or Filters()
        if granularity not in GRANULARITY_MONTHS:
            raise ValueError(f"granularity must be one of {list(GRANULARITY_MONTHS)}")
        if threshold_days is not None and threshold_days <= 0:
            raise ValueError("threshold_days must be positive")
        where, params = self._where(filters, "j.signup_at")

        if stage is None:
            source = f"""
                SELECT j.signup_at, j.signup_at AS start_at, j.completed_at AS done_at,
                       j.last_activity_at AS last_active_at, j.as_of,
                       j.signup_at + to_days(coalesce(?::INTEGER, j.target_days))
                           AS checkpoint
                FROM journeys AS j
                WHERE {where}
            """
            source_params = [threshold_days, *params]
            subject = "go-live"
        else:
            if stage not in self.stages:
                raise ValueError(f"Unknown stage {stage!r}; known: {self.stages}")
            if threshold_days is None:
                raise ValueError("threshold_days is required when stage is given")
            source = f"""
                SELECT j.signup_at, s.started_at AS start_at, s.completed_at AS done_at,
                       s.last_activity_at AS last_active_at, j.as_of,
                       s.started_at + to_days(?::INTEGER) AS checkpoint
                FROM stage_entries AS s
                JOIN journeys AS j ON j.customer_id = s.customer_id
                WHERE s.stage = ? AND {where}
            """
            source_params = [threshold_days, stage, *params]
            subject = f"{stage} completion"

        df = self._query(
            f"""
            SELECT
                date_trunc('{granularity}', signup_at)::DATE AS period_start,
                count(*) AS entries,
                count(*) FILTER (WHERE judgeable AND NOT on_time AND NOT active) AS disengaged,
                count(*) FILTER (WHERE judgeable AND (on_time OR active)) AS eligible,
                count(*) FILTER (WHERE judgeable AND NOT on_time AND active) AS delayed,
                median(date_diff('second', start_at, done_at) / 86400.0)
                    AS completed_only_median_days
            FROM (
                SELECT *,
                    checkpoint + to_days(?::INTEGER) <= as_of AS judgeable,
                    coalesce(done_at <= checkpoint, FALSE) AS on_time,
                    coalesce(last_active_at > checkpoint, FALSE) AS active
                FROM ({source})
            )
            GROUP BY ALL
            ORDER BY period_start
            """,
            [self.inactivity_days, *source_params],
        )

        months = GRANULARITY_MONTHS[granularity]
        periods = [
            DelayPeriod(
                period_start=r.period_start,
                period_end=(
                    pd.Timestamp(r.period_start) + pd.DateOffset(months=months)
                ).date(),
                entries=int(r.entries),
                eligible=int(r.eligible),
                delayed=int(r.delayed),
                disengaged=int(r.disengaged),
                delay_rate=_rate(r.delayed, r.eligible),
                completed_only_median_days=(
                    None
                    if pd.isna(r.completed_only_median_days)
                    else float(r.completed_only_median_days)
                ),
            )
            for r in df.itertuples()
        ]
        eligible_total = sum(p.eligible for p in periods)
        threshold_text = (
            f"{threshold_days} days" if threshold_days else "the segment's target days"
        )
        notes = []
        if not periods:
            notes.append("No journeys match the filters.")
        incomplete = [
            p.period_start.isoformat()
            for p in periods
            if p.eligible + p.disengaged < p.entries
        ]
        if incomplete:
            notes.append(
                "Cohorts with journeys not yet judgeable (checkpoint plus "
                f"{self.inactivity_days}-day inactivity window after as_of): "
                f"{', '.join(incomplete)}."
            )
        disengaged = sum(p.disengaged for p in periods)
        if disengaged:
            notes.append(
                f"{disengaged} journeys had disengaged by their checkpoint; they count as "
                "abandonment, not delay, and are excluded from the delay rate."
            )
        return DelayTrendResult(
            question="How have onboarding delays changed across time?",
            metric="delay_rate_by_cohort",
            definition=(
                f"Delayed / (on time + delayed). On time: {subject} within "
                f"{threshold_text} of starting (the checkpoint). Delayed: not done by "
                "the checkpoint but still active after it. Judged only once the "
                f"checkpoint plus {self.inactivity_days} days has passed. Cohorts by "
                "signup date."
            ),
            unit="proportion",
            value=_rate(sum(p.delayed for p in periods), eligible_total),
            sample_size=eligible_total,
            window=self._window(filters, "signup_at"),
            filters=filters,
            notes=notes,
            stage=stage,
            granularity=granularity,
            threshold_days=threshold_days,
            periods=periods,
        )

    # --- 5. Support tickets --------------------------------------------------------

    def support_ticket_rates(
        self,
        filters: Filters | None = None,
        categories: list[str] | None = None,
        technical_only: bool = False,
    ) -> SupportTicketResult:
        """Tickets per onboarding day for delayed vs on-target customers.

        Delayed: the onboarding window is longer than the segment target. Only
        tickets created inside each customer's onboarding window count, and
        rates are compared within segment. The headline value is a
        segment-standardised ratio: delayed customers' tickets divided by what
        they would have filed at their segment's on-target rate.
        """
        filters = filters or Filters()
        if categories is not None:
            unknown = sorted(set(categories) - set(self.ticket_categories))
            if unknown:
                raise ValueError(
                    f"Unknown ticket categories {unknown}; known: {self.ticket_categories}"
                )
        if technical_only:
            base = categories if categories is not None else self.technical_categories
            categories = [c for c in base if c in self.technical_categories]

        where, params = self._where(filters, "j.signup_at")
        category_sql, category_params = "TRUE", []
        if categories is not None:
            category_sql = (
                f"t.category IN ({', '.join('?' * len(categories))})"
                if categories
                else "FALSE"
            )
            category_params = categories
        df = self._query(
            f"""
            WITH tickets AS (
                SELECT t.customer_id, count(*) AS n
                FROM support_tickets AS t
                JOIN journeys AS j ON j.customer_id = t.customer_id
                WHERE t.created_at IS NOT NULL
                  AND t.created_at >= j.signup_at
                  AND t.created_at <= j.onboarding_end_at
                  AND {category_sql}
                GROUP BY t.customer_id
            )
            SELECT
                j.segment,
                j.target_days IS NULL AS missing_target,
                coalesce(j.onboarding_days > j.target_days, FALSE) AS delayed,
                count(*) AS customers,
                coalesce(sum(tk.n), 0) AS tickets,
                coalesce(sum(j.onboarding_days), 0) AS onboarding_days
            FROM journeys AS j
            LEFT JOIN tickets AS tk ON tk.customer_id = j.customer_id
            WHERE {where}
            GROUP BY ALL
            ORDER BY j.segment, delayed
            """,
            [*category_params, *params],
        )
        excluded = int(df.loc[df["missing_target"], "customers"].sum())
        df = df[~df["missing_target"]]

        rates = [
            SupportRateRow(
                segment=r.segment,
                delayed=bool(r.delayed),
                customers=int(r.customers),
                tickets=int(r.tickets),
                onboarding_days=float(r.onboarding_days),
                tickets_per_onboarding_day=_rate(r.tickets, r.onboarding_days),
                tickets_per_30_onboarding_days=(
                    None
                    if not r.onboarding_days
                    else 30 * r.tickets / r.onboarding_days
                ),
            )
            for r in df.itertuples()
        ]
        by_segment: dict[str, dict[bool, SupportRateRow]] = defaultdict(dict)
        for row in rates:
            by_segment[row.segment][row.delayed] = row

        ratios, observed, expected = [], 0.0, 0.0
        for segment, groups in sorted(by_segment.items()):
            delayed, on_target = groups.get(True), groups.get(False)
            ratio = None
            if delayed and on_target and on_target.tickets_per_onboarding_day:
                ratio = (
                    delayed.tickets_per_onboarding_day
                    / on_target.tickets_per_onboarding_day
                )
                observed += delayed.tickets
                expected += (
                    delayed.onboarding_days * on_target.tickets_per_onboarding_day
                )
            ratios.append(
                SegmentRateRatio(segment=segment, delayed_to_on_target_ratio=ratio)
            )

        notes = []
        if not rates:
            notes.append("No customers match the filters.")
        if excluded:
            notes.append(
                f"{excluded} customers without a known segment target were excluded."
            )
        missing = [r.segment for r in ratios if r.delayed_to_on_target_ratio is None]
        if missing:
            notes.append(
                "No within-segment comparison possible (a group is empty or has no "
                f"tickets) for: {', '.join(missing)}."
            )
        notes.append("Association only: this does not show that delays cause tickets.")
        return SupportTicketResult(
            question="Do delayed customers generate more support tickets per onboarding day?",
            metric="segment_standardised_ticket_rate_ratio",
            definition=(
                "Tickets created during the onboarding window per onboarding day, delayed "
                "(window longer than segment target) vs on-target customers, within "
                "segment. Headline: delayed tickets / expected at on-target segment rates."
            ),
            unit="ratio",
            value=_rate(observed, expected),
            sample_size=int(sum(r.customers for r in rates)),
            window=self._window(filters, "signup_at"),
            filters=filters,
            notes=notes,
            categories=categories,
            rates=rates,
            segment_ratios=ratios,
        )


def run_all(
    analytics: OnboardingAnalytics, filters: Filters | None = None
) -> dict[str, object]:
    """Run the five standard analyses with default parameters."""
    return {
        "completion": analytics.completion_rate(filters),
        "time_to_onboard": analytics.time_to_onboard(filters),
        "stage_abandonment": analytics.stage_abandonment(filters),
        "delay_trend": analytics.delay_trend(filters, granularity="quarter"),
        "support_tickets": analytics.support_ticket_rates(filters, technical_only=True),
    }
