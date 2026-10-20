"""Typed inputs and outputs for the onboarding analytics engine."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, model_validator


class Filters(BaseModel):
    """Customer filters. ``None`` means no filter on that field.

    The period is half-open: ``period_start <= ts < period_end``. Each analysis
    states which timestamp the period applies to (``ReportingWindow.basis``).
    """

    model_config = ConfigDict(extra="forbid")

    segments: list[str] | None = None
    plan_tiers: list[str] | None = None
    regions: list[str] | None = None
    acquisition_channels: list[str] | None = None
    period_start: date | None = None
    period_end: date | None = None

    @model_validator(mode="after")
    def _check_period(self) -> Filters:
        if (
            self.period_start
            and self.period_end
            and self.period_end <= self.period_start
        ):
            raise ValueError("period_end must be after period_start")
        return self


class ReportingWindow(BaseModel):
    basis: str  # the timestamp the period filter and cohorts are based on
    start: date | None  # inclusive; None = unbounded
    end: date | None  # exclusive; None = unbounded
    as_of: datetime  # data snapshot time; nothing later is observable


class AnalysisResult(BaseModel):
    question: str
    metric: str
    definition: str
    unit: str
    value: float | None  # None when the sample is empty or the metric is undefined
    sample_size: int
    window: ReportingWindow
    filters: Filters
    notes: list[str] = []


# --- 1. Completion -------------------------------------------------------------


class CompletionBreakdown(BaseModel):
    segment: str | None
    eligible_customers: int
    completed: int
    abandoned: int
    in_progress: int
    completion_rate: float | None


class CompletionResult(AnalysisResult):
    min_follow_up_days: int
    completed: int
    abandoned: int
    in_progress: int
    excluded_insufficient_follow_up: int
    by_segment: list[CompletionBreakdown]


# --- 2. Time to onboard ----------------------------------------------------------


class TimeToOnboardBreakdown(BaseModel):
    segment: str | None
    customers: int
    completed: int
    censored: int
    median_days: float | None  # Kaplan-Meier; None if not yet reached
    completed_only_median_days: float | None  # biased low; for reference
    target_days: int | None


class TimeToOnboardResult(AnalysisResult):
    by_segment: list[TimeToOnboardBreakdown]


# --- 3. Stage abandonment --------------------------------------------------------


class StageAbandonmentRow(BaseModel):
    stage: str
    stage_order: int
    entered: int
    eligible_entries: int
    abandoned: int
    completed: int
    still_in_stage: int
    abandonment_rate: float | None


class StageAbandonmentResult(AnalysisResult):
    min_follow_up_days: int
    highest_abandonment_stage: str | None
    stages: list[StageAbandonmentRow]


# --- 4. Delays over time ---------------------------------------------------------


class DelayPeriod(BaseModel):
    period_start: date
    period_end: date
    entries: int  # journeys in the cohort
    eligible: int  # judgeable and on time or delayed (the delay-rate denominator)
    delayed: int
    disengaged: int  # judgeable, not done and inactive after the checkpoint
    delay_rate: float | None
    completed_only_median_days: float | None


class DelayTrendResult(AnalysisResult):
    stage: str | None
    granularity: str
    threshold_days: int | None  # None = each segment's target time-to-value
    periods: list[DelayPeriod]


# --- 5. Support tickets ----------------------------------------------------------


class SupportRateRow(BaseModel):
    segment: str
    delayed: bool
    customers: int
    tickets: int
    onboarding_days: float
    tickets_per_onboarding_day: float | None
    tickets_per_30_onboarding_days: float | None


class SegmentRateRatio(BaseModel):
    segment: str
    delayed_to_on_target_ratio: float | None


class SupportTicketResult(AnalysisResult):
    categories: list[str] | None  # None = all categories
    rates: list[SupportRateRow]
    segment_ratios: list[SegmentRateRatio]
