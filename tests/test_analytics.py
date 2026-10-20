"""Tests for the onboarding analytics engine.

Most tests use a tiny hand-built dataset whose answers are computed by hand
below. The final section checks the engine against the generated synthetic
dataset and its ground truth (tests may read ground truth; the engine may not).
"""

import ast
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from opspilot import generate_data as gd
from opspilot.analytics import Filters, OnboardingAnalytics, run_all
from opspilot.analytics.__main__ import main as cli_main
from opspilot.analytics.stats import kaplan_meier_median

AS_OF = datetime(2026, 1, 1)
SEP_1 = datetime(2025, 9, 1)
D = timedelta(days=1)
STAGES = list(gd.STAGES)

# --- Controlled dataset ----------------------------------------------------------
#
# as_of 2026-01-01. Targets: SMB 21 days, Enterprise 75 days.
# C1 SMB  signup 09-01, five 2-day stages        -> live 09-11 (10 days, on target)
# C2 SMB  signup 09-01, five 10-day stages       -> live 10-21 (50 days, delayed)
# C3 SMB  signup 09-01, provisioning 2 days, starts erp 09-03, then silent
#         -> abandoned (flagged 10-03); last activity 09-03
# C4 Ent  signup 09-01, five 10-day stages       -> live 10-21 (50 days, on target)
# C5 Ent  signup 12-20, provisioning started     -> in progress (recent)
# C6 SMB  signup 12-25, no events yet            -> in progress (recent)


def _journey(
    customer_id: str,
    start: datetime,
    durations: list[float],
    *,
    current_stage_activity: list[float] | None = None,
    abandoned: bool = False,
) -> list[tuple]:
    """Completed stages of the given durations, then optionally an open stage."""
    rows, t = [], start
    for order, days in enumerate(durations, start=1):
        stage = STAGES[order - 1]
        rows.append((customer_id, stage, order, "stage_started", t, "started"))
        t += days * D
        rows.append((customer_id, stage, order, "stage_completed", t, "completed"))
    if current_stage_activity is not None:
        order = len(durations) + 1
        stage = STAGES[order - 1]
        rows.append((customer_id, stage, order, "stage_started", t, "started"))
        last = t
        for offset in current_stage_activity:
            last = t + offset * D
            rows.append((customer_id, stage, order, "stage_progress", last, "on_track"))
        if abandoned:
            rows.append(
                (
                    customer_id,
                    stage,
                    order,
                    "stage_abandoned",
                    last + 30 * D,
                    "abandoned",
                )
            )
    return rows


def base_dataset() -> dict[str, list]:
    customers = [
        ("C1", SEP_1, "active"),
        ("C2", SEP_1, "active"),
        ("C3", SEP_1, "churned"),
        ("C4", SEP_1, "active"),
        ("C5", datetime(2025, 12, 20), "onboarding"),
        ("C6", datetime(2025, 12, 25), "onboarding"),
    ]
    attributes = [
        ("C1", "SMB", "starter", "EMEA", "inbound"),
        ("C2", "SMB", "starter", "North America", "inbound"),
        ("C3", "SMB", "professional", "North America", "self_serve"),
        ("C4", "Enterprise", "enterprise", "North America", "outbound_sales"),
        ("C5", "Enterprise", "enterprise", "EMEA", "partner"),
        ("C6", "SMB", "starter", "APAC", "self_serve"),
    ]
    events = (
        _journey("C1", SEP_1, [2] * 5)
        + _journey("C2", SEP_1, [10] * 5)
        + _journey("C3", SEP_1, [2], current_stage_activity=[], abandoned=True)
        + _journey("C4", SEP_1, [10] * 5)
        + _journey("C5", datetime(2025, 12, 20), [], current_stage_activity=[])
    )
    sep = lambda day: datetime(2025, 9, day)
    tickets = [
        ("T01", "C1", sep(5), "erp_integration", "low", sep(6)),
        ("T02", "C2", sep(10), "erp_integration", "high", sep(11)),
        ("T03", "C2", sep(20), "inventory_sync", "medium", sep(22)),
        ("T04", "C2", sep(30), "fulfillment_rules", "medium", None),
        ("T05", "C2", datetime(2025, 10, 5), "platform_bug", "high", None),
        ("T06", "C2", datetime(2025, 10, 15), "shipping_carrier", "low", None),
        ("T07", "C2", sep(15), "billing", "low", sep(16)),
        (
            "T08",
            "C2",
            datetime(2025, 11, 1),
            "platform_bug",
            "low",
            None,
        ),  # post go-live
        ("T09", "C3", sep(2), "how_to", "low", sep(3)),
        ("T10", "C4", sep(10), "erp_integration", "medium", sep(12)),
        ("T11", "C4", datetime(2025, 10, 1), "how_to", "low", None),
    ]
    return {
        "customers": customers,
        "customer_attributes": attributes,
        "onboarding_events": events,
        "support_tickets": tickets,
    }


def write_dataset(directory: Path, data: dict[str, list]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    columns = {
        "customers": ["customer_id", "signup_at", "account_status"],
        "customer_attributes": [
            "customer_id",
            "segment",
            "plan_tier",
            "region",
            "acquisition_channel",
        ],
        "onboarding_events": [
            "customer_id",
            "stage",
            "stage_order",
            "event_type",
            "event_timestamp",
            "outcome",
        ],
        "support_tickets": [
            "ticket_id",
            "customer_id",
            "created_at",
            "category",
            "severity",
            "resolved_at",
        ],
    }
    for name, cols in columns.items():
        df = pd.DataFrame(data[name], columns=cols)
        if name == "onboarding_events":
            df.insert(0, "event_id", [f"E{i:04d}" for i in range(len(df))])
        df.to_parquet(directory / f"{name}.parquet", index=False)
    metadata = gd.build_metadata(gd.GeneratorConfig())
    metadata["as_of"] = AS_OF.isoformat()
    (directory / "dataset_metadata.json").write_text(json.dumps(metadata))
    return directory


@pytest.fixture
def analytics(tmp_path) -> OnboardingAnalytics:
    return OnboardingAnalytics(write_dataset(tmp_path / "synthetic", base_dataset()))


# --- Kaplan-Meier ------------------------------------------------------------------


def test_kaplan_meier_median():
    assert kaplan_meier_median([], []) is None
    assert kaplan_meier_median([1, 2, 3], [True, True, True]) == 2
    # Censored observations stay in the risk set until they drop out.
    assert kaplan_meier_median([2, 7, 10, 50], [False, False, True, True]) == 10
    assert kaplan_meier_median([5, 6], [True, False]) == 5  # S reaches exactly 0.5
    assert kaplan_meier_median([5, 6, 7], [True, False, False]) is None  # S stays 2/3


# --- 1. Completion -----------------------------------------------------------------


def test_completion_rate(analytics):
    r = analytics.completion_rate()
    assert r.value == pytest.approx(3 / 4)
    assert (r.completed, r.abandoned, r.in_progress) == (3, 1, 0)
    assert r.sample_size == 4
    assert r.excluded_insufficient_follow_up == 2  # C5, C6 are too recent
    seg = {b.segment: b for b in r.by_segment}
    assert seg["SMB"].completion_rate == pytest.approx(2 / 3)
    assert seg["Enterprise"].completion_rate == 1.0
    assert r.unit == "proportion"
    assert r.window.basis == "signup_at" and r.window.as_of == AS_OF


def test_completion_recent_customers_are_not_failures(analytics):
    r = analytics.completion_rate(min_follow_up_days=0)
    # With no follow-up requirement, recent journeys are in progress, not abandoned.
    assert (r.completed, r.abandoned, r.in_progress) == (3, 1, 2)
    assert any("lower bound" in n for n in r.notes)


def test_completion_filters(analytics):
    r = analytics.completion_rate(Filters(segments=["Enterprise"]))
    assert (r.value, r.sample_size) == (1.0, 1)
    assert r.filters.segments == ["Enterprise"]
    r = analytics.completion_rate(
        Filters(regions=["North America"], plan_tiers=["starter"])
    )
    assert (r.value, r.sample_size) == (1.0, 1)  # C2 only


def test_completion_empty_cohort(analytics):
    r = analytics.completion_rate(Filters(period_start=date(2025, 12, 1)))
    assert r.value is None and r.sample_size == 0
    assert r.excluded_insufficient_follow_up == 2
    assert any("No eligible customers" in n for n in r.notes)
    r = analytics.completion_rate(Filters(segments=[]))
    assert r.value is None and r.sample_size == 0


# --- Filters -----------------------------------------------------------------------


def test_filter_validation(analytics):
    with pytest.raises(ValueError, match="Unknown segment"):
        analytics.completion_rate(Filters(segments=["Enterprize"]))
    with pytest.raises(ValidationError):
        Filters(period_start=date(2026, 1, 1), period_end=date(2025, 1, 1))
    with pytest.raises(ValidationError):
        Filters(segment=["SMB"])  # typo'd field name
    with pytest.raises(ValueError, match="Unknown stage"):
        analytics.delay_trend(stage="training", threshold_days=5)
    with pytest.raises(ValueError, match="threshold_days is required"):
        analytics.delay_trend(stage="erp_integration")
    with pytest.raises(ValueError, match="granularity"):
        analytics.delay_trend(granularity="week")
    with pytest.raises(ValueError, match="Unknown ticket categories"):
        analytics.support_ticket_rates(categories=["printer_jam"])


def test_period_is_half_open(analytics):
    sep = Filters(period_start=date(2025, 9, 1), period_end=date(2025, 9, 2))
    assert analytics.completion_rate(sep).sample_size == 4
    before = Filters(period_end=date(2025, 9, 1))
    assert analytics.completion_rate(before).sample_size == 0


# --- 2. Time to onboard -------------------------------------------------------------


def test_time_to_onboard_handles_censoring(analytics):
    r = analytics.time_to_onboard()
    seg = {b.segment: b for b in r.by_segment}
    # SMB: 2 (censored, C3), 7 (censored, C6), 10 (C1), 50 (C2) -> KM median 10.
    assert seg["SMB"].median_days == pytest.approx(10)
    assert seg["SMB"].completed_only_median_days == pytest.approx(30)
    assert (seg["SMB"].completed, seg["SMB"].censored) == (2, 2)
    # Enterprise: 12 (censored, C5), 50 (C4) -> 50.
    assert seg["Enterprise"].median_days == pytest.approx(50)
    assert seg["Enterprise"].target_days == 75
    assert r.unit == "days" and r.sample_size == 6


def test_time_to_onboard_median_not_reached(analytics):
    r = analytics.time_to_onboard(Filters(period_start=date(2025, 12, 1)))
    assert r.value is None and r.sample_size == 2
    assert any("median not reached" in n for n in r.notes)


# --- 3. Stage abandonment -----------------------------------------------------------


def test_stage_abandonment_uses_eligible_entries(analytics):
    r = analytics.stage_abandonment()
    rows = {s.stage: s for s in r.stages}
    assert [s.stage for s in r.stages] == STAGES  # all stages, in order
    provisioning = rows["warehouse_provisioning"]
    assert (provisioning.entered, provisioning.eligible_entries) == (
        5,
        4,
    )  # C5 too recent
    assert provisioning.abandonment_rate == 0
    erp = rows["erp_integration"]
    assert (erp.eligible_entries, erp.abandoned, erp.completed) == (4, 1, 3)
    assert erp.abandonment_rate == pytest.approx(0.25)
    assert rows["fulfillment_configuration"].eligible_entries == 3
    assert r.highest_abandonment_stage == "erp_integration"
    assert r.value == pytest.approx(0.25)


def test_stage_abandonment_empty_and_short_follow_up(analytics):
    r = analytics.stage_abandonment(Filters(period_start=date(2025, 12, 1)))
    assert r.value is None and r.highest_abandonment_stage is None
    assert r.stages[0].entered == 1 and r.stages[0].eligible_entries == 0
    assert all(s.abandonment_rate is None for s in r.stages)
    r = analytics.stage_abandonment(min_follow_up_days=10)
    assert any("understated" in n for n in r.notes)


# --- 4. Delays over time ------------------------------------------------------------


def test_delay_trend_whole_onboarding(analytics):
    r = analytics.delay_trend(granularity="month")
    sep, dec = r.periods
    # C1 on time, C2 delayed (still active after day 21), C3 disengaged before the
    # checkpoint (abandonment, not delay), C4 on time vs its 75-day target.
    assert (sep.entries, sep.eligible, sep.delayed, sep.disengaged) == (4, 3, 1, 1)
    assert sep.delay_rate == pytest.approx(1 / 3)
    assert (sep.period_start, sep.period_end) == (date(2025, 9, 1), date(2025, 10, 1))
    # December signups cannot be judged yet.
    assert (dec.entries, dec.eligible, dec.delay_rate) == (2, 0, None)
    assert r.value == pytest.approx(1 / 3) and r.sample_size == 3


def test_delay_trend_single_stage(analytics):
    r = analytics.delay_trend(stage="erp_integration", threshold_days=5)
    (sep,) = r.periods
    # C1 2 days (on time), C2 and C4 10 days (delayed), C3 silent (disengaged).
    assert (sep.eligible, sep.delayed, sep.disengaged) == (3, 2, 1)
    assert sep.completed_only_median_days == pytest.approx(10)


def test_delay_trend_waits_for_inactivity_window(tmp_path):
    data = base_dataset()
    # C7 SMB signs up 2025-11-20: checkpoint 12-11, but 12-11 + 30 days > as_of.
    data["customers"].append(("C7", datetime(2025, 11, 20), "onboarding"))
    data["customer_attributes"].append(("C7", "SMB", "starter", "EMEA", "inbound"))
    data["onboarding_events"] += _journey("C7", datetime(2025, 11, 20), [])
    analytics = OnboardingAnalytics(write_dataset(tmp_path / "synthetic", data))
    nov = next(p for p in analytics.delay_trend().periods if p.period_start.month == 11)
    assert (nov.entries, nov.eligible, nov.disengaged) == (1, 0, 0)


def test_delay_trend_quarterly_and_empty(analytics):
    r = analytics.delay_trend(granularity="quarter")
    assert [p.period_start for p in r.periods] == [date(2025, 7, 1), date(2025, 10, 1)]
    assert r.periods[0].period_end == date(2025, 10, 1)
    r = analytics.delay_trend(Filters(segments=["SMB"], period_start=date(2026, 6, 1)))
    assert r.periods == [] and r.value is None and r.sample_size == 0


# --- 5. Support tickets -------------------------------------------------------------


def test_support_rates_normalise_by_onboarding_days(analytics):
    r = analytics.support_ticket_rates()
    rows = {(x.segment, x.delayed): x for x in r.rates}
    delayed = rows[
        "SMB", True
    ]  # C2: 6 tickets in 50 days; post-go-live ticket excluded
    assert (delayed.customers, delayed.tickets) == (1, 6)
    assert delayed.onboarding_days == pytest.approx(50)
    assert delayed.tickets_per_onboarding_day == pytest.approx(6 / 50)
    on_target = rows[
        "SMB", False
    ]  # C1 10 days, C3 2 days (to last activity), C6 7 days
    assert (on_target.customers, on_target.tickets) == (3, 2)
    assert on_target.onboarding_days == pytest.approx(19)
    assert on_target.tickets_per_30_onboarding_days == pytest.approx(30 * 2 / 19)
    ratios = {x.segment: x.delayed_to_on_target_ratio for x in r.segment_ratios}
    assert ratios["SMB"] == pytest.approx((6 / 50) / (2 / 19))
    assert ratios["Enterprise"] is None  # no delayed Enterprise customers
    assert r.value == pytest.approx(6 / (50 * 2 / 19))
    assert any("Enterprise" in n for n in r.notes)
    assert r.unit == "ratio"


def test_support_rates_category_filters(analytics):
    r = analytics.support_ticket_rates(technical_only=True)
    assert r.categories == list(gd.TECHNICAL_TICKET_CATEGORIES)
    assert r.value == pytest.approx(5 / (50 * 1 / 19))
    r = analytics.support_ticket_rates(categories=["billing"])
    # No on-target SMB billing tickets: the ratio is undefined, not infinite.
    assert r.value is None
    r = analytics.support_ticket_rates(categories=["billing"], technical_only=True)
    assert r.categories == [] and r.value is None


# --- Nulls and observation window ---------------------------------------------------


def test_nulls_and_events_after_as_of(tmp_path):
    data = base_dataset()
    # C7 has no attributes row; C5 has an event after as_of; a ticket lacks created_at.
    data["customers"].append(("C7", SEP_1, "active"))
    data["onboarding_events"] += _journey("C7", SEP_1, [2] * 5)
    data["onboarding_events"].append(
        ("C5", "go_live", 5, "stage_completed", datetime(2026, 2, 1), "completed")
    )
    data["support_tickets"].append(("T99", "C1", None, "how_to", "low", None))
    analytics = OnboardingAnalytics(write_dataset(tmp_path / "synthetic", data))

    completion = analytics.completion_rate(min_follow_up_days=0)
    assert completion.in_progress == 2  # C5's future go-live is not observable
    unknown = next(b for b in completion.by_segment if b.segment is None)
    assert unknown.completed == 1

    support = analytics.support_ticket_rates()
    assert any("without a known segment" in n for n in support.notes)
    assert sum(x.tickets for x in support.rates) == 11 - 1  # unchanged, minus T08

    time = analytics.time_to_onboard()
    assert any(b.segment is None for b in time.by_segment)


# --- Data access guardrails ---------------------------------------------------------


def test_refuses_evaluation_directory(tmp_path):
    with pytest.raises(PermissionError):
        OnboardingAnalytics(write_dataset(tmp_path / "evaluation", base_dataset()))


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_data"):
        OnboardingAnalytics(tmp_path)


def test_never_reads_ground_truth(tmp_path):
    directory = write_dataset(tmp_path / "synthetic", base_dataset())
    trap = directory / "ground_truth.json"
    trap.write_text("{}")
    trap.chmod(0)  # any attempt to read it would fail
    try:
        results = run_all(OnboardingAnalytics(directory))
    finally:
        trap.chmod(0o644)
    assert len(results) == 5


def test_analytics_does_not_import_generator():
    package = Path(gd.__file__).parent / "analytics"
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text())
        imported = [
            node.module if isinstance(node, ast.ImportFrom) else alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in node.names
        ]
        assert not any("generate_data" in (m or "") for m in imported), source.name
        assert "ground_truth" not in source.read_text(), source.name


# --- CLI ----------------------------------------------------------------------------


def test_cli_text_and_json(tmp_path, capsys):
    directory = write_dataset(tmp_path / "synthetic", base_dataset())
    cli_main(["--data-dir", str(directory)])
    text = capsys.readouterr().out
    assert "1. What percentage of customers complete onboarding?" in text
    assert "5. Do delayed customers" in text
    cli_main(["--data-dir", str(directory), "--json", "--segment", "SMB"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "completion",
        "time_to_onboard",
        "stage_abandonment",
        "delay_trend",
        "support_tickets",
    }
    assert payload["completion"]["filters"]["segments"] == ["SMB"]


# --- Consistency with the generated dataset's known patterns ------------------------


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("generated")
    dataset = gd.generate_dataset()
    gd.write_dataset(
        dataset, root / "synthetic", root / "evaluation" / "ground_truth.json"
    )
    ground_truth = json.loads((root / "evaluation" / "ground_truth.json").read_text())
    problems = {p["id"]: p for p in ground_truth["problems"]}
    return OnboardingAnalytics(root / "synthetic"), dataset.tables, problems


def test_statuses_match_account_status(generated):
    analytics, tables, _ = generated
    counts = analytics._query("SELECT status, count(*) AS n FROM journeys GROUP BY 1")
    engine = dict(zip(counts["status"], counts["n"]))
    expected = tables["customers"]["account_status"].value_counts()
    assert engine["completed"] == expected["active"]
    assert engine["abandoned"] == expected["churned"]
    assert engine["in_progress"] == expected["onboarding"]


def test_detects_enterprise_erp_delay(generated):
    analytics, _, problems = generated
    problem = problems["enterprise_erp_integration_delay"]
    change = date.fromisoformat(problem["time_window"]["start"][:10])

    def delay_rates(segment: str) -> tuple[float, float]:
        trend = analytics.delay_trend(
            Filters(segments=[segment]),
            granularity="quarter",
            stage="erp_integration",
            threshold_days=21,
        )
        before = [p for p in trend.periods if p.period_start < change]
        after = [p for p in trend.periods if p.period_start >= change]
        rate = lambda ps: sum(p.delayed for p in ps) / sum(p.eligible for p in ps)
        return rate(before), rate(after)

    before, after = delay_rates(problem["affected_segment"])
    assert after - before > 0.2
    for segment in ("SMB", "Mid-Market"):
        before, after = delay_rates(segment)
        assert after - before < 0.1, segment


def test_detects_configuration_abandonment(generated):
    analytics, _, problems = generated
    problem = problems["fulfillment_configuration_abandonment"]
    change = date.fromisoformat(problem["time_window"]["start"][:10])
    after = analytics.stage_abandonment(Filters(period_start=change))
    before = analytics.stage_abandonment(Filters(period_end=change))
    assert after.highest_abandonment_stage == problem["affected_stage"]
    rate = lambda r: next(
        s.abandonment_rate for s in r.stages if s.stage == problem["affected_stage"]
    )
    assert rate(after) > 2.5 * rate(before)


def test_detects_support_load_from_prolonged_onboarding(generated):
    analytics, _, _ = generated
    r = analytics.support_ticket_rates(technical_only=True)
    assert r.value > 1.5
    ratios = {x.segment: x.delayed_to_on_target_ratio for x in r.segment_ratios}
    assert ratios["Enterprise"] > 1.2 and ratios["Mid-Market"] > 1.2


def test_time_to_onboard_ordering_and_completion(generated):
    analytics, _, _ = generated
    medians = {b.segment: b.median_days for b in analytics.time_to_onboard().by_segment}
    assert medians["SMB"] < medians["Mid-Market"] < medians["Enterprise"]
    completion = analytics.completion_rate()
    assert 0.6 < completion.value < 0.95
    assert completion.excluded_insufficient_follow_up > 0
