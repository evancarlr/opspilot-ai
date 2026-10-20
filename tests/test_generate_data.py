"""Validation tests for the synthetic onboarding dataset.

Pattern tests measure the planted problems from the tables alone, the way an
analyst would, without reading generator internals or the ground truth.
"""

import json

import pandas as pd
import pytest

from opspilot import generate_data as gd

DAY = pd.Timedelta(days=1)


@pytest.fixture(scope="module")
def dataset() -> gd.SyntheticDataset:
    return gd.generate_dataset()


@pytest.fixture(scope="module")
def tables(dataset) -> dict[str, pd.DataFrame]:
    return dataset.tables


@pytest.fixture(scope="module")
def customers(tables) -> pd.DataFrame:
    return tables["customers"].merge(tables["customer_attributes"], on="customer_id")


@pytest.fixture(scope="module")
def events(tables, customers) -> pd.DataFrame:
    return tables["onboarding_events"].merge(
        customers[["customer_id", "signup_at", "segment"]], on="customer_id"
    )


def _stage_events(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    rows = events[events["event_type"] == event_type]
    return rows[["customer_id", "stage", "event_timestamp"]]


@pytest.fixture(scope="module")
def onboarding_exposure(customers, events) -> pd.DataFrame:
    """Observed onboarding window per customer: signup -> end of onboarding."""
    df = customers.set_index("customer_id")
    go_live = _stage_events(events, "stage_completed").query("stage == 'go_live'")
    last_activity = (
        events[events["event_type"] != "stage_abandoned"]
        .groupby("customer_id")["event_timestamp"]
        .max()
    )
    end = pd.Series(gd.AS_OF, index=df.index)
    churned = df["account_status"] == "churned"
    active = df["account_status"] == "active"
    end[churned] = last_activity.reindex(df.index)[churned]
    end[active] = go_live.set_index("customer_id")["event_timestamp"].reindex(df.index)[
        active
    ]
    df["onboarding_end"] = end
    df["onboarding_days"] = (end - df["signup_at"]) / DAY
    df["target_days"] = df["segment"].map(gd.TARGET_ONBOARDING_DAYS)
    df["prolonged"] = df["onboarding_days"] > df["target_days"]
    return df


# --- Integrity ----------------------------------------------------------------


def test_row_counts(tables):
    assert len(tables["customers"]) == 1_000
    assert len(tables["customer_attributes"]) == 1_000
    assert len(tables["onboarding_events"]) > 5_000
    assert len(tables["support_tickets"]) > 1_000


def test_unique_identifiers(tables):
    assert tables["customers"]["customer_id"].is_unique
    assert tables["customer_attributes"]["customer_id"].is_unique
    assert tables["onboarding_events"]["event_id"].is_unique
    assert tables["support_tickets"]["ticket_id"].is_unique


def test_no_missing_values_except_open_tickets(tables):
    for name, df in tables.items():
        cols = [c for c in df.columns if c != "resolved_at"]
        assert not df[cols].isna().any().any(), name


def test_foreign_keys(tables):
    customer_ids = set(tables["customers"]["customer_id"])
    assert set(tables["customer_attributes"]["customer_id"]) == customer_ids
    assert set(tables["onboarding_events"]["customer_id"]) <= customer_ids
    assert set(tables["support_tickets"]["customer_id"]) <= customer_ids


def test_signups_within_period(tables):
    signup = tables["customers"]["signup_at"]
    assert signup.min() >= gd.PERIOD_START
    assert signup.max() < gd.AS_OF
    assert signup.dt.to_period("M").nunique() == 12


def test_technical_categories_are_valid_ticket_categories(dataset):
    assert set(gd.TECHNICAL_TICKET_CATEGORIES) < set(gd.TICKET_CATEGORIES)
    metadata = dataset.metadata["definitions"]
    assert set(metadata["ticket_categories"]) == set(gd.TICKET_CATEGORIES)
    assert metadata["technical_ticket_categories"] == list(
        gd.TECHNICAL_TICKET_CATEGORIES
    )


def test_categorical_values(tables):
    allowed = {
        ("customers", "account_status"): gd.ACCOUNT_STATUSES,
        ("customer_attributes", "segment"): gd.SEGMENTS,
        ("customer_attributes", "plan_tier"): gd.PLAN_TIERS,
        ("customer_attributes", "region"): gd.REGIONS,
        ("customer_attributes", "acquisition_channel"): gd.ACQUISITION_CHANNELS,
        ("onboarding_events", "stage"): gd.STAGES,
        ("onboarding_events", "event_type"): gd.EVENT_TYPES,
        ("onboarding_events", "outcome"): gd.EVENT_OUTCOMES,
        ("support_tickets", "category"): gd.TICKET_CATEGORIES,
        ("support_tickets", "severity"): gd.TICKET_SEVERITIES,
    }
    for (table, column), values in allowed.items():
        assert set(tables[table][column]) <= set(values), (table, column)


def test_event_type_outcome_and_stage_order_agree(tables):
    ev = tables["onboarding_events"]
    valid_outcomes = {
        "stage_started": {"started"},
        "stage_progress": {"on_track", "blocked"},
        "stage_completed": {"completed"},
        "stage_abandoned": {"abandoned"},
    }
    for event_type, group in ev.groupby("event_type"):
        assert set(group["outcome"]) <= valid_outcomes[event_type]
    expected_order = ev["stage"].map({s: i for i, s in enumerate(gd.STAGES, 1)})
    assert (ev["stage_order"] == expected_order).all()


# --- Chronology and journey logic ----------------------------------------------


def test_events_within_signup_and_snapshot(events):
    assert (events["event_timestamp"] >= events["signup_at"]).all()
    assert (events["event_timestamp"] <= gd.AS_OF).all()


def test_events_chronological_per_customer(tables):
    ev = tables["onboarding_events"].sort_values("event_id")
    for _, journey in ev.groupby("customer_id"):
        assert journey["event_timestamp"].is_monotonic_increasing
        assert journey["stage_order"].is_monotonic_increasing
        # Stages are entered in order, starting with the first stage.
        stages_seen = journey["stage_order"].unique()
        assert list(stages_seen) == list(range(1, len(stages_seen) + 1))
        for _, stage in journey.groupby("stage_order"):
            types = list(stage["event_type"])
            assert types[0] == "stage_started"
            assert all(t == "stage_progress" for t in types[1:-1])
            if len(types) > 1:
                assert types[-1] in {
                    "stage_progress",
                    "stage_completed",
                    "stage_abandoned",
                }
        # A later stage only starts once the previous one is completed.
        completed = set(
            journey.loc[journey["event_type"] == "stage_completed", "stage_order"]
        )
        assert set(stages_seen[1:] - 1) <= completed


def test_terminal_events_and_inactivity_rule(tables):
    ev = tables["onboarding_events"].sort_values("event_id")
    terminal = ev["event_type"].eq("stage_abandoned") | (
        ev["event_type"].eq("stage_completed") & ev["stage"].eq("go_live")
    )
    assert terminal.groupby(ev["customer_id"]).sum().max() == 1
    # Nothing happens after a terminal event.
    last_per_customer = ev.groupby("customer_id").tail(1)
    assert set(ev[terminal]["event_id"]) <= set(last_per_customer["event_id"])

    gap = ev.groupby("customer_id")["event_timestamp"].diff()
    abandoned = ev["event_type"] == "stage_abandoned"
    inactivity = pd.Timedelta(days=gd.ABANDONMENT_INACTIVITY_DAYS)
    # Abandonment is recorded exactly after the inactivity window...
    assert (gap[abandoned] == inactivity).all()
    # ...and active journeys never go silent for that long.
    assert (gap[~abandoned].dropna() < inactivity).all()


def test_account_status_matches_events(customers, events):
    completed = set(
        events.query("stage == 'go_live' and event_type == 'stage_completed'")[
            "customer_id"
        ]
    )
    abandoned = set(events.query("event_type == 'stage_abandoned'")["customer_id"])
    status = customers.set_index("customer_id")["account_status"]
    assert set(status[status == "active"].index) == completed
    assert set(status[status == "churned"].index) == abandoned


def test_journey_mix_and_right_censoring(dataset, customers, events):
    counts = customers["account_status"].value_counts()
    for status in gd.ACCOUNT_STATUSES:
        assert counts[status] >= 50, status

    # In-progress journeys have recent activity (or just signed up), so the
    # inactivity rule has not yet classified them as abandoned.
    in_progress = customers[customers["account_status"] == "onboarding"]
    last_event = events.groupby("customer_id")["event_timestamp"].max()
    last_seen = last_event.reindex(in_progress["customer_id"]).fillna(
        in_progress.set_index("customer_id")["signup_at"]
    )
    inactivity = pd.Timedelta(days=gd.ABANDONMENT_INACTIVITY_DAYS)
    assert (gd.AS_OF - last_seen < inactivity).all()

    # Censoring matters: some in-progress journeys are latent abandonments.
    censoring = dataset.ground_truth["censoring"]
    assert censoring["in_progress_at_as_of"] == len(in_progress)
    assert censoring["in_progress_that_would_later_abandon"] > 0


def test_ticket_chronology(tables, customers):
    tickets = tables["support_tickets"].merge(
        customers[["customer_id", "signup_at"]], on="customer_id"
    )
    assert (tickets["created_at"] >= tickets["signup_at"]).all()
    assert (tickets["created_at"] <= gd.AS_OF).all()
    resolved = tickets.dropna(subset=["resolved_at"])
    assert (resolved["resolved_at"] > resolved["created_at"]).all()
    assert (resolved["resolved_at"] <= gd.AS_OF).all()
    assert tickets["resolved_at"].isna().sum() > 0  # some still open


# --- Reproducibility and outputs ----------------------------------------------


def test_reproducible_with_same_seed(tables):
    again = gd.generate_dataset(gd.GeneratorConfig(seed=42))
    for name, df in tables.items():
        pd.testing.assert_frame_equal(df, again.tables[name])


def test_different_seed_changes_data(tables):
    other = gd.generate_dataset(gd.GeneratorConfig(seed=7))
    assert not tables["customers"].equals(other.tables["customers"])


def test_write_dataset_keeps_ground_truth_separate(dataset, tmp_path):
    analysis_dir = tmp_path / "synthetic"
    ground_truth_path = tmp_path / "evaluation" / "ground_truth.json"
    gd.write_dataset(dataset, analysis_dir, ground_truth_path)

    written = {p.name for p in analysis_dir.iterdir()}
    assert written == {
        "customers.parquet",
        "customer_attributes.parquet",
        "onboarding_events.parquet",
        "support_tickets.parquet",
        "dataset_metadata.json",
    }
    metadata = (analysis_dir / "dataset_metadata.json").read_text()
    assert "problems" not in json.loads(metadata)
    assert "ground_truth" not in metadata

    ground_truth = json.loads(ground_truth_path.read_text())
    assert [p["id"] for p in ground_truth["problems"]] == [
        "enterprise_erp_integration_delay",
        "fulfillment_configuration_abandonment",
        "prolonged_onboarding_technical_support",
    ]
    reloaded = pd.read_parquet(analysis_dir / "onboarding_events.parquet")
    pd.testing.assert_frame_equal(
        reloaded, dataset.tables["onboarding_events"], check_dtype=False
    )

    with pytest.raises(ValueError):
        gd.write_dataset(dataset, analysis_dir, analysis_dir / "ground_truth.json")


# --- Planted problem 1: Enterprise delays in the second half --------------------


@pytest.fixture(scope="module")
def erp_integration(events) -> pd.DataFrame:
    """erp_integration stages with at least 21 days of follow-up."""
    starts = _stage_events(events, "stage_started").query("stage == 'erp_integration'")
    ends = (
        _stage_events(events, "stage_completed")
        .query("stage == 'erp_integration'")
        .set_index("customer_id")["event_timestamp"]
    )
    abandoned = events.query(
        "stage == 'erp_integration' and event_type == 'stage_abandoned'"
    )["customer_id"]
    df = starts[~starts["customer_id"].isin(abandoned)].merge(
        events.drop_duplicates("customer_id")[["customer_id", "signup_at", "segment"]]
    )
    df = df[df["event_timestamp"] <= gd.AS_OF - 21 * DAY].copy()
    df["completed_at"] = df["customer_id"].map(ends)
    df["days"] = (df["completed_at"] - df["event_timestamp"]) / DAY
    df["over_21_days"] = df["completed_at"].isna() | (df["days"] > 21)
    df["second_half"] = df["signup_at"] >= pd.Timestamp("2026-04-01")
    return df


def test_enterprise_delays_increase_in_second_half(erp_integration):
    share = erp_integration.groupby(["segment", "second_half"])["over_21_days"].mean()
    assert share["Enterprise", True] - share["Enterprise", False] > 0.2
    assert share["Enterprise", True] > 1.6 * share["Enterprise", False]

    median = erp_integration.groupby(["segment", "second_half"])["days"].median()
    assert median["Enterprise", True] > 1.3 * median["Enterprise", False]

    # Other segments show no comparable change.
    for segment in ("SMB", "Mid-Market"):
        assert share[segment, True] - share[segment, False] < 0.1
        assert median[segment, True] < 1.3 * median[segment, False]


def test_enterprise_delay_is_not_universal(erp_integration):
    h1 = erp_integration.query("segment == 'Enterprise' and not second_half")
    h2 = erp_integration.query("segment == 'Enterprise' and second_half")
    assert (h2["days"] < h1["days"].median()).sum() >= 5


# --- Planted problem 2: abandonment rises at fulfillment configuration --------


@pytest.fixture(scope="module")
def stage_abandonment(events) -> pd.DataFrame:
    """Stage entries with at least 60 days of follow-up, flagged if abandoned."""
    starts = _stage_events(events, "stage_started")
    starts = starts[starts["event_timestamp"] <= gd.AS_OF - 60 * DAY]
    abandoned = _stage_events(events, "stage_abandoned")[["customer_id", "stage"]]
    df = starts.merge(abandoned.assign(abandoned=True), how="left")
    df["abandoned"] = df["abandoned"].fillna(False).astype(bool)
    df["after_change"] = df["event_timestamp"] >= pd.Timestamp("2026-06-01")
    return df


def test_configuration_abandonment_rises(stage_abandonment):
    rate = stage_abandonment.groupby(["stage", "after_change"])["abandoned"].mean()
    before, after = (
        rate["fulfillment_configuration", False],
        rate["fulfillment_configuration", True],
    )
    assert after > 0.12
    assert after > 2.5 * before

    # The rise is specific to fulfillment configuration.
    for stage in ("warehouse_provisioning", "erp_integration", "testing_validation"):
        assert rate[stage, True] - rate[stage, False] < 0.05, stage


def test_configuration_abandonment_is_not_universal(stage_abandonment):
    after = stage_abandonment.query(
        "stage == 'fulfillment_configuration' and after_change"
    )
    assert (~after["abandoned"]).mean() > 0.6


# --- Planted problem 3: prolonged onboarding drives technical tickets ----------


@pytest.fixture(scope="module")
def onboarding_tickets(tables, onboarding_exposure) -> pd.DataFrame:
    """Technical tickets created during each customer's onboarding window."""
    exposure = onboarding_exposure[["signup_at", "onboarding_end", "target_days"]]
    tickets = tables["support_tickets"]
    tickets = tickets[tickets["category"].isin(gd.TECHNICAL_TICKET_CATEGORIES)]
    tickets = tickets.merge(exposure, left_on="customer_id", right_index=True)
    in_window = (tickets["created_at"] >= tickets["signup_at"]) & (
        tickets["created_at"] <= tickets["onboarding_end"]
    )
    return tickets[in_window]


def test_prolonged_onboarding_has_higher_ticket_rate(
    onboarding_tickets, onboarding_exposure
):
    df = onboarding_exposure.copy()
    df["tickets"] = onboarding_tickets.groupby("customer_id").size()
    df["tickets"] = df["tickets"].reindex(df.index).fillna(0)
    by_group = df.groupby(["segment", "prolonged"])[
        ["tickets", "onboarding_days"]
    ].sum()
    rate = by_group["tickets"] / by_group["onboarding_days"]

    # SMB files too few tickets for a reliable per-segment comparison.
    for segment in ("Mid-Market", "Enterprise"):
        assert rate[segment, True] > 1.2 * rate[segment, False], segment

    # Segment-standardised ratio: observed vs expected at on-target rates.
    prolonged = by_group.xs(True, level="prolonged")
    on_target_rate = rate.xs(False, level="prolonged")
    expected = (prolonged["onboarding_days"] * on_target_rate).sum()
    assert prolonged["tickets"].sum() / expected > 1.5


def test_ticket_rate_jumps_after_target_date(onboarding_tickets, onboarding_exposure):
    prolonged = onboarding_exposure[onboarding_exposure["prolonged"]]
    tickets = onboarding_tickets[
        onboarding_tickets["customer_id"].isin(prolonged.index)
    ]
    days_in = (tickets["created_at"] - tickets["signup_at"]) / DAY
    overdue_tickets = (days_in > tickets["target_days"]).sum()
    overdue_days = (prolonged["onboarding_days"] - prolonged["target_days"]).sum()
    on_target_days = prolonged["target_days"].sum()
    before = (len(tickets) - overdue_tickets) / on_target_days
    after = overdue_tickets / overdue_days
    assert after > 2.0 * before


def test_support_pattern_is_not_universal(onboarding_tickets, onboarding_exposure):
    prolonged = onboarding_exposure[onboarding_exposure["prolonged"]]
    with_tickets = prolonged.index.isin(onboarding_tickets["customer_id"])
    assert (~with_tickets).sum() >= 5
