"""Generate a reproducible synthetic onboarding dataset for Relay Technologies.

Relay sells RelayOS, a cloud warehouse fulfillment operations platform, to
e-commerce retailers and third-party logistics (3PL) companies. See
docs/relay-business-context.md.

Run with:

    uv run python -m opspilot.generate_data

Outputs:

    data/synthetic/                    analysis inputs (Parquet tables + metadata)
    data/evaluation/ground_truth.json  planted problems; NEVER an analysis input

Each customer's full onboarding journey is simulated first, then truncated at
the snapshot time ``AS_OF``. Journeys still running at the snapshot are
right-censored: they appear as in-progress, even if the simulation would later
have abandoned them. See docs/synthetic_data.md for definitions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --- Time frame -------------------------------------------------------------

PERIOD_START = pd.Timestamp("2025-10-01")
AS_OF = pd.Timestamp("2026-10-01")  # snapshot time; exclusive end of the period
SECONDS_PER_DAY = 86_400
AS_OF_SECONDS = int((AS_OF - PERIOD_START).total_seconds())

# --- Output locations -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "data" / "synthetic"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "evaluation" / "ground_truth.json"

# --- Domain definitions -----------------------------------------------------

STAGES = (
    "warehouse_provisioning",
    "erp_integration",
    "fulfillment_configuration",
    "testing_validation",
    "go_live",
)
STAGE_DESCRIPTIONS = {
    "warehouse_provisioning": (
        "Account setup: RelayOS tenant created, warehouses, locations and users "
        "provisioned."
    ),
    "erp_integration": (
        "ERP and storefront connectors live; initial inventory import and "
        "ongoing sync verified."
    ),
    "fulfillment_configuration": (
        "Picking, packing, shipping and order-routing rules configured."
    ),
    "testing_validation": (
        "Test orders run end to end; inventory and fulfillment results validated."
    ),
    "go_live": "Live orders processed in RelayOS; onboarding is complete.",
}
SEGMENTS = ("SMB", "Mid-Market", "Enterprise")
PLAN_TIERS = ("starter", "professional", "enterprise")
REGIONS = ("North America", "EMEA", "APAC", "LATAM")
ACQUISITION_CHANNELS = (
    "self_serve",
    "inbound",
    "outbound_sales",
    "partner",
    "referral",
)
ACCOUNT_STATUSES = ("onboarding", "active", "churned")
EVENT_TYPES = ("stage_started", "stage_progress", "stage_completed", "stage_abandoned")
EVENT_OUTCOMES = ("started", "on_track", "blocked", "completed", "abandoned")
TICKET_CATEGORIES = (
    "erp_integration",
    "inventory_sync",
    "fulfillment_rules",
    "shipping_carrier",
    "platform_bug",
    "how_to",
    "user_access",
    "billing",
    "feature_request",
)
# Categories that need engineering or solutions-engineering involvement.
TECHNICAL_TICKET_CATEGORIES = (
    "erp_integration",
    "inventory_sync",
    "fulfillment_rules",
    "shipping_carrier",
    "platform_bug",
)
TICKET_SEVERITIES = ("low", "medium", "high", "critical")

# Onboarding with no activity for this many days is flagged as abandoned.
ABANDONMENT_INACTIVITY_DAYS = 30
# Target time-to-value (signup -> go-live completed) per segment, in days.
TARGET_ONBOARDING_DAYS = {"SMB": 21, "Mid-Market": 40, "Enterprise": 75}

# --- Baseline behaviour -----------------------------------------------------

SEGMENT_MIX = {"SMB": 0.50, "Mid-Market": 0.32, "Enterprise": 0.18}
REGION_MIX = {"North America": 0.45, "EMEA": 0.30, "APAC": 0.17, "LATAM": 0.08}
PLAN_MIX = {
    "SMB": {"starter": 0.65, "professional": 0.33, "enterprise": 0.02},
    "Mid-Market": {"starter": 0.15, "professional": 0.65, "enterprise": 0.20},
    "Enterprise": {"starter": 0.00, "professional": 0.15, "enterprise": 0.85},
}
CHANNEL_MIX = {
    "SMB": {
        "self_serve": 0.40,
        "inbound": 0.30,
        "outbound_sales": 0.05,
        "partner": 0.10,
        "referral": 0.15,
    },
    "Mid-Market": {
        "self_serve": 0.10,
        "inbound": 0.35,
        "outbound_sales": 0.25,
        "partner": 0.20,
        "referral": 0.10,
    },
    "Enterprise": {
        "self_serve": 0.00,
        "inbound": 0.15,
        "outbound_sales": 0.55,
        "partner": 0.25,
        "referral": 0.05,
    },
}

# Median stage duration in days, per segment (lognormal).
STAGE_MEDIAN_DAYS = {
    "warehouse_provisioning": {"SMB": 1.5, "Mid-Market": 3.0, "Enterprise": 6.0},
    "erp_integration": {"SMB": 3.0, "Mid-Market": 7.0, "Enterprise": 14.0},
    "fulfillment_configuration": {"SMB": 2.0, "Mid-Market": 5.0, "Enterprise": 10.0},
    "testing_validation": {"SMB": 1.5, "Mid-Market": 3.0, "Enterprise": 7.0},
    "go_live": {"SMB": 1.0, "Mid-Market": 2.0, "Enterprise": 5.0},
}
STAGE_DURATION_SIGMA = 0.6
CUSTOMER_PACE_SIGMA = 0.35  # shared slowness/speed across a customer's stages
STAGE_GAP_MEAN_DAYS = {"SMB": 0.5, "Mid-Market": 1.0, "Enterprise": 2.0}
MAX_STAGE_GAP_DAYS = 10.0
PROGRESS_GAP_MEAN_DAYS = 6.0
MAX_PROGRESS_GAP_DAYS = 21.0  # keeps active journeys under the inactivity rule
# A progress update is "blocked" once a stage runs past 2x its baseline median.
BLOCKED_AFTER_MEDIAN_MULTIPLE = 2.0

# Probability of abandoning during each stage, before the segment multiplier.
BASE_ABANDON_RATE = {
    "warehouse_provisioning": 0.03,
    "erp_integration": 0.06,
    "fulfillment_configuration": 0.05,
    "testing_validation": 0.03,
    "go_live": 0.01,
}
SEGMENT_ABANDON_MULTIPLIER = {"SMB": 1.3, "Mid-Market": 1.0, "Enterprise": 0.7}

# Support tickets per day while onboarding, before customer-level variation.
TICKET_RATE_PER_DAY = {"SMB": 0.03, "Mid-Market": 0.05, "Enterprise": 0.08}
TICKET_FRAILTY_SHAPE = 3.0  # gamma, mean 1: some customers file far more tickets
POST_ONBOARDING_DAYS = 90
POST_ONBOARDING_RATE_MULTIPLIER = 0.25
# Probabilities in TICKET_CATEGORIES order. Overdue onboarding skews technical.
CATEGORY_MIX = {
    "onboarding": (0.12, 0.08, 0.10, 0.05, 0.12, 0.25, 0.13, 0.05, 0.10),
    "overdue": (0.18, 0.12, 0.14, 0.06, 0.15, 0.20, 0.07, 0.04, 0.04),
    "post_onboarding": (0.04, 0.06, 0.05, 0.10, 0.30, 0.05, 0.05, 0.15, 0.20),
}
SEVERITY_MIX = {
    "onboarding": (0.35, 0.40, 0.20, 0.05),
    "overdue": (0.25, 0.40, 0.27, 0.08),
    "post_onboarding": (0.40, 0.40, 0.17, 0.03),
}
RESOLUTION_MEDIAN_HOURS = {"low": 72, "medium": 40, "high": 18, "critical": 6}
RESOLUTION_SIGMA = 0.8

# --- Planted operational problems -------------------------------------------

ENTERPRISE_DELAY = {
    "segment": "Enterprise",
    "start": pd.Timestamp("2026-04-01"),
    "affected_share": 0.75,
    "duration_multipliers": {"warehouse_provisioning": 1.6, "erp_integration": 2.5},
}
CONFIGURATION_ABANDONMENT = {
    "stage": "fulfillment_configuration",
    "start": pd.Timestamp("2026-06-01"),
    "elevated_rate": 0.20,  # replaces BASE_ABANDON_RATE["fulfillment_configuration"]
}
OVERDUE_TICKET_MULTIPLIER = 5.0  # ticket rate once past the segment target


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    n_customers: int = 1_000


@dataclass
class SyntheticDataset:
    tables: dict[str, pd.DataFrame]
    metadata: dict
    ground_truth: dict


# --- Simulation helpers ------------------------------------------------------


def _to_seconds(days: float) -> int:
    return int(np.rint(days * SECONDS_PER_DAY))


def _start_offset_days(ts: pd.Timestamp) -> float:
    return (ts - PERIOD_START).total_seconds() / SECONDS_PER_DAY


def _draw_signup_days(rng: np.random.Generator, n: int) -> np.ndarray:
    """Signup times in days since PERIOD_START: mild growth, quieter weekends."""
    days = pd.date_range(PERIOD_START, AS_OF - pd.Timedelta(days=1), freq="D")
    weights = np.linspace(0.8, 1.2, len(days)) * np.where(days.dayofweek < 5, 1.0, 0.25)
    day_index = np.sort(rng.choice(len(days), size=n, p=weights / weights.sum()))
    business_hour = rng.uniform(8, 19, size=n)
    return day_index + business_hour / 24


def _choice(rng: np.random.Generator, mix: dict[str, float]) -> str:
    return str(rng.choice(list(mix), p=list(mix.values())))


def _simulate_journey(
    rng: np.random.Generator, signup_day: float, segment: str
) -> tuple[list[tuple], dict]:
    """Simulate a full journey, ignoring the snapshot. Times are in seconds."""
    delay_affected = (
        segment == ENTERPRISE_DELAY["segment"]
        and signup_day >= _start_offset_days(ENTERPRISE_DELAY["start"])
        and rng.random() < ENTERPRISE_DELAY["affected_share"]
    )
    pace = rng.lognormal(0.0, CUSTOMER_PACE_SIGMA)
    config_spike_start = _start_offset_days(CONFIGURATION_ABANDONMENT["start"])

    events: list[tuple] = []
    t = signup_day + min(rng.exponential(0.3), 3.0)
    outcome, abandoned_stage = "completed", None

    for order, stage in enumerate(STAGES, start=1):
        base_median = STAGE_MEDIAN_DAYS[stage][segment]
        median = base_median * pace
        if delay_affected:
            median *= ENTERPRISE_DELAY["duration_multipliers"].get(stage, 1.0)
        duration = rng.lognormal(np.log(median), STAGE_DURATION_SIGMA)

        abandon_rate = BASE_ABANDON_RATE[stage]
        if stage == CONFIGURATION_ABANDONMENT["stage"] and t >= config_spike_start:
            abandon_rate = CONFIGURATION_ABANDONMENT["elevated_rate"]
        abandons = rng.random() < abandon_rate * SEGMENT_ABANDON_MULTIPLIER[segment]

        stage_end = t + duration * (rng.uniform(0.2, 0.9) if abandons else 1.0)
        events.append((stage, order, "stage_started", _to_seconds(t), "started"))

        blocked_after = BLOCKED_AFTER_MEDIAN_MULTIPLE * base_median
        progress_t = t
        while True:
            progress_t += min(
                rng.exponential(PROGRESS_GAP_MEAN_DAYS), MAX_PROGRESS_GAP_DAYS
            )
            if progress_t >= stage_end:
                break
            status = "blocked" if progress_t - t > blocked_after else "on_track"
            events.append(
                (stage, order, "stage_progress", _to_seconds(progress_t), status)
            )

        if abandons:
            # Customer success flags abandonment after a period of silence.
            last_activity = events[-1][3]
            flagged = last_activity + ABANDONMENT_INACTIVITY_DAYS * SECONDS_PER_DAY
            events.append((stage, order, "stage_abandoned", flagged, "abandoned"))
            outcome, abandoned_stage = "abandoned", stage
            break

        events.append(
            (stage, order, "stage_completed", _to_seconds(stage_end), "completed")
        )
        t = stage_end + min(
            rng.exponential(STAGE_GAP_MEAN_DAYS[segment]), MAX_STAGE_GAP_DAYS
        )

    # Last moment the customer was actually engaged (abandon flags don't count).
    engaged_events = [e for e in events if e[2] != "stage_abandoned"]
    journey = {
        "latent_outcome": outcome,
        "abandoned_stage": abandoned_stage,
        "delay_affected": delay_affected,
        "engagement_end": engaged_events[-1][3],
    }
    return events, journey


def _draw_ticket_times(
    rng: np.random.Generator, start: int, end: int, rate_per_day: float
) -> np.ndarray:
    if end <= start:
        return np.empty(0, dtype=np.int64)
    count = rng.poisson(rate_per_day * (end - start) / SECONDS_PER_DAY)
    return rng.integers(start, end, size=count)


def _simulate_tickets(
    rng: np.random.Generator,
    signup: int,
    segment: str,
    journey: dict,
) -> list[tuple]:
    """Tickets during onboarding, at an elevated rate once past the target."""
    rate = TICKET_RATE_PER_DAY[segment] * rng.gamma(
        TICKET_FRAILTY_SHAPE, 1.0 / TICKET_FRAILTY_SHAPE
    )
    onboarding_end = min(journey["engagement_end"], AS_OF_SECONDS)
    target = signup + TARGET_ONBOARDING_DAYS[segment] * SECONDS_PER_DAY

    windows = [
        ("onboarding", signup, min(onboarding_end, target), rate),
        ("overdue", target, onboarding_end, rate * OVERDUE_TICKET_MULTIPLIER),
    ]
    if journey["latent_outcome"] == "completed":
        post_end = min(
            onboarding_end + POST_ONBOARDING_DAYS * SECONDS_PER_DAY, AS_OF_SECONDS
        )
        post_rate = rate * POST_ONBOARDING_RATE_MULTIPLIER
        windows.append(("post_onboarding", onboarding_end, post_end, post_rate))

    tickets = []
    for phase, start, end, phase_rate in windows:
        for created in _draw_ticket_times(rng, start, end, phase_rate):
            category = str(rng.choice(TICKET_CATEGORIES, p=CATEGORY_MIX[phase]))
            severity = str(rng.choice(TICKET_SEVERITIES, p=SEVERITY_MIX[phase]))
            hours = rng.lognormal(
                np.log(RESOLUTION_MEDIAN_HOURS[severity]), RESOLUTION_SIGMA
            )
            resolved = int(created) + max(_to_seconds(hours / 24), 60)
            tickets.append((int(created), category, severity, resolved))
    return tickets


def _timestamps(seconds: pd.Series) -> pd.Series:
    return PERIOD_START + pd.to_timedelta(seconds, unit="s")


# --- Public API --------------------------------------------------------------


def generate_dataset(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Generate all tables, the analysis metadata and the ground truth."""
    config = config or GeneratorConfig()
    rng = np.random.default_rng(config.seed)
    n = config.n_customers
    customer_ids = [f"CUST-{i:04d}" for i in range(1, n + 1)]
    signup_days = _draw_signup_days(rng, n)

    attributes = []
    for customer_id in customer_ids:
        segment = _choice(rng, SEGMENT_MIX)
        attributes.append(
            {
                "customer_id": customer_id,
                "segment": segment,
                "plan_tier": _choice(rng, PLAN_MIX[segment]),
                "region": _choice(rng, REGION_MIX),
                "acquisition_channel": _choice(rng, CHANNEL_MIX[segment]),
            }
        )

    event_rows, ticket_rows, journeys = [], [], []
    for customer_id, signup_day, attrs in zip(customer_ids, signup_days, attributes):
        segment = attrs["segment"]
        events, journey = _simulate_journey(rng, signup_day, segment)
        tickets = _simulate_tickets(rng, _to_seconds(signup_day), segment, journey)
        # Right-censoring: only what happened before the snapshot is observable.
        observed = [e for e in events if e[3] <= AS_OF_SECONDS]
        event_rows += [(customer_id, *e) for e in observed]
        ticket_rows += [(customer_id, *t) for t in tickets if t[0] <= AS_OF_SECONDS]

        observed_types = {(e[0], e[2]) for e in observed}
        if ("go_live", "stage_completed") in observed_types:
            status = "active"
        elif any(event_type == "stage_abandoned" for _, event_type in observed_types):
            status = "churned"
        else:
            status = "onboarding"
        journeys.append(
            {**journey, "customer_id": customer_id, "account_status": status}
        )

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_at": _timestamps(
                pd.Series([_to_seconds(d) for d in signup_days])
            ).to_numpy(),
            "account_status": [j["account_status"] for j in journeys],
        }
    )

    events = pd.DataFrame(
        event_rows,
        columns=[
            "customer_id",
            "stage",
            "stage_order",
            "event_type",
            "seconds",
            "outcome",
        ],
    )
    events = events.sort_values(
        ["customer_id", "seconds", "stage_order"], kind="stable"
    )
    events.insert(0, "event_id", [f"EVT-{i:06d}" for i in range(1, len(events) + 1)])
    events["event_timestamp"] = _timestamps(events.pop("seconds"))
    events = events[
        [
            "event_id",
            "customer_id",
            "stage",
            "stage_order",
            "event_type",
            "event_timestamp",
            "outcome",
        ]
    ].reset_index(drop=True)

    tickets = pd.DataFrame(
        ticket_rows,
        columns=["customer_id", "created", "category", "severity", "resolved"],
    ).sort_values(["created", "customer_id"], kind="stable")
    tickets.insert(0, "ticket_id", [f"TKT-{i:05d}" for i in range(1, len(tickets) + 1)])
    tickets["created_at"] = _timestamps(tickets.pop("created"))
    resolved = tickets.pop("resolved")
    # Tickets still open at the snapshot have no resolution timestamp yet.
    tickets["resolved_at"] = _timestamps(resolved.where(resolved <= AS_OF_SECONDS))
    tickets = tickets[
        [
            "ticket_id",
            "customer_id",
            "created_at",
            "category",
            "severity",
            "resolved_at",
        ]
    ].reset_index(drop=True)

    tables = {
        "customers": customers,
        "customer_attributes": pd.DataFrame(attributes),
        "onboarding_events": events,
        "support_tickets": tickets,
    }
    return SyntheticDataset(
        tables=tables,
        metadata=build_metadata(config),
        ground_truth=build_ground_truth(config, pd.DataFrame(journeys), tables),
    )


def build_metadata(config: GeneratorConfig) -> dict:
    """Context an analyst legitimately needs. Contains no planted-problem info."""
    return {
        "description": (
            "Synthetic customer onboarding data for Relay Technologies, a fictional "
            "B2B SaaS company."
        ),
        "company": {
            "name": "Relay Technologies",
            "product": "RelayOS",
            "product_description": (
                "Cloud warehouse fulfillment operations platform: storefront and ERP "
                "connectors, inventory sync, picking/packing/fulfillment rules, "
                "order processing and shipping performance monitoring."
            ),
            "customers": "E-commerce retailers and third-party logistics (3PL) companies.",
            "business_context_doc": "docs/relay-business-context.md",
        },
        "seed": config.seed,
        "n_customers": config.n_customers,
        "period_start": PERIOD_START.isoformat(),
        "as_of": AS_OF.isoformat(),
        "timezone": "UTC",
        "stages": [
            {"order": i, "name": s, "description": STAGE_DESCRIPTIONS[s]}
            for i, s in enumerate(STAGES, start=1)
        ],
        "definitions": {
            "onboarding_completed": "A go_live stage_completed event exists.",
            "on_target": (
                "Completed within the segment's target time-to-value, measured "
                "from signup_at to the go_live stage_completed timestamp."
            ),
            "abandoned": (
                "A stage_abandoned event exists. Customer success records it "
                f"{ABANDONMENT_INACTIVITY_DAYS} days after the last onboarding activity."
            ),
            "in_progress": (
                "Neither completed nor abandoned at as_of. These journeys are "
                "right-censored and must not be counted as abandoned."
            ),
            "account_status": {
                "onboarding": "Onboarding in progress at as_of.",
                "active": "Onboarding completed.",
                "churned": "Onboarding abandoned.",
            },
            "event_outcomes": {
                "started": "Stage started.",
                "on_track": "Progress update within the stage's expected duration.",
                "blocked": "Progress update after the stage overran expectations.",
                "completed": "Stage completed.",
                "abandoned": "Onboarding abandoned during this stage.",
            },
            "resolved_at": "Null when the ticket was still open at as_of.",
            "ticket_categories": {
                "erp_integration": "ERP or storefront connector errors and mapping.",
                "inventory_sync": "Inventory import or sync mismatches.",
                "fulfillment_rules": "Picking, packing or order-routing rule issues.",
                "shipping_carrier": "Carrier accounts, labels and rate shopping.",
                "platform_bug": "Defects in RelayOS itself.",
                "how_to": "Product usage questions.",
                "user_access": "Logins, roles and permissions.",
                "billing": "Invoices, plans and payments.",
                "feature_request": "Requests for new capabilities.",
            },
            "technical_ticket_categories": list(TECHNICAL_TICKET_CATEGORIES),
        },
        "target_onboarding_days": TARGET_ONBOARDING_DAYS,
        "abandonment_inactivity_days": ABANDONMENT_INACTIVITY_DAYS,
        "tables": {
            "customers": ["customer_id", "signup_at", "account_status"],
            "customer_attributes": [
                "customer_id",
                "segment",
                "plan_tier",
                "region",
                "acquisition_channel",
            ],
            "onboarding_events": [
                "event_id",
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
        },
    }


def build_ground_truth(
    config: GeneratorConfig, journeys: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> dict:
    """Describe the planted problems. For evaluation only, never analysis input."""
    censored = journeys[journeys["account_status"] == "onboarding"]
    return {
        "warning": (
            "EVALUATION ONLY. Do not provide this file to any analysis or AI agent."
        ),
        "seed": config.seed,
        "as_of": AS_OF.isoformat(),
        "row_counts": {name: len(df) for name, df in tables.items()},
        "censoring": {
            "in_progress_at_as_of": len(censored),
            "in_progress_that_would_later_abandon": int(
                (censored["latent_outcome"] == "abandoned").sum()
            ),
            "note": (
                "Some in-progress journeys are latent abandonments that the snapshot "
                "cannot see yet. Counting in-progress as abandoned would be wrong."
            ),
        },
        "problems": [
            {
                "id": "enterprise_erp_integration_delay",
                "description": (
                    "Enterprise ERP and inventory integrations slow down for "
                    "customers who sign up in the second half of the period; "
                    "warehouse provisioning is also somewhat slower."
                ),
                "affected_segment": ENTERPRISE_DELAY["segment"],
                "time_window": {
                    "basis": "signup_at",
                    "start": ENTERPRISE_DELAY["start"].isoformat(),
                    "end": AS_OF.isoformat(),
                },
                "affected_stages": list(ENTERPRISE_DELAY["duration_multipliers"]),
                "mechanism": {
                    "affected_share_of_segment_signups": ENTERPRISE_DELAY[
                        "affected_share"
                    ],
                    "median_duration_multipliers": ENTERPRISE_DELAY[
                        "duration_multipliers"
                    ],
                },
                "affected_customers": int(journeys["delay_affected"].sum()),
                "expected_trend": (
                    "For 2026-04 onward Enterprise signups, the share of "
                    "erp_integration stages lasting over 21 days roughly doubles "
                    "(about 25% -> 55-65%) and the median completed duration rises "
                    "about 1.5x. No comparable change for SMB or Mid-Market."
                ),
                "detection_hint": (
                    "Compare stage durations (stage_started -> stage_completed) by "
                    "segment and signup half. Late cohorts are censored, which biases "
                    "their completed durations downward; prefer 'share exceeding N "
                    "days' among stages with at least N days of follow-up."
                ),
            },
            {
                "id": "fulfillment_configuration_abandonment",
                "description": (
                    "Abandonment during fulfillment workflow configuration jumps for "
                    "customers who start that stage on or after 2026-06-01."
                ),
                "affected_segment": "all",
                "affected_stage": CONFIGURATION_ABANDONMENT["stage"],
                "time_window": {
                    "basis": "fulfillment_configuration stage_started timestamp",
                    "start": CONFIGURATION_ABANDONMENT["start"].isoformat(),
                    "end": AS_OF.isoformat(),
                },
                "mechanism": {
                    "baseline_rate_before_segment_multiplier": BASE_ABANDON_RATE[
                        "fulfillment_configuration"
                    ],
                    "elevated_rate_before_segment_multiplier": CONFIGURATION_ABANDONMENT[
                        "elevated_rate"
                    ],
                    "segment_multipliers": SEGMENT_ABANDON_MULTIPLIER,
                },
                "latent_abandonments_at_stage": int(
                    (journeys["abandoned_stage"] == "fulfillment_configuration").sum()
                ),
                "expected_trend": (
                    "fulfillment_configuration abandonment rate, among journeys with "
                    "at least 60 "
                    "days of follow-up after entering the stage, rises from about 5% "
                    "to about 20%. Other stages stay roughly flat."
                ),
                "detection_hint": (
                    "Use a conditional stage abandonment rate and exclude journeys "
                    "without enough follow-up time (right-censoring)."
                ),
            },
            {
                "id": "prolonged_onboarding_technical_support",
                "description": (
                    "Customers whose onboarding runs past the segment target file "
                    "support tickets at a higher daily rate, skewed toward technical "
                    "categories (ERP integration, inventory sync, fulfillment rules) "
                    "and higher severities."
                ),
                "affected_segment": "all",
                "time_window": {
                    "basis": "days since signup_at beyond target_onboarding_days",
                    "start": PERIOD_START.isoformat(),
                    "end": AS_OF.isoformat(),
                },
                "mechanism": {
                    "target_onboarding_days": TARGET_ONBOARDING_DAYS,
                    "ticket_rate_multiplier_after_target": OVERDUE_TICKET_MULTIPLIER,
                    "customer_level_gamma_frailty_shape": TICKET_FRAILTY_SHAPE,
                    "technical_ticket_categories": list(TECHNICAL_TICKET_CATEGORIES),
                },
                "expected_trend": (
                    "Segment-standardised technical tickets per 30 onboarding days "
                    "are roughly 2-3x higher for prolonged journeys than for on-target "
                    "ones (all tickets: about 2x); strongest in Enterprise. Within "
                    "prolonged journeys, the technical ticket "
                    "rate after the target date is several times the rate before it. "
                    "SMB files few tickets, so its per-segment ratio is noisy."
                ),
                "detection_hint": (
                    "Filter to technical_ticket_categories, normalise ticket counts "
                    "by onboarding exposure time and compare "
                    "within segment. Enterprise delays (problem 1) raise the share of "
                    "prolonged journeys, so segment is a confounder."
                ),
                "causal_note": (
                    "In the generator, prolonged onboarding raises ticket rates. "
                    "Observational data alone cannot establish that direction."
                ),
            },
        ],
    }


def write_dataset(
    dataset: SyntheticDataset,
    analysis_dir: Path = ANALYSIS_DIR,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
) -> None:
    """Write the analysis inputs and the ground truth to separate locations."""
    analysis_dir, ground_truth_path = Path(analysis_dir), Path(ground_truth_path)
    if ground_truth_path.resolve().is_relative_to(analysis_dir.resolve()):
        raise ValueError("Ground truth must be stored outside the analysis directory.")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    for name, df in dataset.tables.items():
        df.to_parquet(analysis_dir / f"{name}.parquet", index=False)
    (analysis_dir / "dataset_metadata.json").write_text(
        json.dumps(dataset.metadata, indent=2) + "\n"
    )

    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path.write_text(json.dumps(dataset.ground_truth, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--n-customers", type=int, default=GeneratorConfig.n_customers)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--ground-truth-path", type=Path, default=GROUND_TRUTH_PATH)
    args = parser.parse_args(argv)

    dataset = generate_dataset(GeneratorConfig(args.seed, args.n_customers))
    write_dataset(dataset, args.analysis_dir, args.ground_truth_path)

    print(f"Wrote analysis inputs to {args.analysis_dir}")
    for name, df in dataset.tables.items():
        print(f"  {name}: {len(df):,} rows")
    print(f"Wrote ground truth (evaluation only) to {args.ground_truth_path}")


if __name__ == "__main__":
    main()
