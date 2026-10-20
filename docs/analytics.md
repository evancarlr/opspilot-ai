# Onboarding analytics engine (Milestone 2)

`src/opspilot/analytics/` is the deterministic layer the future AI agent will
call. It computes every number itself; an LLM only interprets the results.

## Data access

- An in-memory DuckDB connection with views over the four Parquet files in
  `data/synthetic/`. Data is never copied.
- Reads only those files plus `dataset_metadata.json` (as-of date, stages,
  segment targets, ticket categories).
- Refuses directories under `evaluation/`, never imports the generator, and
  never opens `ground_truth.json`. Tests enforce all three.

## Shared definitions (SQL views)

- `journeys`, one row per customer:
  - **status:** completed (go_live completed), abandoned (stage_abandoned
    event) or in progress.
  - **Onboarding window:** signup to go-live, to last activity (abandoned), or
    to as_of (in progress).
- `stage_entries`, one row per customer and stage: start, completion,
  abandonment and last activity.
- Events after `as_of` are ignored.

## Analyses

| Method | Question | Censoring / denominator rule |
|---|---|---|
| `completion_rate` | % completing onboarding | Only signups at least `min_follow_up_days` (120) before as_of |
| `time_to_onboard` | Median days to go-live by segment | Kaplan–Meier; in-progress censored at as_of, abandoned at last activity. Completed-only median shown for reference |
| `stage_abandonment` | Abandonment rate per stage | Stage entries started at least `min_follow_up_days` (60) before as_of; period filter applies to stage start |
| `delay_trend` | Delay rate by signup cohort (whole journey or one stage) | Judged at checkpoint = start + threshold, once checkpoint + 30 days has passed. Delayed = not done but still active; disengaged journeys are abandonment and excluded |
| `support_ticket_rates` | Tickets per onboarding day, delayed vs on-target | Only tickets inside each onboarding window; compared within segment; headline is a segment-standardised ratio |

Every result is a Pydantic model with `value`, `unit`, `definition`,
`sample_size`, `window` (basis, start, end, as_of), `filters`, `notes` and a
breakdown. Empty cohorts return `value=None` and a note. Unknown filter values
raise `ValueError` with the valid options.

## Limitations

- Each delay metric is judged at a single checkpoint. Journeys that are
  delayed at the checkpoint and abandon later count as delayed.
- Kaplan–Meier treats abandonment as censoring, so its median describes
  customers who keep onboarding, not the whole cohort.
- Trends are descriptive. There are no significance tests or confidence
  intervals yet, and small cohorts are noisy.
- Whole-journey delay trends mix segments with different targets. Filter by
  segment for clean comparisons.
