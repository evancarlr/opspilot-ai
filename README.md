# OpsPilot AI

An AI-powered business operations analyst. Initial domain: customer onboarding
at Relay Technologies, a fictional B2B SaaS company selling the RelayOS warehouse
fulfillment platform ([business context](docs/relay-business-context.md)).

## Synthetic data

```bash
uv run python -m opspilot.generate_data          # seed 42, 1,000 customers
uv run python -m opspilot.generate_data --help   # seed / size / output options
uv run pytest
```

- Analysis inputs: `data/synthetic/` (Parquet tables + `dataset_metadata.json`)
- Evaluation only: `data/evaluation/ground_truth.json`. Never give this to an analysis agent.

See [docs/synthetic_data.md](docs/synthetic_data.md) for the data model, stage
definitions, success criteria and right-censoring rules.

## Analytics engine

```bash
uv run python -m opspilot.analytics                          # five analyses, all customers
uv run python -m opspilot.analytics --segment Enterprise     # filter (repeatable)
uv run python -m opspilot.analytics --period-start 2026-04-01 --json
```

See [docs/analytics.md](docs/analytics.md) for metric definitions and limitations.
