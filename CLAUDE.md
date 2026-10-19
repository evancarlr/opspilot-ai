# OpsPilot AI — Development Instructions

## Project
OpsPilot is an AI-powered business operations analyst.

Initial domain: customer onboarding at a fictional B2B SaaS company.

## Stack
Python 3.12, uv, pandas, NumPy, DuckDB,
Pydantic, pytest.

Future: Anthropic API and Streamlit.

## Development principles
- Keep solutions simple and maintainable.
- Separate analytics from AI reasoning.
- Never fabricate business metrics.
- Write tests for important functionality.
- Use reproducible synthetic data.
- Distinguish correlation from causation.
- Document architectural decisions.

## Claude Code workflow
1. Inspect the existing repository.
2. Propose a concise implementation plan.
3. Implement the requested scope.
4. Run tests and fix errors.
5. Summarize design decisions.
6. Stop at the requested milestone.

## Guardrails
- Do not commit changes automatically.
- Do not install dependencies without approval.
- Do not use real customer or patient data.
- Never expose API keys.

## Current Milestone
Generate synthetic SaaS onboarding data.
Do not build the LLM agent or frontend yet.