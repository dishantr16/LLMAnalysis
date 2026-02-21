# LLM Cost Analysis Module (POC)

Streamlit dashboard that pulls OpenAI, Anthropic, and Groq usage/cost signals and visualizes model-wise usage, token consumption, and billing trends.

## What this POC covers
- OpenAI Admin key based authorization (user-pasted, session-only)
- OpenAI org usage data across endpoints:
  - `/v1/organization/usage/completions`
  - `/v1/organization/usage/images`
  - `/v1/organization/usage/moderations`
  - `/v1/organization/usage/audio_speeches`
  - `/v1/organization/usage/audio_transcriptions`
  - `/v1/organization/usage/vector_stores`
  - `/v1/organization/usage/code_interpreter_sessions`
- OpenAI org cost data (`/v1/organization/costs`)
- Anthropic org usage + cost reporting:
  - `/v1/organizations/usage_report/messages`
  - `/v1/organizations/cost_report`
- Groq metrics API usage:
  - `/v1/metrics/prometheus/api/v1/query_range` (requests/tokens metrics)
- Interactive charts for:
  - daily trends
  - weekly/monthly/yearly summaries
  - model-wise token usage
  - model-wise estimated cost
  - project and line-item billing views
- Usage Explorer for non-completions endpoints (metric + dimension breakdowns)
- Forecasts (beta) for 30-day token and cost projections
- Unified provider adapters:
  - OpenAI adapter (implemented)
  - Anthropic adapter (implemented)
  - Groq adapter (implemented, cost estimated from token metrics)
  - Azure OpenAI adapter (scaffold)
- Unified provider insights:
  - Monthly Spend Trend vs Budget
  - Spend by Provider
  - Top Models by Cost
  - Model Cost Breakdown table (`Rank`, `Model`, `Provider`, `Calls (24h)`, `Avg Tokens`, `CPI`, `7-Day Trend`, `Status`)
- CSV export for usage and cost datasets

## What this POC does not cover
- OAuth-style account connection flow
- Raw prompt/response text analytics from OpenAI org endpoints
- True per-request billing from OpenAI (not exposed directly via aggregate endpoints)
- Production-grade multi-tenant secret management
- Production-grade forecasting (current forecast is baseline linear projection)

## Project structure
- `app.py`: Streamlit entrypoint
- `src/openai_client.py`: OpenAI API client with pagination/retries
- `src/providers/`: provider adapter layer and unified schema scaffold
- `src/fetchers.py`: Query objects and endpoint fetch functions
- `src/transformers.py`: JSON to DataFrame normalization
- `src/analytics.py`: KPI and aggregation logic
- `src/charts.py`: Plotly chart builders
- `src/ui.py`: Streamlit layout helpers
- `docs/llm-cost-analysis-poc-plan.md`: implementation plan
- `tests/`: unit tests

## Setup
1. Create/activate virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add keys (either in `.env` or Streamlit sidebar):
   - `OPENAI_ADMIN_KEY`
   - `ANTHROPIC_ADMIN_KEY`
   - `GROQ_API_KEY`

## Run
```bash
streamlit run app.py
```

## Test
```bash
pytest
```

## Notes
- Keep `MODEL_PRICING_PER_MILLION` in `src/config.py` updated manually.
- Keep `GROQ_MODEL_PRICING_PER_MILLION` in `src/config.py` updated manually.
- Estimated model cost is derived from token counts and local pricing map when direct model cost attribution is unavailable.
- Groq integration depends on access to org metrics endpoints; without that entitlement, metrics queries will return empty/forbidden responses.
