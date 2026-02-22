# Multi-Provider Review and Gap Analysis (POC)

## Scope Reviewed
- Provider adapters: OpenAI, Anthropic, Groq
- Unified schema and aggregation pipeline
- Dashboard tabs and visualizations
- Forecasting and model comparison flows
- Rate-limit and capacity handling

## What Is Implemented Correctly

### 1. Multi-Provider Data Ingestion
- OpenAI usage + cost endpoints are integrated and normalized.
- Anthropic usage report + cost report are integrated and normalized.
- Groq metrics endpoint integration is implemented with query fallback and normalization.
- Unified provider schema is used across providers (`timestamp`, `provider`, `model`, `project_id`, `calls`, `input_tokens`, `output_tokens`, `total_tokens`, `cost_usd`, `currency`, `cost_source`).

### 2. Cross-Provider Cost Insights
- Monthly spend trend vs budget.
- Spend by provider.
- Top models by cost.
- Model cost breakdown table with:
  - `Rank`
  - `Model`
  - `Provider`
  - `Calls (24h)`
  - `Avg Tokens`
  - `CPI`
  - `7-Day Trend`
  - `Status`

### 3. Windowed Analytics
- Top models by cost now supports:
  - last 24 hours
  - last 7 days
  - last 30 days
- Cost reduction trend view (current 7d vs previous 7d).

### 4. Rate-Limit & Capacity Awareness
- Observed usage pressure is computed:
  - observed TPM (peak)
  - observed RPM (peak)
  - observed TPD (peak)
  - observed RPD (peak)
- OpenAI project/model limits are fetched via Admin API and joined to observed load.
- Utilization status is labeled (`Healthy`, `Watch`, `High Risk`, `No Limit Data`).
- GPU utilization is surfaced as a static dummy placeholder for this POC.

### 5. Model Intelligence & Migration Support
- Model intelligence tab added with:
  - model-fit table (best-suited-for, latency/reasoning/cost profile)
  - migration recommendation simulator (source -> target)
  - projected spend delta using:
    - observed target CPI (preferred),
    - or pricing-map estimate,
    - or manual CPI override.

## Weak Areas / Constraints

### 1. Provider API Capability Limits
- Anthropic and Groq do not expose OpenAI-style organization/project limit APIs in this POC.
- Groq metrics endpoint may be unavailable for non-entitled orgs (enterprise gating).
- Prompt/response body-level analytics are not exposed by the org usage/cost endpoints used here.

### 2. Model Intelligence Data Source
- Capability labels are lightweight heuristics (not benchmark-ground-truth).
- Migration recommendations are cost-aware first; they are not automatic quality-evaluation results.

### 3. Cost Source Variability
- OpenAI model-level costs are reconciled/estimated from aggregate cost and token distributions.
- Groq costs are estimated from token metrics + local pricing map.
- Anthropic cost report amount parsing assumes cents -> USD conversion.

## Structural Improvements Applied
- Added OpenAI project rate-limit retrieval and normalization.
- Added dedicated analytics helpers for:
  - cross-provider KPIs,
  - windowed top-model ranking,
  - cost reduction trends,
  - provider metric trends,
  - capacity and utilization.
- Added model intelligence module separated from adapters/analytics to keep concerns clean.
- Added unit tests for:
  - new analytics helpers,
  - model intelligence logic,
  - OpenAI rate-limit table normalization.

## Remaining Gaps (Actionable)
- Add explicit benchmark integration (latency/quality) if you want evidence-based migration recommendations.
- Add provider-specific authenticated limit ingestion for Anthropic/Groq when APIs become available.
- Add optional manual limit profiles for Anthropic/Groq in UI for richer capacity planning.
- Add anomaly detection and budget alerting beyond baseline trend deltas.
