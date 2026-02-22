# AI Model Advisor (POC) - Current State and Real-Data Plan

## 1. Purpose
The AI Model Advisor recommends better-suited models (SLM, LLM, Frontier) based on real usage and spend patterns in your organization.

Current provider scope:
- OpenAI
- Anthropic

This document explains:
- What is implemented now.
- What data is truly fetched from provider APIs.
- Where static heuristics are currently used.
- How to move to a real-data-first advisor with minimal assumptions.


## 2. What Is Implemented Today

### 2.1 Advisor Engine
Implemented in `src/model_advisor.py`.

Key capabilities:
- Builds workload profile from normalized usage history.
- Scores model categories: SLM, LLM, Frontier.
- Ranks candidate models and returns:
- recommended category/model/provider
- confidence score
- estimated monthly cost and savings
- reasoning and trade-offs

Main entrypoint:
- `run_ai_model_advisor(...)` in `src/model_advisor.py`.

### 2.2 UI Integration
Implemented in `app.py`.

New tab:
- `AI Model Advisor`

Features in tab:
- Workload and objective inputs.
- Recommendation summary cards.
- Category score chart.
- Candidate comparison chart.
- Alternatives table.
- Assumptions panel.

### 2.3 Charts
Implemented in `src/charts.py`.

New chart builders:
- `advisor_category_score_chart(...)`
- `advisor_candidates_chart(...)`

### 2.4 Tests
Implemented in `tests/test_model_advisor.py`.

Coverage:
- Workload profile generation.
- Structured recommendation output.
- Provider-lock behavior (no cross-provider migration when disabled).


## 3. Real Provider Data Fetched Today

## 3.1 OpenAI Data Sources
Implemented via:
- `src/providers/openai_adapter.py`
- `src/fetchers.py`
- `src/transformers.py`

Fetched endpoints:
- `/organization/usage/completions`
- `/organization/usage/images`
- `/organization/usage/moderations`
- `/organization/usage/audio_speeches`
- `/organization/usage/audio_transcriptions`
- `/organization/usage/vector_stores`
- `/organization/usage/code_interpreter_sessions`
- `/organization/costs`
- `/organization/projects`
- `/organization/projects/{project_id}/rate_limits`

Fields available in normalized usage/cost:
- `timestamp`
- `provider`
- `model`
- `project_id`
- `calls`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `cost_usd`
- `currency`
- `cost_source`

Notes:
- OpenAI model-level cost in unified view is reconciled/allocated from reported costs when direct model attribution is incomplete.
- Additional endpoint datasets are available in Usage Explorer.

## 3.2 Anthropic Data Sources
Implemented via:
- `src/providers/anthropic_adapter.py`

Fetched endpoints:
- `/organizations/usage_report/messages`
- `/organizations/cost_report`

Fields available in normalized usage/cost:
- `timestamp`
- `provider`
- `model`
- `project_id` (workspace mapping)
- `calls`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `cost_usd`
- `currency`
- `cost_source`

Notes:
- Cost rows are allocated to usage rows where necessary.
- If usage exists but cost report returns no rows, costs remain zero for those rows.


## 4. Where Static Heuristics Are Used Right Now
Static/heuristic logic exists only in advisor scoring metadata, not in provider usage fetch:

- Model catalog in `src/model_advisor.py`:
- baseline pricing
- baseline latency
- baseline quality score
- baseline capability tags

Why this exists:
- Provider usage/cost APIs do not provide full model quality metadata or benchmark scores.
- Task-level labels and latency are not available in current provider billing endpoints.

What is already real:
- Usage volumes.
- Token consumption.
- Calls.
- Cost totals and allocations from provider reports.
- Provider/model distribution in your account history.


## 5. Gap to Your Requirement ("No Static Data")
Your requirement is real-data-first recommendations from OpenAI/Anthropic telemetry.

Current gap:
- Candidate ranking still relies partly on static catalog priors for cost/latency/quality on models not strongly observed in your own history.

Implication:
- Advisor is currently useful for directional guidance but not yet purely telemetry-driven for all candidates.


## 6. Real-Data-First Build Plan

## Phase A - Telemetry-Only Advisor Mode (Immediate)
Goal:
- Remove static model assumptions from recommendation decisions.

Implementation:
- Candidate set = models observed in your own data window only.
- CPI = `sum(cost_usd) / sum(calls)` from provider data only.
- Token profile = observed avg input/output per call only.
- Confidence drops when sample size per model is low.
- If insufficient sample size, return "insufficient evidence" instead of guessing.

Output effect:
- Recommendations become strictly evidence-based for your real traffic.
- Coverage may be narrower initially, but fully non-static.

## Phase B - Dynamic Model Registry from Provider APIs
Goal:
- Expand candidate list without hardcoded model examples.

Implementation:
- Add provider catalog fetchers:
- OpenAI model list endpoint.
- Anthropic model list endpoint.
- Persist registry snapshots with timestamp/version.

Important:
- Provider model list endpoints typically do not include full pricing and benchmark quality scores.
- This phase expands discovery, not complete scoring by itself.

## Phase C - Real Pricing Source of Truth
Goal:
- Remove hardcoded pricing from advisor.

Implementation options:
- Use provider-exported billing reports/invoice exports as pricing truth.
- Maintain versioned pricing table in storage with effective dates.
- Backfill historical effective CPI per model from usage + spend.

Result:
- Cost recommendations become reproducible and auditable.

## Phase D - Task Pattern and Latency Ground Truth
Goal:
- Replace guessed task complexity and latency priors with real operational telemetry.

Required instrumentation outside billing APIs:
- Request logs at gateway/application layer:
- `request_id`, `task_type`, `latency_ms`, `status`, `provider`, `model`
- Optional prompt-length bins (not prompt text), output-length bins.

Result:
- Advisor can make task-aware and SLA-aware recommendations with real latency/quality proxies.

## Phase E - Cross-Provider Migration Scoring (Production Quality)
Goal:
- Robust recommendation with measurable trade-offs.

Scoring dimensions:
- Cost efficiency (observed CPI, monthly projection).
- Latency fit (actual p50/p95 from logs).
- Reliability fit (error rate/timeouts).
- Task fit (observed task success proxy).
- Migration effort (provider change overhead).

Result:
- Recommendations become explainable, evidence-backed, and operationally actionable.


## 7. Data Model for Real-Data Advisor

Keep unified schema as current base:
- `timestamp`
- `provider`
- `model`
- `project_id`
- `calls`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `cost_usd`
- `currency`
- `cost_source`

Add optional tables for stronger advisor quality:
- `model_registry_snapshots`
- `model_pricing_versions`
- `request_telemetry` (latency/status/task labels from your app gateway)
- `advisor_recommendation_history`


## 8. Recommendation Trust Levels
To keep output honest, return confidence tiers:

- High confidence:
- model has sufficient observed calls and observed spend.
- Medium confidence:
- model observed but limited sample size.
- Low confidence:
- model not observed or missing cost/latency evidence.

When evidence is low:
- show "insufficient data to recommend safely" instead of forced recommendation.


## 9. What You Have Built So Far (Summary)

Already delivered:
- Multi-provider fetch and normalization for OpenAI + Anthropic.
- Unified analytics for spend, trends, top models, and breakdowns.
- Capacity/rate-limit awareness views.
- Model Intelligence and migration recommendation tab.
- AI Model Advisor tab and engine with interactive charts.
- Test coverage for advisor behavior and core analytics.

Current status of advisor:
- Uses real usage/spend as primary signal.
- Still uses static priors for model capability/latency/quality fallback.
- Not yet 100% telemetry-only for unseen models.


## 10. Next Implementation Increment (Recommended)
If you want strict non-static behavior now, implement this next:

1. Add `advisor_mode`:
- `telemetry_only`
- `hybrid` (current behavior)

2. Set default to `telemetry_only`.

3. In `telemetry_only`:
- candidates = only observed model-provider pairs in selected date range.
- remove fallback to static catalog for scoring.
- remove pricing guess for unseen models.

4. Display explicit coverage metrics:
- percent of spend covered by candidate evidence.
- count of models excluded for low sample size.

5. Add quality gate:
- minimum calls threshold per model (example: 100 calls in window).


## 11. Decisions Needed From You Before Implementation
To finalize the non-static advisor, confirm:

1. Minimum sample size per model:
- Suggested default: 100 calls in selected window.

2. If no model passes threshold:
- Return no recommendation.
- Or return best-effort low-confidence recommendation.

3. Cross-provider scope in telemetry-only mode:
- only providers with observed data in the current date range.
- or include historical windows automatically.

4. Budget handling:
- hard constraint (exclude over-budget models)
- or soft penalty (rank lower but still show).
