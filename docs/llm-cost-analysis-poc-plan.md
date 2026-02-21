# LLM Cost Analysis Module (POC) — Detailed Implementation Plan

## 1. Objective
Build a Streamlit POC that lets a user connect their OpenAI account (via Admin API key), fetch organization usage/cost data, and analyze model-level usage, token consumption, and billing trends with interactive charts.

## 2. Target outcomes
- Replicate key OpenAI dashboard-style insights using official APIs.
- Support daily/weekly/monthly/yearly trend exploration.
- Provide model-wise comparisons for token usage and estimated costs.
- Expose project/user/api-key dimensions when present in usage data.
- Keep implementation simple, readable, and practical for a POC.

## 3. Scope
### In scope
- Admin key-based authorization in Streamlit sidebar.
- API integration with:
  - `GET /v1/organization/usage/completions`
  - `GET /v1/organization/costs`
- Data normalization to pandas DataFrames.
- KPI cards and interactive Plotly charts.
- CSV export for usage and cost datasets.

### Out of scope
- OAuth-like OpenAI account connect flow.
- Raw prompt/response text retrieval.
- True per-request billing from OpenAI org APIs.
- Persistent database layer.

## 4. Authorization design
- User enters `OPENAI_ADMIN_KEY` in sidebar password field.
- Key is stored only in Streamlit session state.
- No disk persistence and no logging of the key.
- Optional `.env` fallback for local single-owner demo.

## 5. API integration plan
### 5.1 Usage endpoint
- Endpoint: `/v1/organization/usage/completions`
- Inputs:
  - `start_time`, `end_time`
  - `bucket_width` (`1d` default, `1h` optional)
  - `group_by` dimensions (`model`, `project_id`, `user_id`, `api_key_id`)
- Output fields consumed:
  - `input_tokens`, `output_tokens`, `input_cached_tokens`
  - `num_model_requests`
  - optional dimensions (`model`, `project_id`, `user_id`, `api_key_id`)

### 5.2 Costs endpoint
- Endpoint: `/v1/organization/costs`
- Inputs:
  - `start_time`, `end_time`, `bucket_width`
- Output fields consumed:
  - `amount.value`, `amount.currency`
  - `line_item`, `project_id`

### 5.3 Reliability handling
- Shared GET client with:
  - retry/backoff for `429/5xx`
  - cursor-based pagination (`next_page`)
  - timeout and structured error handling

## 6. Data model and transformations
### 6.1 Usage DataFrame
Columns:
- `bucket_start`, `bucket_end`
- `model`, `project_id`, `user_id`, `api_key_id`
- `input_tokens`, `output_tokens`, `cached_input_tokens`
- `input_audio_tokens`, `output_audio_tokens`
- `requests`, `total_tokens`

### 6.2 Cost DataFrame
Columns:
- `bucket_start`, `bucket_end`
- `project_id`, `line_item`
- `amount`, `currency`

### 6.3 Normalization rules
- Convert timestamps to UTC datetime.
- Coerce numeric fields with safe defaults (`0`).
- Fill missing dimensions with `unknown` / `unassigned`.

## 7. Analytics plan
### 7.1 KPI cards
- Reported total cost
- Total input tokens
- Total output tokens
- Total requests
- Active model count
- Average reported cost per request

### 7.2 Time aggregates
- Daily (`D`)
- Weekly (`W`)
- Monthly (`M`)
- Yearly (`Y`)

Metrics per period:
- input/output/total tokens
- request counts
- total reported cost

### 7.3 Model-level analysis
- Token usage by model
- Request volume by model
- Estimated model cost using token totals + local pricing map

### 7.4 Billing breakdown
- Top projects by cost
- Top line items by cost

### 7.5 Reconciliation
- Compare:
  - total estimated model cost
  - total reported org cost
- show delta as diagnostic signal

## 8. Chart plan (interactive)
Use Plotly Express for all visualizations.

Charts:
- Daily token trend (area)
- Daily cost trend (line)
- Weekly/monthly/yearly token + cost trends
- Model-wise estimated cost (bar)
- Model token distribution (stacked bar)
- Overall token split input vs output (donut)
- Requests by model (bar)
- Top projects by cost (bar)

## 9. Streamlit UX flow
1. User opens app.
2. Adds admin key in sidebar.
3. Selects date range, interval, group-by dimensions.
4. Clicks `Fetch / Refresh Data`.
5. App loads and caches API data.
6. Dashboard renders KPIs + trends + model/billing insights.
7. User can export CSV and clear cached session.

## 10. Security and privacy
- No key persistence to repository files.
- No key output in logs/UI.
- Session clear button flushes cached data.
- Explicit warning that admin scope is required.

## 11. Testing strategy
### Unit tests
- Usage payload normalization
- Cost payload normalization
- Time-window conversion
- KPI/aggregation computations
- Pricing-based estimation logic

### Mocked integration tests
- Pagination handling
- Query parameter mapping
- API error handling branches

## 12. Acceptance criteria
- User can connect with admin key and fetch data successfully.
- Dashboard renders interactive daily/weekly/monthly/yearly views.
- Model-wise usage and estimated cost views are visible.
- Reported cost from costs endpoint is displayed.
- CSV export works for usage and cost datasets.
- Missing unsupported metrics are clearly explained in UI.

## 13. Implementation sequence
1. Scaffold project and dependencies.
2. Build API client (auth, retries, pagination).
3. Implement fetchers and time-window conversion.
4. Build payload transformers to DataFrames.
5. Implement analytics functions.
6. Add Plotly chart builders.
7. Build Streamlit UI and app wiring.
8. Add tests and run verification.
9. Finalize docs (`README` + this plan).

## 14. Assumptions
- User has an OpenAI Admin API key for org-level endpoints.
- Endpoint payload shape matches current OpenAI API docs.
- Local pricing map may drift and must be maintained manually.
- POC runs in UTC-backed aggregation with local browser rendering.
