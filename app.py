"""Streamlit entrypoint for LLM Cost Analysis Module POC."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.analytics import (
    aggregate_cost,
    aggregate_generic_usage,
    aggregate_usage,
    build_actual_vs_forecast,
    build_dimension_summary,
    build_generic_metric_dimension_summary,
    build_line_item_cost_summary,
    build_model_summary,
    build_project_cost_summary,
    build_token_distribution,
    compute_kpis,
    extract_metric_columns,
    model_cost_breakdown,
    monthly_spend_trend,
    project_current_month_total,
    spend_by_provider,
    top_models_by_cost,
)
from src.charts import (
    cost_trend_chart,
    forecast_chart,
    generic_dimension_bar_chart,
    generic_metric_trend_chart,
    model_cost_chart,
    model_tokens_chart,
    monthly_budget_chart,
    project_cost_chart,
    provider_spend_chart,
    requests_by_model_chart,
    token_distribution_pie,
    top_models_cost_chart,
    usage_trend_chart,
)
from src.config import CACHE_TTL_SECONDS, ENV_OPENAI_ADMIN_KEY, USAGE_ENDPOINT_LABELS
from src.fetchers import build_time_window
from src.providers import (
    AnthropicProviderAdapter,
    AzureOpenAIProviderAdapter,
    OpenAIProviderAdapter,
    empty_unified_df,
)
from src.ui import (
    apply_app_styles,
    render_forecast_disclaimer,
    render_header,
    render_kpi_cards,
    render_limitations,
    render_sidebar,
)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_dashboard_data(
    openai_api_key: str,
    anthropic_api_key: str,
    azure_openai_api_key: str,
    azure_openai_endpoint: str,
    start_time: int,
    end_time: int,
    bucket_width: str,
    group_by: tuple[str, ...],
) -> dict[str, Any]:
    adapters = [
        OpenAIProviderAdapter(openai_api_key),
        AnthropicProviderAdapter(anthropic_api_key),
        AzureOpenAIProviderAdapter(azure_openai_api_key, azure_openai_endpoint),
    ]

    provider_results: dict[str, dict[str, Any]] = {}
    unified_frames: list[pd.DataFrame] = []

    for adapter in adapters:
        if not adapter.is_configured():
            continue

        result = adapter.fetch(
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=group_by,
        )
        provider_results[result.provider] = {
            "raw_payload": result.raw_payload,
            "endpoint_errors": result.endpoint_errors,
            "notices": result.notices,
            "unified_df": result.unified_df,
        }
        if not result.unified_df.empty:
            unified_frames.append(result.unified_df)

    unified_df = pd.concat(unified_frames, ignore_index=True) if unified_frames else empty_unified_df()

    return {
        "provider_results": provider_results,
        "unified_df": unified_df,
    }



def render_overview_tab(usage_df: pd.DataFrame, cost_df: pd.DataFrame, group_by: tuple[str, ...]) -> None:
    model_summary = build_model_summary(usage_df)
    token_distribution_df = build_token_distribution(model_summary)
    project_cost_df = build_project_cost_summary(cost_df)
    line_item_cost_df = build_line_item_cost_summary(cost_df)

    kpis = compute_kpis(usage_df, cost_df, model_summary)
    render_kpi_cards(kpis)

    st.caption(
        f"Estimated model spend: ${kpis['estimated_cost']:.2f} | "
        f"Reported spend: ${kpis['reported_cost']:.2f} | "
        f"Delta: ${kpis['reconciliation_delta']:.2f}"
    )

    st.subheader("Daily Trends")
    daily_usage = aggregate_usage(usage_df, "D")
    daily_cost = aggregate_cost(cost_df, "D")
    col1, col2 = st.columns(2)
    col1.plotly_chart(usage_trend_chart(daily_usage, "Daily Token Usage"), use_container_width=True)
    col2.plotly_chart(cost_trend_chart(daily_cost, "Daily Reported Cost"), use_container_width=True)

    st.subheader("Weekly / Monthly / Yearly")
    trend_tabs = st.tabs(["Weekly", "Monthly", "Yearly"])
    tab_specs = [("Weekly", "W"), ("Monthly", "M"), ("Yearly", "Y")]
    for tab, (label, freq) in zip(trend_tabs, tab_specs):
        with tab:
            freq_usage = aggregate_usage(usage_df, freq)
            freq_cost = aggregate_cost(cost_df, freq)
            tcol1, tcol2 = st.columns(2)
            tcol1.plotly_chart(
                usage_trend_chart(freq_usage, f"{label} Token Usage"),
                use_container_width=True,
            )
            tcol2.plotly_chart(
                cost_trend_chart(freq_cost, f"{label} Reported Cost"),
                use_container_width=True,
            )

    st.subheader("Model Analysis")
    mcol1, mcol2 = st.columns(2)
    mcol1.plotly_chart(model_cost_chart(model_summary), use_container_width=True)
    mcol2.plotly_chart(model_tokens_chart(model_summary), use_container_width=True)

    mcol3, mcol4 = st.columns(2)
    mcol3.plotly_chart(token_distribution_pie(token_distribution_df), use_container_width=True)
    mcol4.plotly_chart(requests_by_model_chart(model_summary), use_container_width=True)

    st.subheader("Billing Breakdown")
    bcol1, bcol2 = st.columns(2)
    bcol1.plotly_chart(project_cost_chart(project_cost_df), use_container_width=True)
    bcol2.dataframe(line_item_cost_df, use_container_width=True, hide_index=True)

    if group_by:
        st.subheader("Organization Dimensions (Completions)")
        for dimension in group_by:
            summary_df = build_dimension_summary(usage_df, dimension)
            st.markdown(f"**Top {dimension} by token usage**")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Export & Raw Data")
    usage_csv = usage_df.to_csv(index=False).encode("utf-8")
    cost_csv = cost_df.to_csv(index=False).encode("utf-8")

    dcol1, dcol2 = st.columns(2)
    dcol1.download_button(
        label="Download Usage CSV",
        data=usage_csv,
        file_name="openai_usage_completions.csv",
        mime="text/csv",
    )
    dcol2.download_button(
        label="Download Cost CSV",
        data=cost_csv,
        file_name="openai_costs.csv",
        mime="text/csv",
    )

    with st.expander("Raw Completions Usage Data", expanded=False):
        st.dataframe(usage_df, use_container_width=True, hide_index=True)

    with st.expander("Raw Cost Data", expanded=False):
        st.dataframe(cost_df, use_container_width=True, hide_index=True)



def render_provider_insights_tab(unified_df: pd.DataFrame) -> None:
    st.subheader("Unified Provider Insights")

    if unified_df.empty:
        st.info("No unified provider rows yet. Connect at least one provider.")
        return

    monthly_budget = st.number_input(
        "Monthly Budget (USD)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        help="Used for monthly spend vs budget visualization.",
    )

    monthly_df = monthly_spend_trend(unified_df)
    provider_df = spend_by_provider(unified_df)
    top_models_df = top_models_by_cost(unified_df, top_n=10)
    breakdown_df = model_cost_breakdown(unified_df, top_n=25)

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        monthly_budget_chart(monthly_df, monthly_budget=monthly_budget),
        use_container_width=True,
    )
    c2.plotly_chart(provider_spend_chart(provider_df), use_container_width=True)

    st.plotly_chart(top_models_cost_chart(top_models_df), use_container_width=True)

    st.markdown("### Model Cost Breakdown")
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)



def render_usage_explorer_tab(aux_usage_frames: dict[str, pd.DataFrame]) -> None:
    st.subheader("Additional OpenAI Usage Endpoints")
    st.caption(
        "Inspect Images, Moderations, Audio, Vector Stores, and Code Interpreter usage with "
        "daily rollups and dimension breakdowns."
    )

    datasets = list(aux_usage_frames.keys())
    if not datasets:
        st.info("No additional usage endpoint data available.")
        return

    selected_dataset = st.selectbox(
        "Usage Endpoint",
        options=datasets,
        format_func=lambda key: USAGE_ENDPOINT_LABELS.get(key, key.replace("_", " ").title()),
    )

    dataset_df = aux_usage_frames.get(selected_dataset, pd.DataFrame())
    if dataset_df.empty:
        st.info("No rows returned for this endpoint and date range.")
        return

    metric_columns = extract_metric_columns(dataset_df)
    if not metric_columns:
        st.info("No numeric metrics available for this endpoint response.")
        return

    metric = st.selectbox(
        "Metric",
        options=metric_columns,
        format_func=lambda name: name.replace("_", " ").title(),
    )

    dimension_candidates = [
        "project_id",
        "model",
        "user_id",
        "api_key_id",
        "size",
        "source",
        "service_tier",
        "batch",
    ]
    available_dimensions = [d for d in dimension_candidates if d in dataset_df.columns]
    if not available_dimensions:
        available_dimensions = ["dataset"]
    selected_dimension = st.selectbox(
        "Breakdown Dimension",
        options=available_dimensions,
        format_func=lambda name: name.replace("_", " ").title(),
    )

    daily_rollup = aggregate_generic_usage(dataset_df, "D")
    dimension_summary = build_generic_metric_dimension_summary(
        dataset_df,
        metric=metric,
        dimension=selected_dimension,
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        generic_metric_trend_chart(
            daily_rollup,
            metric=metric,
            title=f"{USAGE_ENDPOINT_LABELS.get(selected_dataset, selected_dataset)} Daily {metric.replace('_', ' ').title()}",
        ),
        use_container_width=True,
    )
    col2.plotly_chart(
        generic_dimension_bar_chart(
            dimension_summary,
            dimension=selected_dimension,
            metric=metric,
            title=f"Top {selected_dimension.replace('_', ' ').title()} by {metric.replace('_', ' ').title()}",
        ),
        use_container_width=True,
    )

    dataset_csv = dataset_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Endpoint CSV",
        data=dataset_csv,
        file_name=f"openai_usage_{selected_dataset}.csv",
        mime="text/csv",
    )

    with st.expander("Raw Endpoint Data", expanded=False):
        st.dataframe(dataset_df, use_container_width=True, hide_index=True)



def render_forecasts_tab(usage_df: pd.DataFrame, cost_df: pd.DataFrame) -> None:
    st.subheader("Baseline Forecasts")
    render_forecast_disclaimer()

    daily_cost = aggregate_cost(cost_df, "D")
    daily_usage = aggregate_usage(usage_df, "D")

    cost_forecast = build_actual_vs_forecast(
        daily_cost,
        value_column="amount",
        horizon_days=30,
    )
    token_forecast = build_actual_vs_forecast(
        daily_usage[["period", "total_tokens"]] if not daily_usage.empty else pd.DataFrame(),
        value_column="total_tokens",
        horizon_days=30,
    )

    cost_projection = project_current_month_total(daily_cost, value_column="amount")
    token_projection = project_current_month_total(
        daily_usage[["period", "total_tokens"]] if not daily_usage.empty else pd.DataFrame(),
        value_column="total_tokens",
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cost MTD", f"${cost_projection['actual_to_date']:.2f}")
    k2.metric("Cost Month-End (Avg)", f"${cost_projection['avg_daily_projection']:.2f}")
    k3.metric("Cost Month-End (Linear)", f"${cost_projection['linear_projection']:.2f}")
    k4.metric("Days Remaining", f"{int(cost_projection['days_remaining'])}")

    t1, t2, t3 = st.columns(3)
    t1.metric("Tokens MTD", f"{int(token_projection['actual_to_date']):,}")
    t2.metric("Tokens Month-End (Avg)", f"{int(token_projection['avg_daily_projection']):,}")
    t3.metric("Tokens Month-End (Linear)", f"{int(token_projection['linear_projection']):,}")

    fcol1, fcol2 = st.columns(2)
    fcol1.plotly_chart(
        forecast_chart(
            cost_forecast,
            metric="amount",
            title="Cost Forecast (30 Days)",
        ),
        use_container_width=True,
    )
    fcol2.plotly_chart(
        forecast_chart(
            token_forecast,
            metric="total_tokens",
            title="Token Forecast (30 Days)",
        ),
        use_container_width=True,
    )



def render_future_roadmap_tab() -> None:
    st.subheader("Future Roadmap")
    st.markdown(
        """
### Vision
Evolve this OpenAI-focused POC into a multi-provider cost intelligence platform with a unified analytics layer.

### Phase 1: Multi-Provider Connectors
- Add provider adapters for Anthropic, Azure OpenAI, AWS Bedrock, and Google Vertex/Gemini.
- Keep provider-specific authentication isolated in connector modules.

### Phase 2: Unified Data Model
- Normalize usage and billing into a shared schema: provider, project, model, tokens, requests, cost, currency, timestamp.
- Preserve provider-native fields in optional metadata columns.

### Phase 3: Forecasting and Insights
- Add seasonal forecasting and anomaly detection.
- Provide budget risk flags and projected month-end spend by provider/project/model.

### Phase 4: Unified Dashboard and Governance
- Single-pane dashboard with cross-provider filters and drill-downs.
- Add role-based access, auditability, and scheduled reporting.
        """
    )



def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="LLM Cost Analysis Module", layout="wide")
    apply_app_styles()
    render_header()

    default_openai_key = st.session_state.get("openai_admin_key") or os.getenv(ENV_OPENAI_ADMIN_KEY, "")
    filters = render_sidebar(default_api_key=default_openai_key)

    if filters.api_key:
        st.session_state["openai_admin_key"] = filters.api_key

    if not (filters.api_key or filters.anthropic_api_key or filters.azure_openai_api_key):
        st.warning("Enter at least one provider credential in the sidebar to continue.")
        render_limitations()
        return

    if filters.date_from > filters.date_to:
        st.error("Invalid date range: `From` must be before or equal to `To`.")
        return

    start_time, end_time = build_time_window(filters.date_from, filters.date_to)

    filter_signature = (
        filters.date_from.isoformat(),
        filters.date_to.isoformat(),
        filters.bucket_width,
        filters.group_by,
        bool(filters.api_key),
        bool(filters.anthropic_api_key),
        bool(filters.azure_openai_api_key),
        filters.azure_openai_endpoint,
    )

    has_cached_data = "dashboard_payload" in st.session_state
    has_filter_change = has_cached_data and st.session_state.get("filter_signature") != filter_signature

    if has_filter_change and not filters.fetch_clicked:
        st.warning("Filters changed. Click `Fetch / Refresh Data` to reload charts.")

    if filters.fetch_clicked:
        with st.spinner("Fetching provider usage and billing data..."):
            try:
                payload = fetch_dashboard_data(
                    openai_api_key=filters.api_key,
                    anthropic_api_key=filters.anthropic_api_key,
                    azure_openai_api_key=filters.azure_openai_api_key,
                    azure_openai_endpoint=filters.azure_openai_endpoint,
                    start_time=start_time,
                    end_time=end_time,
                    bucket_width=filters.bucket_width,
                    group_by=filters.group_by,
                )
            except Exception as exc:
                st.error(f"Unexpected error while fetching data: {exc}")
                st.stop()

        st.session_state["dashboard_payload"] = payload
        st.session_state["filter_signature"] = filter_signature

    if not has_cached_data and not filters.fetch_clicked:
        st.info("Configure filters and click `Fetch / Refresh Data`.")
        render_limitations()
        return

    payload = st.session_state["dashboard_payload"]
    provider_results = payload.get("provider_results", {})
    unified_df = payload.get("unified_df", empty_unified_df())

    openai_payload = provider_results.get("openai", {}).get("raw_payload", {})
    usage_df = openai_payload.get("usage_df", pd.DataFrame())
    cost_df = openai_payload.get("cost_df", pd.DataFrame())
    aux_usage_frames = openai_payload.get("aux_usage_frames", {})

    if provider_results:
        with st.expander("Provider notices", expanded=False):
            for provider, result in provider_results.items():
                display_name = provider.replace("_", " ").title()
                for message in result.get("notices", []):
                    st.info(f"{display_name}: {message}")
                for endpoint_name, message in result.get("endpoint_errors", {}).items():
                    label = USAGE_ENDPOINT_LABELS.get(endpoint_name, endpoint_name.replace("_", " ").title())
                    st.warning(f"{display_name} - {label}: {message}")

    if unified_df.empty and usage_df.empty and cost_df.empty:
        st.warning("No accessible usage/cost data found for selected providers and date range.")
        render_limitations()
        return

    tabs = st.tabs(
        [
            "Overview",
            "Provider Insights",
            "Usage Explorer",
            "Forecasts (beta)",
            # "Future Roadmap",
        ]
    )
    with tabs[0]:
        if usage_df.empty and cost_df.empty:
            st.info("OpenAI detailed overview requires OpenAI data access.")
        else:
            render_overview_tab(usage_df, cost_df, filters.group_by)
        render_limitations()
    with tabs[1]:
        render_provider_insights_tab(unified_df)
    with tabs[2]:
        render_usage_explorer_tab(aux_usage_frames)
    with tabs[3]:
        if usage_df.empty and cost_df.empty:
            st.info("Forecasts currently run on OpenAI usage/cost series.")
        else:
            render_forecasts_tab(usage_df, cost_df)
    # with tabs[4]:
    #     render_future_roadmap_tab()


if __name__ == "__main__":
    main()
