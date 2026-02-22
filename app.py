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
    apply_openai_rate_limits,
    build_actual_vs_forecast,
    build_dimension_summary,
    build_generic_metric_dimension_summary,
    build_line_item_cost_summary,
    build_model_summary,
    build_project_cost_summary,
    build_token_distribution,
    cost_reduction_trends,
    compute_kpis,
    extract_metric_columns,
    model_cost_breakdown,
    monthly_spend_trend,
    observed_capacity_metrics,
    provider_cost_usage_trend,
    project_current_month_total,
    spend_by_provider,
    top_models_by_cost_for_window,
    unified_cost_kpis,
)
from src.charts import (
    advisor_candidates_chart,
    advisor_category_score_chart,
    capacity_utilization_chart,
    cost_trend_chart,
    cost_reduction_chart,
    forecast_chart,
    generic_dimension_bar_chart,
    generic_metric_trend_chart,
    model_cost_chart,
    model_tokens_chart,
    monthly_budget_chart,
    project_cost_chart,
    provider_spend_chart,
    provider_metric_trend_chart,
    requests_by_model_chart,
    token_distribution_pie,
    top_models_cost_chart,
    usage_trend_chart,
)
from src.config import (
    CACHE_TTL_SECONDS,
    DUMMY_GPU_UTILIZATION_PCT,
    ENV_ANTHROPIC_ADMIN_KEY,
    ENV_GROQ_API_KEY,
    ENV_OPENAI_ADMIN_KEY,
    GROQ_MODEL_PRICING_PER_MILLION,
    GROQ_METRICS_BASE_URL,
    MODEL_PRICING_PER_MILLION,
    USAGE_ENDPOINT_LABELS,
)
from src.fetchers import build_time_window
from src.model_intelligence import (
    build_model_intelligence_table,
    list_provider_models,
    recommend_migration,
)
from src.model_advisor import build_usage_workload_profile, run_ai_model_advisor
from src.providers import (
    AnthropicProviderAdapter,
    AzureOpenAIProviderAdapter,
    GroqProviderAdapter,
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
    groq_api_key: str,
    groq_metrics_base_url: str,
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
        GroqProviderAdapter(groq_api_key, metrics_base_url=groq_metrics_base_url),
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
    col1.plotly_chart(usage_trend_chart(daily_usage, "Daily Token Usage"), width="stretch")
    col2.plotly_chart(cost_trend_chart(daily_cost, "Daily Reported Cost"), width="stretch")

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
                width="stretch",
            )
            tcol2.plotly_chart(
                cost_trend_chart(freq_cost, f"{label} Reported Cost"),
                width="stretch",
            )

    st.subheader("Model Analysis")
    mcol1, mcol2 = st.columns(2)
    mcol1.plotly_chart(model_cost_chart(model_summary), width="stretch")
    mcol2.plotly_chart(model_tokens_chart(model_summary), width="stretch")

    mcol3, mcol4 = st.columns(2)
    mcol3.plotly_chart(token_distribution_pie(token_distribution_df), width="stretch")
    mcol4.plotly_chart(requests_by_model_chart(model_summary), width="stretch")

    st.subheader("Billing Breakdown")
    bcol1, bcol2 = st.columns(2)
    bcol1.plotly_chart(project_cost_chart(project_cost_df), width="stretch")
    bcol2.dataframe(line_item_cost_df, width="stretch", hide_index=True)

    if group_by:
        st.subheader("Organization Dimensions (Completions)")
        for dimension in group_by:
            summary_df = build_dimension_summary(usage_df, dimension)
            st.markdown(f"**Top {dimension} by token usage**")
            st.dataframe(summary_df, width="stretch", hide_index=True)

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
        st.dataframe(usage_df, width="stretch", hide_index=True)

    with st.expander("Raw Cost Data", expanded=False):
        st.dataframe(cost_df, width="stretch", hide_index=True)



def render_provider_insights_tab(unified_df: pd.DataFrame) -> None:
    st.subheader("Unified Provider Insights")

    if unified_df.empty:
        st.info("No unified provider rows yet. Connect at least one provider.")
        return

    kpis = unified_cost_kpis(unified_df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total AI Spend", f"${kpis['total_spend_usd']:.2f}")
    k2.metric("Avg Cost / Inference", f"${kpis['avg_cost_per_inference']:.6f}")
    k3.metric("Active Providers", f"{int(kpis['active_providers'])}")
    k4.metric("Active Models", f"{int(kpis['active_models'])}")

    monthly_budget = st.number_input(
        "Monthly Budget (USD)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        help="Used for monthly spend vs budget visualization.",
    )

    monthly_df = monthly_spend_trend(unified_df)
    provider_df = spend_by_provider(unified_df)
    provider_daily = provider_cost_usage_trend(unified_df, freq="D")
    breakdown_df = model_cost_breakdown(unified_df, top_n=25)
    reduction_df = cost_reduction_trends(unified_df, window_days=7)

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        monthly_budget_chart(monthly_df, monthly_budget=monthly_budget),
        width="stretch",
    )
    c2.plotly_chart(provider_spend_chart(provider_df), width="stretch")

    c3, c4 = st.columns(2)
    c3.plotly_chart(
        provider_metric_trend_chart(
            provider_daily,
            metric="cost_usd",
            title="Provider Cost Trend (Daily)",
        ),
        width="stretch",
    )
    c4.plotly_chart(
        provider_metric_trend_chart(
            provider_daily,
            metric="calls",
            title="Provider Request Trend (Daily)",
        ),
        width="stretch",
    )

    st.markdown("### Top Models by Cost")
    window_tabs = st.tabs(["Last 24 Hours", "Last 7 Days", "Last 30 Days"])
    window_specs = [("24h", 1), ("7d", 7), ("30d", 30)]

    udf = unified_df.copy()
    udf["timestamp"] = pd.to_datetime(udf["timestamp"], utc=True)
    latest_ts = udf["timestamp"].max()
    udf["calls"] = pd.to_numeric(udf["calls"], errors="coerce").fillna(0.0)

    for tab, (label, days) in zip(window_tabs, window_specs):
        with tab:
            top_models_df = top_models_by_cost_for_window(unified_df, lookback_days=days, top_n=10)
            st.plotly_chart(top_models_cost_chart(top_models_df), width="stretch")

            window_start = latest_ts - pd.Timedelta(days=days)
            window_df = udf[udf["timestamp"] >= window_start]
            model_calls_df = (
                window_df.groupby(["model", "provider"], as_index=False)
                .agg(calls=("calls", "sum"), window_cost=("cost_usd", "sum"))
            )
            explanation_df = top_models_df.merge(model_calls_df, on=["model", "provider"], how="left")
            explanation_df["calls"] = explanation_df["calls"].fillna(0.0)
            explanation_df["window_cost"] = explanation_df["window_cost"].fillna(explanation_df["cost_usd"])
            total_window_cost = float(window_df["cost_usd"].sum()) if not window_df.empty else 0.0
            explanation_df["share_pct"] = (
                (explanation_df["window_cost"] / total_window_cost) * 100.0 if total_window_cost > 0 else 0.0
            )
            explanation_df["avg_cost_per_call"] = (
                explanation_df["window_cost"] / explanation_df["calls"].replace(0, pd.NA)
            ).fillna(0.0)

            if not explanation_df.empty:
                top_row = explanation_df.iloc[0]
                st.caption(
                    f"{label} leader: `{top_row['model']}` ({top_row['provider']}) "
                    f"accounts for {float(top_row['share_pct']):.1f}% of spend in this window."
                )

            st.dataframe(
                explanation_df.rename(
                    columns={
                        "model": "Model",
                        "provider": "Provider",
                        "cost_usd": "Cost (USD)",
                        "calls": "Calls",
                        "share_pct": "Spend Share (%)",
                        "avg_cost_per_call": "Avg Cost/Call",
                    }
                )[
                    [
                        "Model",
                        "Provider",
                        "Cost (USD)",
                        "Calls",
                        "Spend Share (%)",
                        "Avg Cost/Call",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

    st.markdown("### AI Cost Reduction Trend (7d vs Previous 7d)")
    st.plotly_chart(
        cost_reduction_chart(reduction_df, title="Cost Delta Percentage by Provider"),
        width="stretch",
    )
    st.dataframe(
        reduction_df.rename(
            columns={
                "provider": "Provider",
                "current_period_cost": "Current 7d Cost",
                "previous_period_cost": "Previous 7d Cost",
                "cost_delta": "Delta (USD)",
                "cost_delta_pct": "Delta (%)",
                "trend": "Trend",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Model Cost Breakdown")
    st.dataframe(breakdown_df, width="stretch", hide_index=True)



def render_capacity_limits_tab(
    unified_df: pd.DataFrame,
    openai_project_rate_limits_df: pd.DataFrame,
    *,
    bucket_width: str,
) -> None:
    st.subheader("Rate Limits & Capacity Awareness")

    if unified_df.empty:
        st.info("No provider usage rows available for capacity analysis.")
        return

    st.caption(
        "Observed TPM/RPM/TPD/RPD are derived from fetched usage buckets. OpenAI limits are API-fetched. "
        "Anthropic and Groq limit caps are not exposed via a comparable org limits API in this POC. "
        "GPU utilization shown here is a static dummy placeholder."
    )

    observed_df = observed_capacity_metrics(unified_df, bucket_width=bucket_width)
    capacity_df = apply_openai_rate_limits(observed_df, openai_project_rate_limits_df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tracked Model-Provider Pairs", f"{len(capacity_df)}")
    openai_with_limits = capacity_df[
        (capacity_df["provider"] == "openai") & capacity_df["tpm_limit"].notna()
    ]
    k2.metric("OpenAI Models With API Limits", f"{len(openai_with_limits)}")
    max_openai_util = (
        float(openai_with_limits["tpm_utilization_pct"].max())
        if not openai_with_limits.empty
        else 0.0
    )
    k3.metric("Highest OpenAI TPM Utilization", f"{max_openai_util:.1f}%")
    k4.metric("GPU Utilization (Dummy)", f"{DUMMY_GPU_UTILIZATION_PCT:.1f}%")

    st.plotly_chart(capacity_utilization_chart(capacity_df), width="stretch")

    display_df = capacity_df.copy()
    for col in ["max_tpm", "max_rpm", "tpm_utilization_pct", "rpm_utilization_pct"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
    for col in ["max_tpd", "max_rpd", "tpm_limit", "rpm_limit", "tpd_limit", "rpd_limit"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(0)

    st.dataframe(
        display_df.rename(
            columns={
                "provider": "Provider",
                "model": "Model",
                "max_tpm": "Observed TPM (Peak)",
                "max_rpm": "Observed RPM (Peak)",
                "max_tpd": "Observed TPD (Peak)",
                "max_rpd": "Observed RPD (Peak)",
                "tpm_limit": "TPM Limit",
                "rpm_limit": "RPM Limit",
                "tpd_limit": "TPD Limit",
                "rpd_limit": "RPD Limit",
                "tpm_utilization_pct": "TPM Utilization (%)",
                "rpm_utilization_pct": "RPM Utilization (%)",
                "status": "Status",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Raw OpenAI Project Rate Limits", expanded=False):
        if openai_project_rate_limits_df.empty:
            st.info("No OpenAI project rate limit rows returned.")
        else:
            st.dataframe(openai_project_rate_limits_df, width="stretch", hide_index=True)


def render_model_intelligence_tab(unified_df: pd.DataFrame) -> None:
    st.subheader("Model Intelligence")
    st.caption(
        "Profiles below combine observed usage/cost patterns with lightweight capability tags for migration planning."
    )

    if unified_df.empty:
        st.info("No unified provider rows available for model intelligence.")
        return

    intelligence_df = build_model_intelligence_table(unified_df)
    st.dataframe(intelligence_df, width="stretch", hide_index=True)

    model_pairs = list_provider_models(unified_df)
    if not model_pairs:
        st.info("No models available for migration simulation.")
        return

    st.markdown("### Migration Recommendation")
    option_map = {
        f"{provider.title()} · {model}": (provider, model)
        for provider, model in model_pairs
    }
    options = list(option_map.keys())
    default_source_idx = 0
    default_target_idx = 1 if len(options) > 1 else 0

    c1, c2 = st.columns(2)
    source_label = c1.selectbox("Source Model", options=options, index=default_source_idx)
    target_label = c2.selectbox("Target Model", options=options, index=default_target_idx)

    source_provider, source_model = option_map[source_label]
    target_provider, target_model = option_map[target_label]

    target_cpi_override = st.number_input(
        "Target CPI Override (optional USD/request)",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
        help="Use when the target model has no observed CPI and no pricing map entry.",
    )

    pricing_maps = {
        "openai": MODEL_PRICING_PER_MILLION,
        "groq": GROQ_MODEL_PRICING_PER_MILLION,
        "anthropic": {},
    }
    result = recommend_migration(
        unified_df,
        source_provider=source_provider,
        source_model=source_model,
        target_provider=target_provider,
        target_model=target_model,
        pricing_maps=pricing_maps,
        target_cpi_override=target_cpi_override if target_cpi_override > 0 else None,
    )

    if "error" in result:
        st.warning(result["error"])
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Source Spend", f"${result['source_cost']:.2f}")
    m2.metric("Projected Target Spend", f"${result['estimated_target_cost']:.2f}")
    m3.metric("Spend Delta", f"${result['cost_delta']:.2f}")
    m4.metric("Delta (%)", f"{result['cost_delta_pct']:+.2f}%")

    st.caption(f"Target CPI source: `{result['target_cpi_source']}`")
    st.markdown(f"**Recommendation:** {result['recommendation']}")

    src_profile = result["source_profile"]
    tgt_profile = result["target_profile"]
    st.markdown(
        f"**Source ({source_model})**: {src_profile['best_for']} | "
        f"Latency: {src_profile['latency']} | Reasoning: {src_profile['reasoning']}"
    )
    st.markdown(
        f"**Target ({target_model})**: {tgt_profile['best_for']} | "
        f"Latency: {tgt_profile['latency']} | Reasoning: {tgt_profile['reasoning']}"
    )


def render_ai_model_advisor_tab(unified_df: pd.DataFrame) -> None:
    st.subheader("AI Model Advisor")
    st.caption(
        "Personalized model recommendations using observed usage, token mix, spend, and workload constraints. "
        "Current scope: OpenAI + Anthropic."
    )

    if unified_df.empty:
        st.info("No provider rows available for advisor analysis.")
        return

    advisor_df = unified_df[unified_df["provider"].isin(["openai", "anthropic"])].copy()
    if advisor_df.empty:
        st.info("Advisor currently supports OpenAI and Anthropic rows only.")
        return

    profile = build_usage_workload_profile(advisor_df)
    provider_label_map = {"openai": "OpenAI", "anthropic": "Anthropic"}

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Current Model", str(profile["current_model"]))
    current_provider = str(profile["current_provider"]).lower()
    k2.metric("Current Provider", provider_label_map.get(current_provider, current_provider.title()))
    k3.metric("Calls / Day (Avg)", f"{profile['avg_calls_per_day']:.1f}")
    k4.metric("Observed CPI", f"${profile['current_cpi_usd']:.6f}")
    k5.metric("Projected Monthly Spend", f"${profile['estimated_monthly_spend_usd']:.2f}")

    st.caption(
        "Advisor uses historical aggregates only; prompt/response text and task labels are not available from these APIs."
    )

    objective_options = {
        "balanced": "Balanced",
        "min_cost": "Minimize Cost",
        "max_quality": "Maximize Quality",
    }
    complexity_options = ["low", "medium", "high", "very_high"]
    task_options = {
        "general_assistant": "General Assistant",
        "classification_routing": "Classification / Routing",
        "coding_assistant": "Coding Assistant",
        "document_analysis": "Document Analysis",
        "multimodal": "Multimodal",
    }

    c1, c2, c3 = st.columns(3)
    objective = c1.selectbox(
        "Optimization Objective",
        options=list(objective_options.keys()),
        format_func=lambda key: objective_options[key],
        index=0,
    )
    complexity_level = c2.selectbox(
        "Workload Complexity",
        options=complexity_options,
        format_func=lambda value: value.replace("_", " ").title(),
        index=1,
    )
    primary_task = c3.selectbox(
        "Primary Task Pattern",
        options=list(task_options.keys()),
        format_func=lambda key: task_options[key],
        index=0,
    )

    d1, d2, d3 = st.columns(3)
    max_latency_default = int(min(2500, max(150, round(profile["avg_total_tokens_per_call"] * 0.3))))
    max_latency_ms = d1.slider(
        "Max Latency (ms)",
        min_value=100,
        max_value=3000,
        value=max_latency_default,
        step=50,
    )
    budget_default = float(max(50.0, round(profile["estimated_monthly_spend_usd"], 2)))
    monthly_budget_usd = d2.number_input(
        "Monthly Budget (USD)",
        min_value=0.0,
        value=budget_default,
        step=50.0,
        help="Set to 0 to disable budget pressure in scoring.",
    )
    allow_cross_provider = d3.checkbox(
        "Allow Cross-Provider Recommendation",
        value=True,
        help="When disabled, advisor keeps recommendations in the current provider family.",
    )

    required_capabilities = st.multiselect(
        "Required Capabilities",
        options=["reasoning", "code", "vision", "long_context", "research"],
        default=[],
    )
    preferred_providers = st.multiselect(
        "Providers to Consider",
        options=["openai", "anthropic"],
        default=["openai", "anthropic"],
    )

    if not preferred_providers:
        st.warning("Select at least one provider to run advisor scoring.")
        return

    result = run_ai_model_advisor(
        advisor_df,
        objective=objective,
        complexity_level=complexity_level,
        max_latency_ms=max_latency_ms,
        monthly_budget_usd=monthly_budget_usd if monthly_budget_usd > 0 else None,
        required_capabilities=tuple(required_capabilities),
        primary_task=primary_task,
        preferred_providers=tuple(preferred_providers),
        allow_cross_provider=allow_cross_provider,
    )

    if "error" in result:
        st.warning(str(result["error"]))
        return

    primary = result["primary_recommendation"]
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Recommended Category", str(primary["recommended_category"]))
    p2.metric("Recommended Model", str(primary["recommended_model"]))
    p3.metric("Provider", str(primary["provider"]))
    p4.metric("Confidence Score", f"{float(primary['confidence_score']):.1f}%")
    p5.metric("Estimated Monthly Cost", f"${float(primary['estimated_monthly_cost_usd']):.2f}")

    s1, s2 = st.columns(2)
    s1.metric("Estimated Monthly Savings", f"${float(primary['estimated_monthly_savings_usd']):.2f}")
    s2.metric("Estimated Savings (%)", f"{float(primary['estimated_monthly_savings_pct']):+.1f}%")

    st.plotly_chart(
        advisor_category_score_chart(result.get("category_scores", {})),
        width="stretch",
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("### Recommendation Reasoning")
        for message in primary.get("reasoning", []):
            st.write(f"- {message}")
    with rc2:
        st.markdown("### Trade-offs")
        for message in primary.get("trade_offs", []):
            st.write(f"- {message}")

    candidates_df = result.get("candidates_df", pd.DataFrame())
    if not candidates_df.empty:
        st.markdown("### Candidate Comparison")
        st.plotly_chart(advisor_candidates_chart(candidates_df), width="stretch")

        display_columns = [
            "category",
            "model",
            "provider",
            "advisor_score",
            "estimated_monthly_cost_usd",
            "estimated_cpi_usd",
            "latency_ms",
            "capability_fit",
            "cost_score",
            "cost_source",
        ]
        available_cols = [col for col in display_columns if col in candidates_df.columns]
        st.dataframe(
            candidates_df[available_cols].rename(
                columns={
                    "category": "Category",
                    "model": "Model",
                    "provider": "Provider",
                    "advisor_score": "Advisor Score (%)",
                    "estimated_monthly_cost_usd": "Estimated Monthly Cost (USD)",
                    "estimated_cpi_usd": "Estimated CPI (USD)",
                    "latency_ms": "Latency (ms)",
                    "capability_fit": "Capability Fit (%)",
                    "cost_score": "Cost Score (%)",
                    "cost_source": "Cost Source",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    alternatives = result.get("alternatives", [])
    if alternatives:
        st.markdown("### Alternatives")
        st.dataframe(pd.DataFrame(alternatives), width="stretch", hide_index=True)

    with st.expander("Advisor Inputs & Assumptions", expanded=False):
        st.json(result.get("inputs_used", {}))
        st.caption(
            "Model pricing/performance metadata in advisor catalog is heuristic and should be validated "
            "with your own quality and latency benchmarks before production migration."
        )


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
        width="stretch",
    )
    col2.plotly_chart(
        generic_dimension_bar_chart(
            dimension_summary,
            dimension=selected_dimension,
            metric=metric,
            title=f"Top {selected_dimension.replace('_', ' ').title()} by {metric.replace('_', ' ').title()}",
        ),
        width="stretch",
    )

    dataset_csv = dataset_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Endpoint CSV",
        data=dataset_csv,
        file_name=f"openai_usage_{selected_dataset}.csv",
        mime="text/csv",
    )

    with st.expander("Raw Endpoint Data", expanded=False):
        st.dataframe(dataset_df, width="stretch", hide_index=True)



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
        width="stretch",
    )
    fcol2.plotly_chart(
        forecast_chart(
            token_forecast,
            metric="total_tokens",
            title="Token Forecast (30 Days)",
        ),
        width="stretch",
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
    default_anthropic_key = os.getenv(ENV_ANTHROPIC_ADMIN_KEY, "")
    default_groq_key = os.getenv(ENV_GROQ_API_KEY, "")
    filters = render_sidebar(
        default_api_key=default_openai_key,
        default_anthropic_api_key=default_anthropic_key,
        default_groq_api_key=default_groq_key,
        default_groq_metrics_base_url=GROQ_METRICS_BASE_URL,
    )

    if filters.api_key:
        st.session_state["openai_admin_key"] = filters.api_key

    if not (
        filters.api_key
        or filters.anthropic_api_key
        or filters.groq_api_key
        or filters.azure_openai_api_key
    ):
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
        bool(filters.groq_api_key),
        filters.groq_metrics_base_url,
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
                    groq_api_key=filters.groq_api_key,
                    groq_metrics_base_url=filters.groq_metrics_base_url,
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
    openai_project_rate_limits_df = openai_payload.get("project_rate_limits_df", pd.DataFrame())

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
            "Capacity & Limits",
            "Model Intelligence",
            "AI Model Advisor",
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
        render_capacity_limits_tab(
            unified_df,
            openai_project_rate_limits_df,
            bucket_width=filters.bucket_width,
        )
    with tabs[3]:
        render_model_intelligence_tab(unified_df)
    with tabs[4]:
        render_ai_model_advisor_tab(unified_df)
    with tabs[5]:
        render_usage_explorer_tab(aux_usage_frames)
    with tabs[6]:
        if usage_df.empty and cost_df.empty:
            st.info("Forecasts currently run on OpenAI usage/cost series.")
        else:
            render_forecasts_tab(usage_df, cost_df)
    # with tabs[4]:
    #     render_future_roadmap_tab()


if __name__ == "__main__":
    main()
