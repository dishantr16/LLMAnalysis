"""UI helpers for Streamlit layout and controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import streamlit as st

from src.config import DEFAULT_BUCKET_WIDTH, DEFAULT_LOOKBACK_DAYS, DIMENSION_OPTIONS


@dataclass(frozen=True)
class DashboardFilters:
    api_key: str
    anthropic_api_key: str
    groq_api_key: str
    groq_metrics_base_url: str
    azure_openai_api_key: str
    azure_openai_endpoint: str
    date_from: date
    date_to: date
    bucket_width: str
    group_by: tuple[str, ...]
    fetch_clicked: bool


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1250px;
            }
            [data-testid="stSidebar"] {
                border-right: 1px solid #e5e7eb;
            }
            [data-testid="metric-container"] {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 10px 12px;
                background: #f8fafc;
            }
            .main h1, .main h2, .main h3 {
                letter-spacing: -0.01em;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.title("LLM Cost Analysis Module")
    st.caption(
        "Multi-provider usage and cost analytics POC (OpenAI, Anthropic, Groq) with interactive "
        "insights, baseline forecasting, and migration-oriented model intelligence."
    )


def render_sidebar(
    default_api_key: str = "",
    *,
    default_anthropic_api_key: str = "",
    default_groq_api_key: str = "",
    default_groq_metrics_base_url: str = "https://api.groq.com/v1/metrics/prometheus",
) -> DashboardFilters:
    st.sidebar.header("Connection & Filters")

    api_key = st.sidebar.text_input(
        "OpenAI Admin API Key",
        value=default_api_key,
        type="password",
        help="Use an Admin key to access organization usage and costs endpoints.",
    )

    with st.sidebar.expander("Additional Providers", expanded=False):
        anthropic_api_key = st.text_input(
            "Anthropic API Key",
            value=default_anthropic_api_key,
            type="password",
            help="Used for Anthropic usage/cost report APIs.",
        )
        groq_api_key = st.text_input(
            "Groq API Key",
            value=default_groq_api_key,
            type="password",
            help="Used for Groq metrics API.",
        )
        groq_metrics_base_url = st.text_input(
            "Groq Metrics Base URL",
            value=default_groq_metrics_base_url,
            help="Default: https://api.groq.com/v1/metrics/prometheus",
        )
        azure_openai_api_key = st.text_input(
            "Azure OpenAI API Key",
            value="",
            type="password",
            help="Optional scaffold input.",
        )
        azure_openai_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            value="",
            help="Example: https://your-resource.openai.azure.com/",
        )

    today = date.today()
    default_start = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    date_from = st.sidebar.date_input("From", value=default_start)
    date_to = st.sidebar.date_input("To", value=today)

    bucket_width = st.sidebar.selectbox(
        "Bucket Width",
        options=["1d", "1h"],
        index=0 if DEFAULT_BUCKET_WIDTH == "1d" else 1,
        help="1d is recommended for this POC.",
    )

    group_by = st.sidebar.multiselect(
        "Group usage by",
        options=DIMENSION_OPTIONS,
        default=["model"],
        help="These dimensions apply to usage endpoint aggregation.",
    )

    fetch_clicked = st.sidebar.button("Fetch / Refresh Data", type="primary")

    if st.sidebar.button("Clear Cached Session"):
        st.cache_data.clear()
        for key in ["dashboard_payload", "filter_signature", "openai_admin_key"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    return DashboardFilters(
        api_key=api_key.strip(),
        anthropic_api_key=anthropic_api_key.strip(),
        groq_api_key=groq_api_key.strip(),
        groq_metrics_base_url=groq_metrics_base_url.strip(),
        azure_openai_api_key=azure_openai_api_key.strip(),
        azure_openai_endpoint=azure_openai_endpoint.strip(),
        date_from=date_from,
        date_to=date_to,
        bucket_width=bucket_width,
        group_by=tuple(group_by),
        fetch_clicked=fetch_clicked,
    )


def render_kpi_cards(kpis: dict[str, object]) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Reported Cost", f"${kpis['reported_cost']:.2f}")
    c2.metric("Input Tokens", f"{int(kpis['total_input_tokens']):,}")
    c3.metric("Output Tokens", f"{int(kpis['total_output_tokens']):,}")
    c4.metric("Total Requests", f"{int(kpis['total_requests']):,}")
    c5.metric("Active Models", f"{int(kpis['active_models'])}")
    c6.metric("Avg Cost / Request", f"${kpis['avg_reported_cost_per_request']:.6f}")


def render_limitations() -> None:
    st.info(
        "Prompt text, response text, and true per-request billing are not exposed by the "
        "organization usage/cost APIs used in this POC. Model-level cost values may be estimated "
        "using local pricing maps when direct model billing attribution is unavailable. "
        "Model intelligence labels are lightweight heuristics and should be validated with offline quality tests."
    )


def render_forecast_disclaimer() -> None:
    st.caption(
        "Forecasts are baseline linear projections from historical aggregates and are directional, "
        "not finance-grade forecasts."
    )
