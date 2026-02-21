"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_TEMPLATE = "plotly_white"


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=message,
        showarrow=False,
        xref="paper",
        yref="paper",
        font={"size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=360, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def usage_trend_chart(usage_rollup: pd.DataFrame, title: str) -> go.Figure:
    if usage_rollup.empty:
        return empty_figure("No usage data for selected range")

    melt_df = usage_rollup.melt(
        id_vars=["period"],
        value_vars=["input_tokens", "output_tokens"],
        var_name="token_type",
        value_name="tokens",
    )
    fig = px.area(
        melt_df,
        x="period",
        y="tokens",
        color="token_type",
        title=title,
        labels={"period": "Date", "tokens": "Tokens", "token_type": "Token Type"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def cost_trend_chart(cost_rollup: pd.DataFrame, title: str) -> go.Figure:
    if cost_rollup.empty:
        return empty_figure("No cost data for selected range")

    fig = px.line(
        cost_rollup,
        x="period",
        y="amount",
        title=title,
        markers=True,
        labels={"period": "Date", "amount": "Cost (USD)"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(line={"width": 2})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def model_cost_chart(model_summary: pd.DataFrame) -> go.Figure:
    if model_summary.empty:
        return empty_figure("No model-level usage data")

    plot_df = model_summary.dropna(subset=["estimated_cost_usd"]).copy()
    if plot_df.empty:
        return empty_figure("No pricing map entries for current models")

    fig = px.bar(
        plot_df,
        x="model",
        y="estimated_cost_usd",
        color="has_pricing",
        title="Model-wise Estimated Cost (USD)",
        labels={"estimated_cost_usd": "Estimated Cost (USD)", "model": "Model"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


def model_tokens_chart(model_summary: pd.DataFrame) -> go.Figure:
    if model_summary.empty:
        return empty_figure("No model-level usage data")

    melt_df = model_summary.melt(
        id_vars=["model"],
        value_vars=["input_tokens", "output_tokens"],
        var_name="token_type",
        value_name="tokens",
    )

    fig = px.bar(
        melt_df,
        x="model",
        y="tokens",
        color="token_type",
        title="Token Distribution by Model",
        labels={"model": "Model", "tokens": "Tokens", "token_type": "Token Type"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def token_distribution_pie(token_distribution_df: pd.DataFrame) -> go.Figure:
    if token_distribution_df.empty:
        return empty_figure("No token distribution data")

    fig = px.pie(
        token_distribution_df,
        names="token_type",
        values="tokens",
        title="Overall Token Distribution",
        template=PLOTLY_TEMPLATE,
        hole=0.4,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def project_cost_chart(project_cost_df: pd.DataFrame) -> go.Figure:
    if project_cost_df.empty:
        return empty_figure("No project-level cost data")

    fig = px.bar(
        project_cost_df,
        x="project_id",
        y="amount",
        title="Top Projects by Cost",
        labels={"project_id": "Project", "amount": "Cost (USD)"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def requests_by_model_chart(model_summary: pd.DataFrame) -> go.Figure:
    if model_summary.empty:
        return empty_figure("No model request data")

    fig = px.bar(
        model_summary,
        x="model",
        y="requests",
        title="Requests by Model",
        labels={"model": "Model", "requests": "Requests"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def generic_metric_trend_chart(metric_rollup: pd.DataFrame, *, metric: str, title: str) -> go.Figure:
    if metric_rollup.empty or metric not in metric_rollup.columns:
        return empty_figure("No metric data for selected endpoint")

    fig = px.line(
        metric_rollup,
        x="period",
        y=metric,
        markers=True,
        title=title,
        labels={"period": "Date", metric: metric.replace("_", " ").title()},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(line={"width": 2})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def generic_dimension_bar_chart(
    dimension_df: pd.DataFrame,
    *,
    dimension: str,
    metric: str,
    title: str,
) -> go.Figure:
    if dimension_df.empty or metric not in dimension_df.columns or dimension not in dimension_df.columns:
        return empty_figure("No dimension data available")

    fig = px.bar(
        dimension_df,
        x=dimension,
        y=metric,
        title=title,
        labels={
            dimension: dimension.replace("_", " ").title(),
            metric: metric.replace("_", " ").title(),
        },
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def forecast_chart(forecast_df: pd.DataFrame, *, metric: str, title: str) -> go.Figure:
    if forecast_df.empty or metric not in forecast_df.columns:
        return empty_figure("Not enough data to build forecast")

    fig = px.line(
        forecast_df,
        x="period",
        y=metric,
        color="series",
        line_dash="series",
        markers=True,
        title=title,
        labels={"period": "Date", metric: metric.replace("_", " ").title(), "series": "Series"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def monthly_budget_chart(monthly_df: pd.DataFrame, *, monthly_budget: float) -> go.Figure:
    if monthly_df.empty:
        return empty_figure("No monthly spend data")

    df = monthly_df.copy()
    df["budget"] = monthly_budget

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["period"],
            y=df["cost_usd"],
            name="Actual Spend",
            marker_color="#0f766e",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["budget"],
            name="Budget",
            mode="lines+markers",
            line={"color": "#b91c1c", "width": 2, "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Monthly Spend Trend vs Budget",
        template=PLOTLY_TEMPLATE,
        yaxis_title="Cost (USD)",
        xaxis_title="Month",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def provider_spend_chart(provider_df: pd.DataFrame) -> go.Figure:
    if provider_df.empty:
        return empty_figure("No provider spend data")

    fig = px.pie(
        provider_df,
        names="provider",
        values="cost_usd",
        hole=0.4,
        title="Spend by Provider",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def top_models_cost_chart(models_df: pd.DataFrame) -> go.Figure:
    if models_df.empty:
        return empty_figure("No model cost data")

    plot_df = models_df.copy()
    plot_df["model_provider"] = plot_df["model"] + " · " + plot_df["provider"]
    fig = px.bar(
        plot_df,
        x="model_provider",
        y="cost_usd",
        color="provider",
        title="Top Models by Cost",
        labels={"model_provider": "Model", "cost_usd": "Cost (USD)", "provider": "Provider"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig
