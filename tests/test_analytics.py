import pandas as pd
import pytest

from src.analytics import (
    aggregate_cost,
    aggregate_generic_usage,
    aggregate_usage,
    build_actual_vs_forecast,
    build_generic_metric_dimension_summary,
    build_model_summary,
    build_token_distribution,
    compute_kpis,
    model_cost_breakdown,
    monthly_spend_trend,
    project_current_month_total,
    spend_by_provider,
    top_models_by_cost,
)


def _sample_usage_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bucket_start": pd.Timestamp("2026-01-01", tz="UTC"),
                "model": "gpt-4o",
                "input_tokens": 1_000_000,
                "output_tokens": 500_000,
                "total_tokens": 1_500_000,
                "requests": 100,
            },
            {
                "bucket_start": pd.Timestamp("2026-01-02", tz="UTC"),
                "model": "gpt-4o-mini",
                "input_tokens": 2_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 3_000_000,
                "requests": 200,
            },
        ]
    )


def _sample_cost_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"bucket_start": pd.Timestamp("2026-01-01", tz="UTC"), "amount": 10.0, "currency": "usd"},
            {"bucket_start": pd.Timestamp("2026-01-02", tz="UTC"), "amount": 5.0, "currency": "usd"},
        ]
    )


def test_build_model_summary_and_kpis() -> None:
    usage_df = _sample_usage_df()
    cost_df = _sample_cost_df()

    custom_pricing = {
        "gpt-4o": {"input": 2.0, "output": 8.0},
        "gpt-4o-mini": {"input": 0.2, "output": 0.8},
    }
    model_summary = build_model_summary(usage_df, pricing_map=custom_pricing)

    gpt4o_cost = model_summary.loc[model_summary["model"] == "gpt-4o", "estimated_cost_usd"].iloc[0]
    gpt4omini_cost = model_summary.loc[
        model_summary["model"] == "gpt-4o-mini", "estimated_cost_usd"
    ].iloc[0]

    assert gpt4o_cost == pytest.approx(6.0)
    assert gpt4omini_cost == pytest.approx(1.2)

    kpis = compute_kpis(usage_df, cost_df, model_summary)
    assert kpis["total_tokens"] == 4_500_000
    assert kpis["total_requests"] == 300
    assert kpis["reported_cost"] == 15.0


def test_aggregate_and_token_distribution() -> None:
    usage_df = _sample_usage_df()
    cost_df = _sample_cost_df()

    daily_usage = aggregate_usage(usage_df, "D")
    daily_cost = aggregate_cost(cost_df, "D")

    assert len(daily_usage) == 2
    assert len(daily_cost) == 2

    model_summary = build_model_summary(usage_df, pricing_map={})
    token_distribution = build_token_distribution(model_summary)
    assert set(token_distribution["token_type"]) == {"Input Tokens", "Output Tokens"}


def test_generic_aggregate_dimension_and_forecast() -> None:
    generic_df = pd.DataFrame(
        [
            {
                "bucket_start": pd.Timestamp("2026-01-01", tz="UTC"),
                "project_id": "proj_a",
                "num_requests": 10,
            },
            {
                "bucket_start": pd.Timestamp("2026-01-02", tz="UTC"),
                "project_id": "proj_b",
                "num_requests": 20,
            },
        ]
    )

    aggregated = aggregate_generic_usage(generic_df, "D")
    assert len(aggregated) == 2
    assert aggregated["num_requests"].sum() == 30

    by_project = build_generic_metric_dimension_summary(
        generic_df,
        metric="num_requests",
        dimension="project_id",
    )
    assert by_project.iloc[0]["num_requests"] == 20

    forecast_frame = build_actual_vs_forecast(
        aggregated[["period", "num_requests"]],
        value_column="num_requests",
        horizon_days=7,
    )
    assert set(forecast_frame["series"]) == {"actual", "forecast"}
    assert len(forecast_frame) == len(aggregated) + 7

    projection = project_current_month_total(
        aggregated.rename(columns={"num_requests": "amount"}),
        value_column="amount",
    )
    assert "linear_projection" in projection


def test_provider_insights_aggregations() -> None:
    unified_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "provider": "openai",
                "model": "gpt-4o",
                "calls": 100,
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
                "cost_usd": 10.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-15", tz="UTC"),
                "provider": "anthropic",
                "model": "claude-x",
                "calls": 50,
                "input_tokens": 500,
                "output_tokens": 250,
                "total_tokens": 750,
                "cost_usd": 6.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-20", tz="UTC"),
                "provider": "openai",
                "model": "gpt-4o",
                "calls": 120,
                "input_tokens": 1200,
                "output_tokens": 600,
                "total_tokens": 1800,
                "cost_usd": 12.0,
            },
        ]
    )

    monthly = monthly_spend_trend(unified_df)
    assert monthly["cost_usd"].sum() == 28.0

    provider_spend = spend_by_provider(unified_df)
    assert set(provider_spend["provider"]) == {"openai", "anthropic"}

    top_models = top_models_by_cost(unified_df)
    assert not top_models.empty
    assert top_models.iloc[0]["cost_usd"] >= top_models.iloc[-1]["cost_usd"]

    breakdown = model_cost_breakdown(unified_df)
    expected_columns = {
        "Rank",
        "Model",
        "Provider",
        "Calls (24h)",
        "Avg Tokens",
        "CPI",
        "7-Day Trend",
        "Status",
    }
    assert expected_columns.issubset(set(breakdown.columns))
