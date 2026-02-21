import pandas as pd

from src.providers.anthropic_adapter import (
    build_anthropic_cost_df,
    build_anthropic_unified_df,
    build_anthropic_usage_df,
)
from src.providers.groq_adapter import _build_query_range_urls, build_groq_unified_df


def test_anthropic_usage_cost_and_unified_normalization() -> None:
    usage_rows = [
        {
            "starting_at": "2026-02-01T00:00:00Z",
            "ending_at": "2026-02-02T00:00:00Z",
            "results": [
                {
                    "model": "claude-3-5-sonnet",
                    "workspace_id": "ws_1",
                    "requests": 10,
                    "uncached_input_tokens": 1000,
                    "cache_read_input_tokens": 200,
                    "output_tokens": 500,
                }
            ],
        }
    ]
    cost_rows = [
        {
            "starting_at": "2026-02-01T00:00:00Z",
            "ending_at": "2026-02-02T00:00:00Z",
            "results": [
                {
                    "workspace_id": "ws_1",
                    "model": "claude-3-5-sonnet",
                    "amount": {"value": "250.0", "currency": "USD"},
                }
            ],
        }
    ]

    usage_df = build_anthropic_usage_df(usage_rows)
    cost_df = build_anthropic_cost_df(cost_rows)
    unified_df = build_anthropic_unified_df(usage_df, cost_df)

    assert usage_df["input_tokens"].sum() == 1200
    assert usage_df["output_tokens"].sum() == 500
    assert usage_df["total_tokens"].sum() == 1700
    assert cost_df["amount"].sum() == 2.5
    assert unified_df["cost_usd"].sum() == 2.5
    assert set(unified_df["provider"]) == {"anthropic"}


def test_groq_unified_builds_cost_from_metrics() -> None:
    base_timestamp = pd.Timestamp("2026-02-01T00:00:00Z")
    base_keys = {"timestamp": base_timestamp, "model": "llama-3.1-8b-instant", "project_id": "proj_a"}
    metric_frames = {
        "calls": pd.DataFrame([{**base_keys, "calls": 100.0}]),
        "input_tokens": pd.DataFrame([{**base_keys, "input_tokens": 1_000_000.0}]),
        "output_tokens": pd.DataFrame([{**base_keys, "output_tokens": 1_000_000.0}]),
    }

    unified_df = build_groq_unified_df(metric_frames)

    assert len(unified_df) == 1
    assert unified_df.iloc[0]["total_tokens"] == 2_000_000.0
    assert unified_df.iloc[0]["cost_usd"] == 0.13
    assert unified_df.iloc[0]["cost_source"] == "estimated"
    assert unified_df.iloc[0]["provider"] == "groq"


def test_groq_query_range_url_builder() -> None:
    assert _build_query_range_urls("https://api.groq.com/v1/metrics/prometheus") == [
        "https://api.groq.com/v1/metrics/prometheus/api/v1/query_range",
        "https://api.groq.com/v1/metrics/prometheus/query_range",
    ]
    assert _build_query_range_urls("https://api.groq.com/v1/metrics/prometheus/api/v1") == [
        "https://api.groq.com/v1/metrics/prometheus/api/v1/query_range"
    ]
