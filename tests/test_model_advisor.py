import pandas as pd

from src.model_advisor import build_usage_workload_profile, run_ai_model_advisor


def _sample_unified_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-02-01T00:00:00Z"),
                "provider": "openai",
                "model": "gpt-4o",
                "calls": 300,
                "input_tokens": 3_000_000,
                "output_tokens": 1_500_000,
                "total_tokens": 4_500_000,
                "cost_usd": 120.0,
            },
            {
                "timestamp": pd.Timestamp("2026-02-02T00:00:00Z"),
                "provider": "openai",
                "model": "gpt-4o-mini",
                "calls": 200,
                "input_tokens": 2_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 3_000_000,
                "cost_usd": 12.0,
            },
            {
                "timestamp": pd.Timestamp("2026-02-03T00:00:00Z"),
                "provider": "anthropic",
                "model": "claude-sonnet",
                "calls": 100,
                "input_tokens": 1_000_000,
                "output_tokens": 500_000,
                "total_tokens": 1_500_000,
                "cost_usd": 35.0,
            },
        ]
    )


def test_build_usage_workload_profile_has_expected_summary() -> None:
    profile = build_usage_workload_profile(_sample_unified_df())

    assert profile["total_calls"] == 600.0
    assert profile["current_provider"] == "openai"
    assert profile["current_model"] == "gpt-4o"
    assert profile["estimated_monthly_calls"] > 0
    assert profile["avg_total_tokens_per_call"] > 0


def test_run_ai_model_advisor_returns_structured_output() -> None:
    result = run_ai_model_advisor(
        _sample_unified_df(),
        objective="balanced",
        complexity_level="medium",
        max_latency_ms=900,
        monthly_budget_usd=2000.0,
        required_capabilities=("reasoning", "code"),
        primary_task="coding_assistant",
        preferred_providers=("openai", "anthropic"),
        allow_cross_provider=True,
    )

    assert "error" not in result
    assert "primary_recommendation" in result
    assert "category_scores" in result
    assert set(result["category_scores"].keys()) == {"SLM", "LLM", "Frontier"}
    assert not result["candidates_df"].empty

    recommendation = result["primary_recommendation"]
    assert recommendation["recommended_category"] in {"SLM", "LLM", "Frontier"}
    assert recommendation["provider"] in {"OpenAI", "Anthropic"}
    assert recommendation["confidence_score"] >= 35.0
    assert isinstance(recommendation["reasoning"], list)


def test_run_ai_model_advisor_respects_provider_lock() -> None:
    result = run_ai_model_advisor(
        _sample_unified_df(),
        objective="min_cost",
        complexity_level="low",
        max_latency_ms=500,
        preferred_providers=("openai", "anthropic"),
        allow_cross_provider=False,
    )

    assert "error" not in result
    candidates_df = result["candidates_df"]
    assert set(candidates_df["provider"].str.lower()) == {"openai"}
