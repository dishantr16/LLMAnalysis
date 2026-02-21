import pandas as pd

from src.model_intelligence import (
    build_model_intelligence_table,
    list_provider_models,
    recommend_migration,
)


def _sample_unified_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "provider": "openai",
                "model": "gpt-4o",
                "calls": 100,
                "input_tokens": 1_000_000,
                "output_tokens": 500_000,
                "total_tokens": 1_500_000,
                "cost_usd": 20.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-02T00:00:00Z"),
                "provider": "anthropic",
                "model": "claude-sonnet",
                "calls": 100,
                "input_tokens": 1_000_000,
                "output_tokens": 500_000,
                "total_tokens": 1_500_000,
                "cost_usd": 18.0,
            },
        ]
    )


def test_build_model_intelligence_table_and_list_models() -> None:
    unified_df = _sample_unified_df()
    table = build_model_intelligence_table(unified_df)
    assert not table.empty
    assert {"Model", "Provider", "Best Suited For"}.issubset(table.columns)

    pairs = list_provider_models(unified_df)
    assert ("openai", "gpt-4o") in pairs
    assert ("anthropic", "claude-sonnet") in pairs


def test_recommend_migration_uses_observed_target_cpi() -> None:
    unified_df = _sample_unified_df()
    result = recommend_migration(
        unified_df,
        source_provider="openai",
        source_model="gpt-4o",
        target_provider="anthropic",
        target_model="claude-sonnet",
        pricing_maps={},
    )

    assert "error" not in result
    assert result["target_cpi_source"] == "observed_history"
    assert result["source_calls"] == 100.0
    assert result["estimated_target_cost"] == 18.0


def test_recommend_migration_with_override() -> None:
    unified_df = _sample_unified_df()
    result = recommend_migration(
        unified_df,
        source_provider="openai",
        source_model="gpt-4o",
        target_provider="groq",
        target_model="llama-3.1-8b-instant",
        pricing_maps={},
        target_cpi_override=0.1,
    )
    assert "error" not in result
    assert result["target_cpi_source"] == "manual_override"
