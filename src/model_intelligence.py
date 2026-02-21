"""Model intelligence helpers for cross-provider comparison and migration analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_model_intelligence_table(unified_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Model",
        "Provider",
        "Best Suited For",
        "Latency Profile",
        "Reasoning Profile",
        "Cost Profile",
        "Calls",
        "Avg Tokens/Call",
        "Observed CPI",
        "Observed Spend (USD)",
    ]
    if unified_df.empty:
        return pd.DataFrame(columns=columns)

    df = unified_df.copy()
    df["calls"] = pd.to_numeric(df["calls"], errors="coerce").fillna(0.0)
    df["total_tokens"] = pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0.0)
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)

    grouped = (
        df.groupby(["provider", "model"], as_index=False)
        .agg(
            calls=("calls", "sum"),
            total_tokens=("total_tokens", "sum"),
            total_cost=("cost_usd", "sum"),
        )
        .sort_values("total_cost", ascending=False)
    )

    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        provider = str(row["provider"])
        model = str(row["model"])
        profile = infer_model_profile(provider=provider, model=model)
        calls = float(row["calls"])
        total_tokens = float(row["total_tokens"])
        total_cost = float(row["total_cost"])
        avg_tokens = total_tokens / calls if calls > 0 else 0.0
        cpi = total_cost / calls if calls > 0 else 0.0

        rows.append(
            {
                "Model": model,
                "Provider": provider.title(),
                "Best Suited For": profile["best_for"],
                "Latency Profile": profile["latency"],
                "Reasoning Profile": profile["reasoning"],
                "Cost Profile": profile["cost_profile"],
                "Calls": int(calls),
                "Avg Tokens/Call": round(avg_tokens, 2),
                "Observed CPI": f"${cpi:.6f}",
                "Observed Spend (USD)": round(total_cost, 4),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def list_provider_models(unified_df: pd.DataFrame) -> list[tuple[str, str]]:
    if unified_df.empty:
        return []
    grouped = (
        unified_df.groupby(["provider", "model"], as_index=False)["cost_usd"]
        .sum()
        .sort_values("cost_usd", ascending=False)
    )
    return [(str(row["provider"]), str(row["model"])) for _, row in grouped.iterrows()]


def recommend_migration(
    unified_df: pd.DataFrame,
    *,
    source_provider: str,
    source_model: str,
    target_provider: str,
    target_model: str,
    pricing_maps: dict[str, dict[str, dict[str, float]]] | None = None,
    target_cpi_override: float | None = None,
) -> dict[str, Any]:
    if unified_df.empty:
        return {"error": "No usage/cost rows available for migration analysis."}

    df = unified_df.copy()
    df["provider"] = df["provider"].astype(str)
    df["model"] = df["model"].astype(str)
    df["calls"] = pd.to_numeric(df["calls"], errors="coerce").fillna(0.0)
    df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0.0)
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0.0)
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)

    source_df = df[(df["provider"] == source_provider) & (df["model"] == source_model)]
    if source_df.empty:
        return {"error": "No historical usage found for the selected source model."}

    source_calls = float(source_df["calls"].sum())
    source_input_tokens = float(source_df["input_tokens"].sum())
    source_output_tokens = float(source_df["output_tokens"].sum())
    source_cost = float(source_df["cost_usd"].sum())
    source_cpi = source_cost / source_calls if source_calls > 0 else 0.0

    target_df = df[(df["provider"] == target_provider) & (df["model"] == target_model)]
    target_observed_calls = float(target_df["calls"].sum())
    target_observed_cost = float(target_df["cost_usd"].sum())
    target_observed_cpi = (
        target_observed_cost / target_observed_calls if target_observed_calls > 0 else 0.0
    )

    pricing_based_cpi = _estimate_target_cpi_from_pricing(
        source_calls=source_calls,
        source_input_tokens=source_input_tokens,
        source_output_tokens=source_output_tokens,
        target_provider=target_provider,
        target_model=target_model,
        pricing_maps=pricing_maps or {},
    )

    target_cpi = 0.0
    cpi_source = "unavailable"
    if target_cpi_override is not None and target_cpi_override > 0:
        target_cpi = float(target_cpi_override)
        cpi_source = "manual_override"
    elif target_observed_cpi > 0:
        target_cpi = target_observed_cpi
        cpi_source = "observed_history"
    elif pricing_based_cpi > 0:
        target_cpi = pricing_based_cpi
        cpi_source = "pricing_estimate"

    if target_cpi <= 0:
        return {
            "error": "Insufficient target cost data. Use a model with observed CPI or provide a CPI override.",
            "source_calls": source_calls,
            "source_cost": source_cost,
            "source_cpi": source_cpi,
        }

    estimated_target_cost = target_cpi * source_calls
    cost_delta = estimated_target_cost - source_cost
    delta_pct = (cost_delta / source_cost) * 100.0 if source_cost > 0 else 0.0

    source_profile = infer_model_profile(provider=source_provider, model=source_model)
    target_profile = infer_model_profile(provider=target_provider, model=target_model)

    recommendation = _build_recommendation(cost_delta=cost_delta, delta_pct=delta_pct)

    return {
        "source_provider": source_provider,
        "source_model": source_model,
        "target_provider": target_provider,
        "target_model": target_model,
        "source_calls": source_calls,
        "source_cost": source_cost,
        "source_cpi": source_cpi,
        "target_cpi": target_cpi,
        "target_cpi_source": cpi_source,
        "estimated_target_cost": estimated_target_cost,
        "cost_delta": cost_delta,
        "cost_delta_pct": delta_pct,
        "source_profile": source_profile,
        "target_profile": target_profile,
        "recommendation": recommendation,
    }


def infer_model_profile(provider: str, model: str) -> dict[str, str]:
    model_l = model.lower()
    provider_l = provider.lower()

    best_for = "General assistant workloads"
    latency = "Balanced"
    reasoning = "Balanced"
    cost_profile = "Balanced"

    if "opus" in model_l:
        best_for = "High-complexity reasoning and long-form coding/planning"
        latency = "Higher latency"
        reasoning = "Top-tier"
        cost_profile = "Premium"
    elif "sonnet" in model_l or "gpt-4" in model_l or "gpt-5" in model_l:
        best_for = "Strong reasoning, coding, and production-grade assistants"
        latency = "Moderate"
        reasoning = "Advanced"
        cost_profile = "Mid to high"
    elif "haiku" in model_l or "mini" in model_l or "nano" in model_l or "instant" in model_l:
        best_for = "Low-latency chat, routing, and high-throughput tasks"
        latency = "Low latency"
        reasoning = "Light to medium"
        cost_profile = "Budget-friendly"
    elif "llama" in model_l or "qwen" in model_l:
        best_for = "Cost-sensitive generation and scalable inference workloads"
        latency = "Low to moderate"
        reasoning = "Medium"
        cost_profile = "Budget to mid"

    if provider_l == "groq":
        latency = "Very low latency"

    return {
        "best_for": best_for,
        "latency": latency,
        "reasoning": reasoning,
        "cost_profile": cost_profile,
    }


def _estimate_target_cpi_from_pricing(
    *,
    source_calls: float,
    source_input_tokens: float,
    source_output_tokens: float,
    target_provider: str,
    target_model: str,
    pricing_maps: dict[str, dict[str, dict[str, float]]],
) -> float:
    if source_calls <= 0:
        return 0.0

    provider_map = pricing_maps.get(target_provider, {})
    price = provider_map.get(target_model.lower())
    if not price:
        return 0.0

    avg_input_tokens = source_input_tokens / source_calls
    avg_output_tokens = source_output_tokens / source_calls
    return ((avg_input_tokens / 1_000_000) * float(price.get("input", 0))) + (
        (avg_output_tokens / 1_000_000) * float(price.get("output", 0))
    )


def _build_recommendation(*, cost_delta: float, delta_pct: float) -> str:
    if delta_pct <= -20:
        return "Strong cost-downside is favorable. Migration is likely beneficial if quality meets your bar."
    if delta_pct < 0:
        return "Potential cost savings detected. Run an A/B quality check before full migration."
    if delta_pct <= 15:
        return "Near cost parity. Decide based on quality, latency, and feature fit rather than spend."
    return "Likely cost increase. Migrate only if quality/latency gains justify the higher spend."
