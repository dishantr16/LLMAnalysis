"""Cross-provider cost comparison engine and scenario modeling.

Implements AC-3.1 (cross-provider equivalent cost), AC-3.3 (scenario modeling),
and the cost analysis backbone that feeds the recommendation engine.

Given normalized GPT usage metrics, estimates equivalent cost for:
- GPT model variants (GPT-4.x family, GPT-5, o-series)
- Claude models (Haiku, Sonnet, Opus)
- Any model registered in the pricing registry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.pricing_registry import (
    ModelPricingEntry,
    PricingRegistry,
    get_pricing_registry,
)
from src.token_normalizer import (
    NormalizedTokenUsage,
    ReasoningRatioTracker,
    TokenConversionResult,
    compute_token_conversion_certainty,
    normalize_token_usage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkloadProfile:
    """Summarized workload characteristics for cost projection.

    Attributes:
        avg_input_tokens_per_call: Average input tokens per request.
        avg_output_tokens_per_call: Average output tokens per request.
        avg_reasoning_tokens_per_call: Average reasoning tokens per request
            (0 for non-reasoning models).
        avg_cached_ratio: Fraction of input tokens served from cache (0.0–1.0).
        total_calls: Total requests in observation window.
        window_days: Number of days in observation window.
        monthly_calls: Projected monthly request count.
        source_provider: Provider of the observed workload.
        source_model: Model of the observed workload.
        observed_monthly_cost: Actual observed monthly spend.
    """

    avg_input_tokens_per_call: float = 0.0
    avg_output_tokens_per_call: float = 0.0
    avg_reasoning_tokens_per_call: float = 0.0
    avg_cached_ratio: float = 0.0
    total_calls: float = 0.0
    window_days: float = 1.0
    monthly_calls: float = 0.0
    source_provider: str = "unknown"
    source_model: str = "unknown"
    observed_monthly_cost: float = 0.0
    reasoning_tracker: ReasoningRatioTracker | None = None


@dataclass(frozen=True)
class ScenarioResult:
    """Cost projection for a single model under a workload scenario (AC-3.3).

    Attributes:
        provider: Target provider.
        model_id: Target model.
        family: Model family.
        current_usage_cost: Total cost for current observation window.
        projected_monthly_cost: Estimated monthly cost at observed call rate.
        cost_per_request: Cost per single request.
        cost_per_1k_tokens: Cost per 1,000 total billed tokens.
        savings_vs_current_usd: Savings compared to current model (positive = cheaper).
        savings_vs_current_pct: Savings percentage.
        break_even_calls: Number of calls at which this model equals current spend.
        conversion_certainty: Confidence in the cost estimate (0.0–1.0).
        assumptions: List of assumptions made.
        context_window: Maximum context length.
        supports_reasoning: Whether model supports reasoning tokens.
    """

    provider: str
    model_id: str
    family: str = ""
    current_usage_cost: float = 0.0
    projected_monthly_cost: float = 0.0
    cost_per_request: float = 0.0
    cost_per_1k_tokens: float = 0.0
    savings_vs_current_usd: float = 0.0
    savings_vs_current_pct: float = 0.0
    break_even_calls: float | None = None
    conversion_certainty: float = 0.5
    assumptions: tuple[str, ...] = ()
    context_window: int = 0
    supports_reasoning: bool = False


# ---------------------------------------------------------------------------
# Workload profiling from unified DataFrame
# ---------------------------------------------------------------------------


def build_workload_profile(
    unified_df: pd.DataFrame,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> WorkloadProfile:
    """Extract a workload profile from the unified DataFrame.

    Args:
        unified_df: Unified schema DataFrame (all providers).
        provider: Optional filter to a specific provider.
        model: Optional filter to a specific model.

    Returns:
        ``WorkloadProfile`` summarizing the observed workload.
    """
    if unified_df.empty:
        return WorkloadProfile()

    df = unified_df.copy()
    for col in ("calls", "input_tokens", "output_tokens", "total_tokens", "cost_usd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if provider:
        df = df[df["provider"].astype(str).str.lower() == provider.lower()]
    if model:
        df = df[df["model"].astype(str) == model]

    if df.empty:
        return WorkloadProfile()

    total_calls = float(df["calls"].sum())
    total_input = float(df["input_tokens"].sum())
    total_output = float(df["output_tokens"].sum())
    total_cost = float(df["cost_usd"].sum())

    reasoning_col = "reasoning_tokens" if "reasoning_tokens" in df.columns else None
    total_reasoning = float(df[reasoning_col].sum()) if reasoning_col else 0.0

    cached_col = "cached_input_tokens" if "cached_input_tokens" in df.columns else None
    total_cached = float(df[cached_col].sum()) if cached_col else 0.0
    cached_ratio = total_cached / total_input if total_input > 0 else 0.0

    ts_col = "timestamp" if "timestamp" in df.columns else None
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        t_min = df[ts_col].min()
        t_max = df[ts_col].max()
        window_days = float(max(1, (t_max.normalize() - t_min.normalize()).days + 1)) if pd.notna(t_min) else 1.0
    else:
        window_days = 1.0

    avg_calls_per_day = total_calls / window_days
    monthly_calls = avg_calls_per_day * 30.0
    monthly_cost = (total_cost / window_days) * 30.0

    dominant = (
        df.groupby(["provider", "model"], as_index=False)
        .agg(cost=("cost_usd", "sum"))
        .sort_values("cost", ascending=False)
    )
    src_provider = str(dominant.iloc[0]["provider"]) if not dominant.empty else "unknown"
    src_model = str(dominant.iloc[0]["model"]) if not dominant.empty else "unknown"

    tracker = ReasoningRatioTracker.from_dataframe(unified_df)
    if not tracker.is_empty:
        logger.info(
            "Reasoning ratio tracker built with %d model(s): %s",
            len(tracker.tracked_models),
            ", ".join(tracker.tracked_models),
        )

    return WorkloadProfile(
        avg_input_tokens_per_call=total_input / total_calls if total_calls > 0 else 0.0,
        avg_output_tokens_per_call=total_output / total_calls if total_calls > 0 else 0.0,
        avg_reasoning_tokens_per_call=total_reasoning / total_calls if total_calls > 0 else 0.0,
        avg_cached_ratio=cached_ratio,
        total_calls=total_calls,
        window_days=window_days,
        monthly_calls=monthly_calls,
        source_provider=src_provider,
        source_model=src_model,
        observed_monthly_cost=monthly_cost,
        reasoning_tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Single-model cost estimation
# ---------------------------------------------------------------------------


def estimate_model_cost(
    profile: WorkloadProfile,
    target_entry: ModelPricingEntry,
    *,
    registry: PricingRegistry | None = None,
    use_conversion: bool = True,
) -> ScenarioResult:
    """Estimate cost of running the given workload on a target model.

    When ``use_conversion=True`` (default), applies cross-model conversion
    factors that account for tokenizer differences, output length behavior,
    reasoning overhead, and quality gaps. This produces more accurate
    estimates than naive token-count transfer.

    Args:
        profile: Observed workload characteristics.
        target_entry: Target model pricing entry.
        registry: Pricing registry (used for freshness check).
        use_conversion: If True, use the conversion engine for adjusted
            token estimates. If False, use naive token transfer (legacy).

    Returns:
        ``ScenarioResult`` with full cost projection.
    """
    reg = registry or get_pricing_registry()
    p = target_entry.pricing
    assumptions: list[str] = []

    avg_input = profile.avg_input_tokens_per_call
    avg_output = profile.avg_output_tokens_per_call
    avg_reasoning = profile.avg_reasoning_tokens_per_call
    cached_ratio = profile.avg_cached_ratio

    if use_conversion and profile.source_provider != "unknown":
        from src.conversion_engine import estimate_converted_workload

        observed_ratio: float | None = None
        if profile.reasoning_tracker is not None:
            ratio_info = profile.reasoning_tracker.get_ratio(profile.source_model)
            if ratio_info is not None:
                observed_ratio = ratio_info.ratio

        converted = estimate_converted_workload(
            profile.source_provider,
            profile.source_model,
            target_entry.provider,
            target_entry.model_id,
            avg_input_tokens=avg_input,
            avg_output_tokens=avg_output,
            avg_reasoning_tokens=avg_reasoning,
            total_calls=profile.total_calls,
            monthly_calls=profile.monthly_calls,
            source_monthly_cost=profile.observed_monthly_cost,
            cached_ratio=cached_ratio,
            observed_reasoning_ratio=observed_ratio,
            registry=reg,
        )

        total_tokens = converted.est_total_tokens_per_call
        cost_per_1k = (
            (converted.est_cost_per_call / total_tokens * 1000.0)
            if total_tokens > 0
            else 0.0
        )

        break_even: float | None = None
        if converted.est_cost_per_call > 0 and profile.observed_monthly_cost > 0:
            break_even = profile.observed_monthly_cost / converted.est_cost_per_call

        return ScenarioResult(
            provider=target_entry.provider,
            model_id=target_entry.model_id,
            family=target_entry.family,
            current_usage_cost=round(converted.est_cost_per_call * profile.total_calls, 4),
            projected_monthly_cost=converted.est_monthly_cost,
            cost_per_request=round(converted.est_cost_per_call, 8),
            cost_per_1k_tokens=round(cost_per_1k, 6),
            savings_vs_current_usd=converted.savings_usd,
            savings_vs_current_pct=converted.savings_pct,
            break_even_calls=break_even,
            conversion_certainty=converted.conversion_confidence,
            assumptions=converted.conversion_factors.assumptions,
            context_window=target_entry.context_window,
            supports_reasoning=target_entry.supports_reasoning,
        )

    # Legacy naive path (no conversion factors)
    effective_input = avg_input * (1.0 - cached_ratio)
    cached_input = avg_input * cached_ratio

    if target_entry.supports_reasoning and avg_reasoning > 0:
        output_for_cost = avg_output
        reasoning_for_cost = avg_reasoning
        assumptions.append("Reasoning token ratio preserved from source workload")
    elif target_entry.supports_reasoning and avg_reasoning == 0:
        output_for_cost = avg_output
        reasoning_for_cost = 0.0
        assumptions.append("Source workload has no reasoning tokens; target reasoning overhead unknown")
    else:
        output_for_cost = avg_output + avg_reasoning
        reasoning_for_cost = 0.0
        if avg_reasoning > 0:
            assumptions.append(
                "Target model does not support reasoning; reasoning tokens treated as output"
            )

    input_rate = p.input
    cached_rate = p.cached_input if p.cached_input is not None else p.input
    output_rate = p.output
    reasoning_rate = p.reasoning_output if p.reasoning_output is not None else p.output

    cpi = (
        (effective_input / 1_000_000) * input_rate
        + (cached_input / 1_000_000) * cached_rate
        + (output_for_cost / 1_000_000) * output_rate
        + (reasoning_for_cost / 1_000_000) * reasoning_rate
    )

    monthly_cost = cpi * profile.monthly_calls
    window_cost = cpi * profile.total_calls

    total_tokens_per_call = avg_input + avg_output + avg_reasoning
    cost_per_1k = (cpi / total_tokens_per_call * 1000.0) if total_tokens_per_call > 0 else 0.0

    savings_usd = profile.observed_monthly_cost - monthly_cost
    savings_pct = (savings_usd / profile.observed_monthly_cost * 100.0) if profile.observed_monthly_cost > 0 else 0.0

    break_even: float | None = None
    if cpi > 0 and profile.observed_monthly_cost > 0:
        break_even_monthly = profile.observed_monthly_cost / cpi
        break_even = break_even_monthly

    certainty = compute_token_conversion_certainty(
        has_pricing=True,
        has_reasoning_split=avg_reasoning > 0 or not target_entry.supports_reasoning,
        has_reported_cost=profile.observed_monthly_cost > 0,
        pricing_freshness_days=reg.pricing_freshness_days(),
    )

    if cached_ratio > 0:
        assumptions.append(f"Cached input ratio {cached_ratio:.1%} preserved from source")

    return ScenarioResult(
        provider=target_entry.provider,
        model_id=target_entry.model_id,
        family=target_entry.family,
        current_usage_cost=window_cost,
        projected_monthly_cost=round(monthly_cost, 4),
        cost_per_request=round(cpi, 8),
        cost_per_1k_tokens=round(cost_per_1k, 6),
        savings_vs_current_usd=round(savings_usd, 4),
        savings_vs_current_pct=round(savings_pct, 2),
        break_even_calls=round(break_even, 0) if break_even is not None else None,
        conversion_certainty=certainty,
        assumptions=tuple(assumptions),
        context_window=target_entry.context_window,
        supports_reasoning=target_entry.supports_reasoning,
    )


# ---------------------------------------------------------------------------
# Multi-model comparison (AC-3.1)
# ---------------------------------------------------------------------------


def compare_across_providers(
    profile: WorkloadProfile,
    *,
    providers: tuple[str, ...] | None = None,
    registry: PricingRegistry | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Estimate equivalent cost for a workload across all registered models.

    Given normalized usage metrics, estimates equivalent cost for:
    - GPT model variants (GPT-4.x, GPT-5)
    - Claude models (Haiku, Sonnet, Opus)
    - Any configured future provider

    Args:
        profile: Observed workload profile.
        providers: Tuple of provider keys to include (None = all).
        registry: Pricing registry.
        top_n: Max models to return, sorted by projected cost.

    Returns:
        DataFrame with columns for each ``ScenarioResult`` field, sorted by
        ``projected_monthly_cost`` ascending (cheapest first).
    """
    reg = registry or get_pricing_registry()

    entries = reg.all_entries()
    if providers:
        allowed = {p.lower() for p in providers}
        entries = [e for e in entries if e.provider in allowed]

    if not entries:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for entry in entries:
        result = estimate_model_cost(profile, entry, registry=reg)
        rows.append({
            "provider": result.provider,
            "model_id": result.model_id,
            "family": result.family,
            "projected_monthly_cost": result.projected_monthly_cost,
            "cost_per_request": result.cost_per_request,
            "cost_per_1k_tokens": result.cost_per_1k_tokens,
            "savings_vs_current_usd": result.savings_vs_current_usd,
            "savings_vs_current_pct": result.savings_vs_current_pct,
            "break_even_calls": result.break_even_calls,
            "conversion_certainty": result.conversion_certainty,
            "context_window": result.context_window,
            "supports_reasoning": result.supports_reasoning,
            "assumptions": "; ".join(result.assumptions) if result.assumptions else "",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("projected_monthly_cost", ascending=True).reset_index(drop=True)
    return df.head(top_n)


# ---------------------------------------------------------------------------
# Scenario modeling (AC-3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioModelingResult:
    """Complete scenario analysis for a workload (AC-3.3).

    Attributes:
        current_model: Source model scenario result.
        alternatives: List of alternative model scenario results.
        comparison_df: DataFrame with all models compared.
        workload_profile: The workload profile used.
        cheapest_model: The cheapest alternative.
        best_value_model: Best value (cost × quality heuristic).
    """

    current_model: ScenarioResult | None
    alternatives: list[ScenarioResult]
    comparison_df: pd.DataFrame
    workload_profile: WorkloadProfile
    cheapest_model: ScenarioResult | None = None
    best_value_model: ScenarioResult | None = None


def run_scenario_modeling(
    unified_df: pd.DataFrame,
    *,
    source_provider: str | None = None,
    source_model: str | None = None,
    target_providers: tuple[str, ...] | None = None,
    registry: PricingRegistry | None = None,
    top_n: int = 15,
) -> ScenarioModelingResult:
    """Run full scenario modeling for a workload (AC-3.3).

    Supports:
    - Current usage cost
    - Projected monthly cost
    - Cost per request
    - Cost per 1K tokens
    - Break-even analysis

    Args:
        unified_df: Unified schema DataFrame.
        source_provider: Filter workload to this provider.
        source_model: Filter workload to this model.
        target_providers: Providers to compare against.
        registry: Pricing registry.
        top_n: Max alternatives to return.

    Returns:
        ``ScenarioModelingResult`` with current + alternatives + comparison.
    """
    reg = registry or get_pricing_registry()
    profile = build_workload_profile(
        unified_df,
        provider=source_provider,
        model=source_model,
    )

    comparison_df = compare_across_providers(
        profile,
        providers=target_providers,
        registry=reg,
        top_n=top_n + 5,
    )

    current_result: ScenarioResult | None = None
    alternatives: list[ScenarioResult] = []

    src_prov = profile.source_provider.lower()
    src_model = profile.source_model.lower()

    entries = reg.all_entries()
    if target_providers:
        allowed = {p.lower() for p in target_providers}
        entries = [e for e in entries if e.provider in allowed]

    for entry in entries:
        result = estimate_model_cost(profile, entry, registry=reg)
        if entry.provider.lower() == src_prov and entry.model_id.lower() == src_model:
            current_result = result
        else:
            alternatives.append(result)

    alternatives.sort(key=lambda r: r.projected_monthly_cost)
    alternatives = alternatives[:top_n]

    cheapest = alternatives[0] if alternatives else None
    best_value = _pick_best_value(alternatives) if alternatives else None

    return ScenarioModelingResult(
        current_model=current_result,
        alternatives=alternatives,
        comparison_df=comparison_df,
        workload_profile=profile,
        cheapest_model=cheapest,
        best_value_model=best_value,
    )


def _pick_best_value(alternatives: list[ScenarioResult]) -> ScenarioResult | None:
    """Select the model with the best savings-to-certainty ratio."""
    if not alternatives:
        return None

    scored: list[tuple[float, ScenarioResult]] = []
    for alt in alternatives:
        value_score = alt.savings_vs_current_pct * alt.conversion_certainty
        scored.append((value_score, alt))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1] if scored else None
