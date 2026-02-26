"""Enhanced recommendation engine with multi-factor ranking and confidence indicators.

Implements:
- AC-4.1: Cost optimization recommendation (recommend cheapest viable alternative)
- AC-4.2: Multi-factor ranking (cost, latency, context window, quality score)
- AC-4.3: Confidence indicator (data completeness, pricing freshness, token certainty)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.cost_engine import (
    ScenarioResult,
    WorkloadProfile,
    build_workload_profile,
    estimate_model_cost,
)
from src.pricing_registry import (
    ModelPricingEntry,
    PricingRegistry,
    get_pricing_registry,
)

logger = logging.getLogger(__name__)

# Default ranking weights — configurable per AC-4.2
DEFAULT_RANKING_WEIGHTS: dict[str, float] = {
    "cost": 0.35,
    "latency": 0.20,
    "context_window": 0.15,
    "quality_score": 0.30,
}


@dataclass(frozen=True)
class RankingWeights:
    """Configurable ranking weights for multi-factor scoring (AC-4.2).

    All weights should sum to 1.0. If they don't, they are normalized.
    """

    cost: float = 0.35
    latency: float = 0.20
    context_window: float = 0.15
    quality_score: float = 0.30

    def normalized(self) -> dict[str, float]:
        """Return weights normalized to sum to 1.0."""
        total = self.cost + self.latency + self.context_window + self.quality_score
        if total <= 0:
            return {"cost": 0.25, "latency": 0.25, "context_window": 0.25, "quality_score": 0.25}
        return {
            "cost": self.cost / total,
            "latency": self.latency / total,
            "context_window": self.context_window / total,
            "quality_score": self.quality_score / total,
        }


@dataclass(frozen=True)
class ConfidenceIndicator:
    """Confidence breakdown for a recommendation (AC-4.3).

    Attributes:
        overall: Overall confidence score (0–100%).
        data_completeness: Score for how complete the usage data is.
        pricing_freshness: Score based on how recently pricing was updated.
        token_conversion_certainty: Score for token→cost mapping reliability.
        reasoning: Human-readable confidence explanation.
    """

    overall: float = 50.0
    data_completeness: float = 50.0
    pricing_freshness: float = 50.0
    token_conversion_certainty: float = 50.0
    reasoning: str = ""


@dataclass(frozen=True)
class RankedModel:
    """A single model in the ranked recommendation list.

    Attributes:
        rank: Position in ranking (1 = best).
        provider: Provider key.
        model_id: Model identifier.
        family: Model family.
        composite_score: Multi-factor ranking score (0–1).
        cost_score: Cost component score (0–1, higher = cheaper).
        latency_score: Latency component score (0–1, higher = faster).
        context_score: Context window score (0–1, higher = larger).
        quality_score: Quality component score (0–1, higher = better).
        scenario: Full cost scenario for this model.
        is_recommended: Whether this is the primary recommendation.
        recommendation_reason: Why this model is recommended (or not).
    """

    rank: int
    provider: str
    model_id: str
    family: str = ""
    composite_score: float = 0.0
    cost_score: float = 0.0
    latency_score: float = 0.0
    context_score: float = 0.0
    quality_score: float = 0.0
    scenario: ScenarioResult | None = None
    is_recommended: bool = False
    recommendation_reason: str = ""


@dataclass
class RecommendationResult:
    """Complete output from the recommendation engine.

    Attributes:
        primary: The top-ranked recommended model.
        ranked_models: All ranked models.
        confidence: Confidence indicator for the recommendation.
        workload_profile: Observed workload used for analysis.
        weights_used: The ranking weights applied.
        ranked_df: DataFrame representation for UI display.
    """

    primary: RankedModel | None = None
    ranked_models: list[RankedModel] = field(default_factory=list)
    confidence: ConfidenceIndicator = field(default_factory=ConfidenceIndicator)
    workload_profile: WorkloadProfile = field(default_factory=WorkloadProfile)
    weights_used: dict[str, float] = field(default_factory=dict)
    ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Model metadata catalog (latency + quality)
#
# Primary source: data/model_metadata.yaml (editable without code changes)
# Fallback: hardcoded defaults below (emergency only)
# ---------------------------------------------------------------------------

_HARDCODED_MODEL_METADATA: dict[tuple[str, str], dict[str, float]] = {
    ("openai", "gpt-4.1-nano"): {"latency_ms": 120, "quality_score": 0.66},
    ("openai", "gpt-4.1-mini"): {"latency_ms": 200, "quality_score": 0.78},
    ("openai", "gpt-4.1"): {"latency_ms": 340, "quality_score": 0.87},
    ("openai", "gpt-4o-mini"): {"latency_ms": 180, "quality_score": 0.73},
    ("openai", "gpt-4o"): {"latency_ms": 360, "quality_score": 0.88},
    ("openai", "o4-mini"): {"latency_ms": 600, "quality_score": 0.85},
    ("openai", "o3-mini"): {"latency_ms": 700, "quality_score": 0.84},
    ("openai", "o3"): {"latency_ms": 1500, "quality_score": 0.97},
    ("openai", "o3-pro"): {"latency_ms": 2500, "quality_score": 0.98},
    ("openai", "gpt-5"): {"latency_ms": 850, "quality_score": 0.96},
    ("openai", "gpt-5.2"): {"latency_ms": 850, "quality_score": 0.96},
    ("anthropic", "claude-3-haiku"): {"latency_ms": 150, "quality_score": 0.68},
    ("anthropic", "claude-3.5-haiku"): {"latency_ms": 220, "quality_score": 0.75},
    ("anthropic", "claude-3.5-sonnet"): {"latency_ms": 440, "quality_score": 0.90},
    ("anthropic", "claude-sonnet-4-20250514"): {"latency_ms": 420, "quality_score": 0.92},
    ("anthropic", "claude-3.7-sonnet"): {"latency_ms": 500, "quality_score": 0.91},
    ("anthropic", "claude-3-opus"): {"latency_ms": 1200, "quality_score": 0.95},
    ("anthropic", "claude-opus-4-20250514"): {"latency_ms": 1100, "quality_score": 0.96},
    ("groq", "llama-3.1-8b-instant"): {"latency_ms": 50, "quality_score": 0.55},
    ("groq", "llama-3.3-70b-versatile"): {"latency_ms": 120, "quality_score": 0.72},
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"): {"latency_ms": 80, "quality_score": 0.65},
    ("groq", "meta-llama/llama-4-maverick-17b-128e-instruct"): {"latency_ms": 100, "quality_score": 0.70},
    ("groq", "qwen/qwen3-32b"): {"latency_ms": 90, "quality_score": 0.68},
}

_MODEL_METADATA: dict[tuple[str, str], dict[str, float]] = {}
_metadata_loaded = False


def _ensure_metadata_loaded() -> None:
    """Load model metadata from YAML, falling back to hardcoded defaults."""
    global _MODEL_METADATA, _metadata_loaded  # noqa: PLW0603
    if _metadata_loaded:
        return
    _metadata_loaded = True

    from src.dynamic_pricing import load_model_metadata
    yaml_meta = load_model_metadata()
    if yaml_meta:
        _MODEL_METADATA.update(yaml_meta)
        logger.info("Loaded model metadata from YAML (%d models).", len(yaml_meta))
    else:
        logger.info("No metadata YAML found; using hardcoded defaults.")

    for key, val in _HARDCODED_MODEL_METADATA.items():
        _MODEL_METADATA.setdefault(key, val)


def _get_model_meta(provider: str, model_id: str) -> dict[str, float]:
    """Look up latency and quality metadata for a model."""
    _ensure_metadata_loaded()
    key = (provider.lower(), model_id.lower())
    if key in _MODEL_METADATA:
        return _MODEL_METADATA[key]

    for (prov, mid), meta in _MODEL_METADATA.items():
        if prov == provider.lower() and model_id.lower().startswith(mid):
            return meta

    return {"latency_ms": 500.0, "quality_score": 0.70}


# ---------------------------------------------------------------------------
# Multi-factor ranking (AC-4.2)
# ---------------------------------------------------------------------------


def rank_models(
    profile: WorkloadProfile,
    *,
    weights: RankingWeights | None = None,
    target_providers: tuple[str, ...] | None = None,
    min_context_window: int = 0,
    registry: PricingRegistry | None = None,
    top_n: int = 15,
) -> list[RankedModel]:
    """Rank all candidate models using configurable multi-factor weights.

    Models are ranked using a normalized composite score from:
    - **cost**: Lower projected monthly cost → higher score
    - **latency**: Lower latency → higher score
    - **context_window**: Larger context → higher score
    - **quality_score**: Higher quality → higher score

    Args:
        profile: Observed workload characteristics.
        weights: Configurable ranking weights.
        target_providers: Filter to specific providers.
        min_context_window: Minimum context window filter.
        registry: Pricing registry.
        top_n: Max models to return.

    Returns:
        List of ``RankedModel`` sorted by composite score descending.
    """
    reg = registry or get_pricing_registry()
    w = (weights or RankingWeights()).normalized()

    entries = reg.all_entries()
    if target_providers:
        allowed = {p.lower() for p in target_providers}
        entries = [e for e in entries if e.provider in allowed]

    if min_context_window > 0:
        entries = [e for e in entries if e.context_window >= min_context_window]

    if not entries:
        return []

    raw_data: list[dict[str, Any]] = []
    for entry in entries:
        scenario = estimate_model_cost(profile, entry, registry=reg)
        meta = _get_model_meta(entry.provider, entry.model_id)
        raw_data.append({
            "entry": entry,
            "scenario": scenario,
            "monthly_cost": scenario.projected_monthly_cost,
            "latency_ms": meta["latency_ms"],
            "context_window": entry.context_window,
            "quality_score": meta["quality_score"],
        })

    costs = [d["monthly_cost"] for d in raw_data]
    latencies = [d["latency_ms"] for d in raw_data]
    contexts = [d["context_window"] for d in raw_data]
    qualities = [d["quality_score"] for d in raw_data]

    min_cost, max_cost = min(costs), max(costs)
    min_lat, max_lat = min(latencies), max(latencies)
    min_ctx, max_ctx = min(contexts), max(contexts)
    min_qual, max_qual = min(qualities), max(qualities)

    ranked: list[RankedModel] = []
    for data in raw_data:
        cost_score = _normalize_inverted(data["monthly_cost"], min_cost, max_cost)
        latency_score = _normalize_inverted(data["latency_ms"], min_lat, max_lat)
        context_score = _normalize(data["context_window"], min_ctx, max_ctx)
        quality_s = _normalize(data["quality_score"], min_qual, max_qual)

        composite = (
            w["cost"] * cost_score
            + w["latency"] * latency_score
            + w["context_window"] * context_score
            + w["quality_score"] * quality_s
        )

        entry: ModelPricingEntry = data["entry"]
        scenario: ScenarioResult = data["scenario"]

        ranked.append(RankedModel(
            rank=0,
            provider=entry.provider,
            model_id=entry.model_id,
            family=entry.family,
            composite_score=round(composite, 4),
            cost_score=round(cost_score, 4),
            latency_score=round(latency_score, 4),
            context_score=round(context_score, 4),
            quality_score=round(quality_s, 4),
            scenario=scenario,
        ))

    ranked.sort(key=lambda m: m.composite_score, reverse=True)

    final: list[RankedModel] = []
    for i, model in enumerate(ranked[:top_n]):
        is_primary = i == 0
        reason = _build_recommendation_reason(model, profile, is_primary=is_primary)
        final.append(RankedModel(
            rank=i + 1,
            provider=model.provider,
            model_id=model.model_id,
            family=model.family,
            composite_score=model.composite_score,
            cost_score=model.cost_score,
            latency_score=model.latency_score,
            context_score=model.context_score,
            quality_score=model.quality_score,
            scenario=model.scenario,
            is_recommended=is_primary,
            recommendation_reason=reason,
        ))

    return final


# ---------------------------------------------------------------------------
# Cost optimization recommendation (AC-4.1)
# ---------------------------------------------------------------------------


def find_cost_optimized_model(
    profile: WorkloadProfile,
    *,
    target_providers: tuple[str, ...] | None = None,
    min_quality_score: float = 0.0,
    registry: PricingRegistry | None = None,
) -> RankedModel | None:
    """Find the most cost-efficient model meeting minimum quality (AC-4.1).

    Given comparative cost analysis, when a cheaper alternative exists,
    recommends the most cost-efficient model.

    Args:
        profile: Workload profile.
        target_providers: Allowed providers.
        min_quality_score: Minimum acceptable quality (0–1).
        registry: Pricing registry.

    Returns:
        The most cost-efficient ``RankedModel`` or None.
    """
    cost_weights = RankingWeights(cost=0.70, latency=0.10, context_window=0.05, quality_score=0.15)
    ranked = rank_models(
        profile,
        weights=cost_weights,
        target_providers=target_providers,
        registry=registry,
        top_n=30,
    )

    for model in ranked:
        meta = _get_model_meta(model.provider, model.model_id)
        if meta["quality_score"] >= min_quality_score:
            return RankedModel(
                rank=1,
                provider=model.provider,
                model_id=model.model_id,
                family=model.family,
                composite_score=model.composite_score,
                cost_score=model.cost_score,
                latency_score=model.latency_score,
                context_score=model.context_score,
                quality_score=model.quality_score,
                scenario=model.scenario,
                is_recommended=True,
                recommendation_reason=(
                    f"Most cost-efficient model meeting quality threshold "
                    f"({min_quality_score:.0%}). "
                    f"Projected savings: ${model.scenario.savings_vs_current_usd:.2f}/mo "
                    f"({model.scenario.savings_vs_current_pct:+.1f}%)"
                    if model.scenario
                    else "Most cost-efficient model meeting quality threshold."
                ),
            )
    return None


# ---------------------------------------------------------------------------
# Confidence indicator (AC-4.3)
# ---------------------------------------------------------------------------


def compute_recommendation_confidence(
    *,
    profile: WorkloadProfile,
    ranked_models: list[RankedModel],
    registry: PricingRegistry | None = None,
) -> ConfidenceIndicator:
    """Compute confidence indicator for recommendation quality (AC-4.3).

    Based on:
    - Data completeness: window days, call volume, provider coverage
    - Pricing freshness: days since last pricing update
    - Token conversion certainty: presence of reasoning split, reported costs

    Args:
        profile: Workload profile.
        ranked_models: Ranked model list.
        registry: Pricing registry.

    Returns:
        ``ConfidenceIndicator`` with per-dimension scores and reasoning.
    """
    reg = registry or get_pricing_registry()

    # --- Data completeness (0–100) ---
    days_score = min(1.0, profile.window_days / 14.0) * 40.0
    volume_score = min(1.0, math.log10(max(1.0, profile.total_calls)) / 4.0) * 40.0
    has_cost = 20.0 if profile.observed_monthly_cost > 0 else 0.0
    data_completeness = min(100.0, days_score + volume_score + has_cost)

    # --- Pricing freshness (0–100) ---
    freshness_days = reg.pricing_freshness_days()
    if freshness_days <= 7:
        pricing_freshness = 100.0
    elif freshness_days <= 30:
        pricing_freshness = 80.0
    elif freshness_days <= 90:
        pricing_freshness = 50.0
    else:
        pricing_freshness = 20.0

    # --- Token conversion certainty (0–100) ---
    has_reasoning = profile.avg_reasoning_tokens_per_call > 0
    token_certainty_components = [
        30.0,  # baseline
        25.0 if profile.observed_monthly_cost > 0 else 0.0,
        20.0 if not has_reasoning else 10.0,  # simpler = more certain
        15.0 if freshness_days <= 30 else 5.0,
        10.0 if profile.total_calls >= 100 else 0.0,
    ]
    token_certainty = min(100.0, sum(token_certainty_components))

    overall = (data_completeness * 0.40 + pricing_freshness * 0.30 + token_certainty * 0.30)

    reasoning_parts: list[str] = []
    if profile.window_days < 7:
        reasoning_parts.append(f"Limited data window ({profile.window_days:.0f} days)")
    if profile.total_calls < 50:
        reasoning_parts.append(f"Low call volume ({profile.total_calls:.0f} calls)")
    if freshness_days > 30:
        reasoning_parts.append(f"Pricing data is {freshness_days} days old")
    if has_reasoning:
        reasoning_parts.append("Reasoning token estimation adds uncertainty")
    if profile.observed_monthly_cost <= 0:
        reasoning_parts.append("No reported cost data for validation")

    if not reasoning_parts:
        reasoning_parts.append("Good data coverage and fresh pricing")

    return ConfidenceIndicator(
        overall=round(overall, 1),
        data_completeness=round(data_completeness, 1),
        pricing_freshness=round(pricing_freshness, 1),
        token_conversion_certainty=round(token_certainty, 1),
        reasoning="; ".join(reasoning_parts),
    )


# ---------------------------------------------------------------------------
# Full recommendation pipeline
# ---------------------------------------------------------------------------


def run_recommendation_engine(
    unified_df: pd.DataFrame,
    *,
    source_provider: str | None = None,
    source_model: str | None = None,
    target_providers: tuple[str, ...] | None = None,
    weights: RankingWeights | None = None,
    min_context_window: int = 0,
    min_quality_score: float = 0.0,
    registry: PricingRegistry | None = None,
    top_n: int = 15,
) -> RecommendationResult:
    """Run the full recommendation pipeline.

    Combines:
    1. Workload profiling from unified_df
    2. Multi-factor ranking (AC-4.2)
    3. Cost optimization (AC-4.1)
    4. Confidence indicator (AC-4.3)

    Args:
        unified_df: Unified schema DataFrame.
        source_provider: Source provider filter.
        source_model: Source model filter.
        target_providers: Target providers for recommendation.
        weights: Custom ranking weights.
        min_context_window: Minimum context window filter.
        min_quality_score: Minimum quality for cost optimization.
        registry: Pricing registry.
        top_n: Max ranked models to return.

    Returns:
        ``RecommendationResult`` with ranked models, confidence, and DataFrame.
    """
    reg = registry or get_pricing_registry()

    profile = build_workload_profile(
        unified_df,
        provider=source_provider,
        model=source_model,
    )

    if profile.total_calls <= 0:
        return RecommendationResult(
            workload_profile=profile,
            weights_used=(weights or RankingWeights()).normalized(),
        )

    ranked = rank_models(
        profile,
        weights=weights,
        target_providers=target_providers,
        min_context_window=min_context_window,
        registry=reg,
        top_n=top_n,
    )

    confidence = compute_recommendation_confidence(
        profile=profile,
        ranked_models=ranked,
        registry=reg,
    )

    primary = ranked[0] if ranked else None

    rows: list[dict[str, Any]] = []
    for m in ranked:
        row: dict[str, Any] = {
            "Rank": m.rank,
            "Provider": m.provider.title(),
            "Model": m.model_id,
            "Family": m.family,
            "Composite Score": round(m.composite_score * 100, 1),
            "Cost Score": round(m.cost_score * 100, 1),
            "Latency Score": round(m.latency_score * 100, 1),
            "Context Score": round(m.context_score * 100, 1),
            "Quality Score": round(m.quality_score * 100, 1),
        }
        if m.scenario:
            row.update({
                "Monthly Cost (USD)": round(m.scenario.projected_monthly_cost, 2),
                "Cost/Request (USD)": round(m.scenario.cost_per_request, 6),
                "Cost/1K Tokens": round(m.scenario.cost_per_1k_tokens, 6),
                "Savings (USD/mo)": round(m.scenario.savings_vs_current_usd, 2),
                "Savings (%)": round(m.scenario.savings_vs_current_pct, 1),
                "Certainty (%)": round(m.scenario.conversion_certainty * 100, 1),
            })
        rows.append(row)

    ranked_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    return RecommendationResult(
        primary=primary,
        ranked_models=ranked,
        confidence=confidence,
        workload_profile=profile,
        weights_used=(weights or RankingWeights()).normalized(),
        ranked_df=ranked_df,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0–1 range (higher value = higher score)."""
    if max_val <= min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def _normalize_inverted(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0–1 range (lower value = higher score)."""
    if max_val <= min_val:
        return 0.5
    return 1.0 - ((value - min_val) / (max_val - min_val))


def _build_recommendation_reason(
    model: RankedModel,
    profile: WorkloadProfile,
    *,
    is_primary: bool,
) -> str:
    """Build human-readable recommendation reasoning."""
    parts: list[str] = []

    if is_primary:
        parts.append("Top-ranked model for your workload profile.")
    else:
        parts.append(f"Ranked #{model.rank}.")

    if model.scenario and model.scenario.savings_vs_current_usd > 0:
        parts.append(
            f"Estimated savings: ${model.scenario.savings_vs_current_usd:.2f}/mo "
            f"({model.scenario.savings_vs_current_pct:+.1f}%)"
        )
    elif model.scenario and model.scenario.savings_vs_current_usd < 0:
        parts.append(
            f"Cost increase: ${abs(model.scenario.savings_vs_current_usd):.2f}/mo "
            f"— justified by quality/capability gains."
        )

    strengths: list[str] = []
    if model.cost_score >= 0.8:
        strengths.append("cost-efficient")
    if model.latency_score >= 0.8:
        strengths.append("low-latency")
    if model.quality_score >= 0.8:
        strengths.append("high-quality")
    if model.context_score >= 0.8:
        strengths.append("large context")

    if strengths:
        parts.append(f"Strengths: {', '.join(strengths)}.")

    return " ".join(parts)
