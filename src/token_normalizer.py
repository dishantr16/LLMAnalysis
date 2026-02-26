"""Provider-specific token normalization and cost attribution.

Handles the fundamental difference in how providers count and bill tokens:
- **OpenAI o3/o4**: Output tokens include internal reasoning tokens billed
  at the same output rate, inflating effective cost-per-useful-output-token.
- **OpenAI GPT-5/4o**: Standard input/output billing with optional cached
  input discount.
- **Anthropic**: Cache creation, cache read, and standard tokens with
  different rates.
- **Groq**: Standard input/output with no provider-reported cost API.

When cross-provider comparisons are generated, this module applies
provider-specific conversion logic with clearly documented assumptions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from src.pricing_registry import ModelPricingEntry, PricingRegistry, get_pricing_registry

logger = logging.getLogger(__name__)

_FALLBACK_REASONING_RATIO = 0.70


class TokenType(str, Enum):
    """Canonical token type taxonomy across all providers."""

    INPUT = "input"
    OUTPUT = "output"
    CACHED_INPUT = "cached_input"
    REASONING_OUTPUT = "reasoning_output"
    INPUT_AUDIO = "input_audio"
    OUTPUT_AUDIO = "output_audio"


@dataclass(frozen=True)
class NormalizedTokenUsage:
    """Provider-agnostic representation of token usage for a single row/bucket.

    All counts are raw provider-reported values. ``effective_cost_usd`` is the
    cost computed using the pricing registry with token-type-specific rates.

    Attributes:
        provider: Provider key.
        model: Model identifier as reported by provider.
        input_tokens: Standard input tokens.
        output_tokens: Standard output tokens (excluding reasoning).
        cached_input_tokens: Tokens served from prompt cache.
        reasoning_tokens: Internal reasoning/thinking tokens (o3/o4, Claude 3.7).
        input_audio_tokens: Audio input tokens.
        output_audio_tokens: Audio output tokens.
        total_billed_tokens: Weighted sum used for cost calculation.
        effective_cost_usd: Cost attributed using token-type-specific pricing.
        conversion_certainty: 0.0–1.0 confidence in the token→cost mapping.
        assumptions: Human-readable notes about any assumptions made.
    """

    provider: str
    model: str
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    cached_input_tokens: float = 0.0
    reasoning_tokens: float = 0.0
    input_audio_tokens: float = 0.0
    output_audio_tokens: float = 0.0
    total_billed_tokens: float = 0.0
    effective_cost_usd: float = 0.0
    conversion_certainty: float = 0.5
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class TokenConversionResult:
    """Result of normalizing raw API usage into cost-attributed token breakdown."""

    normalized: NormalizedTokenUsage
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    pricing_entry_used: ModelPricingEntry | None = None


# ---------------------------------------------------------------------------
# Observed reasoning ratio tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedModelRatio:
    """Per-model observed reasoning token ratio computed from live API data.

    Attributes:
        model: Model identifier.
        ratio: Observed ``reasoning_tokens / output_tokens`` (0.0–1.0).
        sample_buckets: Number of usage buckets the ratio was computed from.
        total_reasoning_tokens: Sum of observed reasoning tokens.
        total_output_tokens: Sum of observed output tokens in those buckets.
        stddev: Standard deviation of per-bucket ratios (measures variance).
    """

    model: str
    ratio: float
    sample_buckets: int
    total_reasoning_tokens: float
    total_output_tokens: float
    stddev: float = 0.0


class ReasoningRatioTracker:
    """Builds and stores per-model reasoning token ratios from live usage data.

    Instead of relying on a hardcoded 70% estimate, this tracker computes
    actual reasoning-to-output ratios from rows where the OpenAI API reported
    ``reasoning_tokens > 0``.

    Usage::

        tracker = ReasoningRatioTracker.from_dataframe(unified_df)
        ratio_info = tracker.get_ratio("o3")
        if ratio_info is not None:
            print(f"Observed ratio: {ratio_info.ratio:.1%} from {ratio_info.sample_buckets} buckets")

    The tracker is built from **dynamic data** — the same ``unified_df`` that
    the dashboard fetches from provider APIs every session.
    """

    def __init__(self) -> None:
        self._ratios: dict[str, ObservedModelRatio] = {}

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ReasoningRatioTracker":
        """Build a tracker from a unified-schema DataFrame.

        Scans all rows where ``reasoning_tokens > 0`` and ``output_tokens > 0``
        to compute the observed ratio per model.

        Args:
            df: Unified schema DataFrame (must have ``model``,
                ``output_tokens``, ``reasoning_tokens`` columns).

        Returns:
            A populated ``ReasoningRatioTracker``.
        """
        tracker = cls()
        if df.empty:
            return tracker

        required_cols = {"model", "output_tokens", "reasoning_tokens"}
        if not required_cols.issubset(df.columns):
            return tracker

        work = df.copy()
        work["output_tokens"] = pd.to_numeric(work["output_tokens"], errors="coerce").fillna(0.0)
        work["reasoning_tokens"] = pd.to_numeric(work["reasoning_tokens"], errors="coerce").fillna(0.0)

        has_reasoning = work[(work["reasoning_tokens"] > 0) & (work["output_tokens"] > 0)]
        if has_reasoning.empty:
            return tracker

        for model_id, group in has_reasoning.groupby("model"):
            model_str = str(model_id)
            out_tokens = group["output_tokens"].to_numpy(dtype=float)
            reason_tokens = group["reasoning_tokens"].to_numpy(dtype=float)

            total_out = float(out_tokens.sum())
            total_reason = float(reason_tokens.sum())
            if total_out <= 0:
                continue

            aggregate_ratio = total_reason / total_out

            per_bucket_ratios = reason_tokens / out_tokens
            valid_mask = out_tokens > 0
            if valid_mask.sum() > 1:
                stddev = float(per_bucket_ratios[valid_mask].std())
            else:
                stddev = 0.0

            tracker._ratios[model_str.lower()] = ObservedModelRatio(
                model=model_str,
                ratio=min(1.0, max(0.0, aggregate_ratio)),
                sample_buckets=int(valid_mask.sum()),
                total_reasoning_tokens=total_reason,
                total_output_tokens=total_out,
                stddev=stddev if not math.isnan(stddev) else 0.0,
            )
            logger.info(
                "Observed reasoning ratio for %s: %.1f%% from %d buckets (stddev=%.3f)",
                model_str,
                aggregate_ratio * 100,
                int(valid_mask.sum()),
                stddev if not math.isnan(stddev) else 0.0,
            )

        return tracker

    def get_ratio(self, model: str) -> ObservedModelRatio | None:
        """Look up the observed ratio for a model.

        Tries exact match first, then prefix match (e.g. ``o3-2025-04-16`` → ``o3``).

        Args:
            model: Model identifier.

        Returns:
            ``ObservedModelRatio`` if observed data exists, else ``None``.
        """
        model_l = model.strip().lower()

        if model_l in self._ratios:
            return self._ratios[model_l]

        best: ObservedModelRatio | None = None
        best_len = 0
        for key, info in self._ratios.items():
            if model_l.startswith(key) and len(key) > best_len:
                best = info
                best_len = len(key)
        return best

    @property
    def tracked_models(self) -> list[str]:
        """Return list of models with observed reasoning ratios."""
        return [info.model for info in self._ratios.values()]

    @property
    def is_empty(self) -> bool:
        return len(self._ratios) == 0

    def summary(self) -> list[dict[str, Any]]:
        """Return a summary list suitable for display in the UI."""
        rows: list[dict[str, Any]] = []
        for info in sorted(self._ratios.values(), key=lambda r: r.model):
            rows.append({
                "Model": info.model,
                "Observed Ratio": f"{info.ratio:.1%}",
                "Sample Buckets": info.sample_buckets,
                "Total Reasoning Tokens": f"{info.total_reasoning_tokens:,.0f}",
                "Total Output Tokens": f"{info.total_output_tokens:,.0f}",
                "Stddev": f"{info.stddev:.3f}",
            })
        return rows


# ---------------------------------------------------------------------------
# Provider-specific token extraction strategies
# ---------------------------------------------------------------------------

_OPENAI_REASONING_MODEL_FAMILIES = frozenset({"o3", "o3-mini", "o3-pro", "o4-mini", "o1", "o1-mini", "o1-pro"})

_OPENAI_REASONING_KEYWORD_HINTS = frozenset({"o3", "o4", "o1"})


def _is_reasoning_model(provider: str, model: str) -> bool:
    """Detect if a model uses internal reasoning tokens."""
    if provider.lower() != "openai":
        return False
    model_l = model.lower()
    if model_l in _OPENAI_REASONING_MODEL_FAMILIES:
        return True
    return any(hint in model_l for hint in _OPENAI_REASONING_KEYWORD_HINTS)


def extract_openai_tokens(
    row: dict[str, Any],
    *,
    registry: PricingRegistry | None = None,
    reasoning_tracker: ReasoningRatioTracker | None = None,
) -> TokenConversionResult:
    """Normalize an OpenAI usage row into token-type-specific attribution.

    For o3/o4 models, ``output_tokens`` from the API includes reasoning tokens.
    Resolution priority for the reasoning/output split:

    1. **Exact** — API reported ``reasoning_tokens > 0``: use as-is (certainty 0.90).
    2. **Observed** — ``ReasoningRatioTracker`` has a ratio learned from other
       rows in the same fetch window: use observed ratio (certainty 0.70–0.85
       depending on sample size).
    3. **Fallback** — No observed data at all: use the static default
       ``_FALLBACK_REASONING_RATIO`` (certainty 0.55).

    Args:
        row: Dict with keys from the usage API or unified schema.
        registry: Pricing registry to use for cost calculation.
        reasoning_tracker: Observed reasoning ratio tracker built from live
            data.  Pass ``None`` to skip observed lookup.

    Returns:
        ``TokenConversionResult`` with per-token-type costs.
    """
    reg = registry or get_pricing_registry()
    model = str(row.get("model", "unknown"))
    provider = "openai"

    raw_input = _safe_float(row.get("input_tokens", 0))
    raw_output = _safe_float(row.get("output_tokens", 0))
    raw_cached = _safe_float(row.get("cached_input_tokens", 0))
    raw_reasoning = _safe_float(row.get("reasoning_tokens", 0))
    raw_input_audio = _safe_float(row.get("input_audio_tokens", 0))
    raw_output_audio = _safe_float(row.get("output_audio_tokens", 0))

    entry = reg.lookup(provider, model)
    assumptions: list[str] = []
    certainty = 0.9

    is_reasoning = _is_reasoning_model(provider, model)

    if is_reasoning and raw_reasoning == 0 and raw_output > 0:
        ratio, certainty, assumption = _resolve_reasoning_ratio(
            model, reasoning_tracker
        )
        raw_reasoning = raw_output * ratio
        raw_output = raw_output * (1.0 - ratio)
        assumptions.append(assumption)

    effective_input = max(0.0, raw_input - raw_cached)

    cost_breakdown: dict[str, float] = {}
    total_cost = 0.0

    if entry:
        p = entry.pricing
        cost_input = (effective_input / 1_000_000) * p.input
        cost_cached = (raw_cached / 1_000_000) * (p.cached_input if p.cached_input is not None else p.input)
        cost_output = (raw_output / 1_000_000) * p.output

        reasoning_rate = p.reasoning_output if p.reasoning_output is not None else p.output
        cost_reasoning = (raw_reasoning / 1_000_000) * reasoning_rate

        cost_audio_in = (raw_input_audio / 1_000_000) * (p.input_audio or 0.0)
        cost_audio_out = (raw_output_audio / 1_000_000) * (p.output_audio or 0.0)

        cost_breakdown = {
            "input": round(cost_input, 8),
            "cached_input": round(cost_cached, 8),
            "output": round(cost_output, 8),
            "reasoning_output": round(cost_reasoning, 8),
            "input_audio": round(cost_audio_in, 8),
            "output_audio": round(cost_audio_out, 8),
        }
        total_cost = sum(cost_breakdown.values())
    else:
        assumptions.append(f"No pricing entry found for {provider}/{model}")
        certainty = 0.2

    billed_total = effective_input + raw_cached + raw_output + raw_reasoning + raw_input_audio + raw_output_audio

    normalized = NormalizedTokenUsage(
        provider=provider,
        model=model,
        input_tokens=effective_input,
        output_tokens=raw_output,
        cached_input_tokens=raw_cached,
        reasoning_tokens=raw_reasoning,
        input_audio_tokens=raw_input_audio,
        output_audio_tokens=raw_output_audio,
        total_billed_tokens=billed_total,
        effective_cost_usd=round(total_cost, 8),
        conversion_certainty=certainty,
        assumptions=tuple(assumptions),
    )
    return TokenConversionResult(
        normalized=normalized,
        cost_breakdown=cost_breakdown,
        pricing_entry_used=entry,
    )


def extract_anthropic_tokens(
    row: dict[str, Any],
    *,
    registry: PricingRegistry | None = None,
) -> TokenConversionResult:
    """Normalize an Anthropic usage row.

    Anthropic reports cache_creation_input_tokens and cache_read_input_tokens
    separately. Cache reads are cheaper; cache creation is at input rate.

    Args:
        row: Dict with keys from Anthropic usage API.
        registry: Pricing registry to use.

    Returns:
        ``TokenConversionResult`` with per-token-type costs.
    """
    reg = registry or get_pricing_registry()
    model = str(row.get("model", "unknown"))
    provider = "anthropic"

    raw_input = _safe_float(row.get("input_tokens", 0))
    raw_output = _safe_float(row.get("output_tokens", 0))
    cache_read = _safe_float(
        row.get("cache_read_input_tokens", row.get("cached_input_tokens", 0))
    )
    cache_creation = _safe_float(row.get("cache_creation_input_tokens", 0))

    entry = reg.lookup(provider, model)
    assumptions: list[str] = []
    certainty = 0.85

    effective_input = max(0.0, raw_input - cache_read - cache_creation)

    cost_breakdown: dict[str, float] = {}
    total_cost = 0.0

    if entry:
        p = entry.pricing
        cost_input = (effective_input / 1_000_000) * p.input
        cost_cache_read = (cache_read / 1_000_000) * (p.cached_input if p.cached_input is not None else p.input)
        cost_cache_create = (cache_creation / 1_000_000) * p.input
        cost_output = (raw_output / 1_000_000) * p.output

        cost_breakdown = {
            "input": round(cost_input, 8),
            "cached_input_read": round(cost_cache_read, 8),
            "cache_creation": round(cost_cache_create, 8),
            "output": round(cost_output, 8),
        }
        total_cost = sum(cost_breakdown.values())
    else:
        assumptions.append(f"No pricing entry found for {provider}/{model}")
        certainty = 0.2

    billed_total = effective_input + cache_read + cache_creation + raw_output

    normalized = NormalizedTokenUsage(
        provider=provider,
        model=model,
        input_tokens=effective_input,
        output_tokens=raw_output,
        cached_input_tokens=cache_read + cache_creation,
        reasoning_tokens=0.0,
        total_billed_tokens=billed_total,
        effective_cost_usd=round(total_cost, 8),
        conversion_certainty=certainty,
        assumptions=tuple(assumptions),
    )
    return TokenConversionResult(
        normalized=normalized,
        cost_breakdown=cost_breakdown,
        pricing_entry_used=entry,
    )


def extract_groq_tokens(
    row: dict[str, Any],
    *,
    registry: PricingRegistry | None = None,
) -> TokenConversionResult:
    """Normalize a Groq usage row (standard input/output only)."""
    reg = registry or get_pricing_registry()
    model = str(row.get("model", "unknown"))
    provider = "groq"

    raw_input = _safe_float(row.get("input_tokens", 0))
    raw_output = _safe_float(row.get("output_tokens", 0))

    entry = reg.lookup(provider, model)
    assumptions: list[str] = []
    certainty = 0.8

    cost_breakdown: dict[str, float] = {}
    total_cost = 0.0

    if entry:
        p = entry.pricing
        cost_input = (raw_input / 1_000_000) * p.input
        cost_output = (raw_output / 1_000_000) * p.output
        cost_breakdown = {
            "input": round(cost_input, 8),
            "output": round(cost_output, 8),
        }
        total_cost = sum(cost_breakdown.values())
    else:
        assumptions.append(f"No pricing entry found for {provider}/{model}")
        certainty = 0.2

    normalized = NormalizedTokenUsage(
        provider=provider,
        model=model,
        input_tokens=raw_input,
        output_tokens=raw_output,
        total_billed_tokens=raw_input + raw_output,
        effective_cost_usd=round(total_cost, 8),
        conversion_certainty=certainty,
        assumptions=tuple(assumptions),
    )
    return TokenConversionResult(
        normalized=normalized,
        cost_breakdown=cost_breakdown,
        pricing_entry_used=entry,
    )


# ---------------------------------------------------------------------------
# Unified normalizer dispatcher
# ---------------------------------------------------------------------------

def normalize_token_usage(
    provider: str,
    row: dict[str, Any],
    *,
    registry: PricingRegistry | None = None,
    reasoning_tracker: ReasoningRatioTracker | None = None,
) -> TokenConversionResult:
    """Dispatch to the correct provider-specific token normalizer.

    Args:
        provider: Provider key.
        row: Usage row as a dict.
        registry: Optional custom pricing registry.
        reasoning_tracker: Observed reasoning ratio tracker (OpenAI only).

    Returns:
        ``TokenConversionResult`` with normalized tokens and attributed cost.
    """
    provider_l = provider.strip().lower()
    if provider_l == "openai":
        return extract_openai_tokens(
            row, registry=registry, reasoning_tracker=reasoning_tracker,
        )
    if provider_l == "anthropic":
        return extract_anthropic_tokens(row, registry=registry)
    if provider_l == "groq":
        return extract_groq_tokens(row, registry=registry)

    logger.warning("No token extractor for provider '%s'; returning zero-cost result.", provider)
    return TokenConversionResult(
        normalized=NormalizedTokenUsage(
            provider=provider,
            model=str(row.get("model", "unknown")),
            conversion_certainty=0.0,
            assumptions=(f"Unsupported provider: {provider}",),
        ),
    )


def compute_token_conversion_certainty(
    *,
    has_pricing: bool,
    has_reasoning_split: bool,
    has_reported_cost: bool,
    pricing_freshness_days: int,
) -> float:
    """Compute an aggregate certainty score for token-to-cost conversion.

    Used by the confidence indicator (AC-4.3). Scale: 0.0–1.0.

    Args:
        has_pricing: Whether a pricing entry exists for this model.
        has_reasoning_split: Whether reasoning tokens are explicitly reported.
        has_reported_cost: Whether provider reports actual billing cost.
        pricing_freshness_days: Days since pricing data was updated.

    Returns:
        Float certainty between 0.0 and 1.0.
    """
    score = 0.3  # baseline

    if has_pricing:
        score += 0.25
    if has_reasoning_split:
        score += 0.15
    if has_reported_cost:
        score += 0.20

    if pricing_freshness_days <= 7:
        score += 0.10
    elif pricing_freshness_days <= 30:
        score += 0.05

    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Reasoning ratio resolution
# ---------------------------------------------------------------------------

_MIN_BUCKETS_FOR_OBSERVED = 3


def _resolve_reasoning_ratio(
    model: str,
    tracker: ReasoningRatioTracker | None,
) -> tuple[float, float, str]:
    """Determine the best reasoning ratio for a model when not reported by API.

    Resolution priority:
    1. Observed ratio from tracker (if ≥ ``_MIN_BUCKETS_FOR_OBSERVED`` samples).
    2. Static fallback ``_FALLBACK_REASONING_RATIO``.

    Args:
        model: Model identifier.
        tracker: Observed ratio tracker (may be ``None``).

    Returns:
        Tuple of ``(ratio, certainty, assumption_text)``.
    """
    if tracker is not None:
        observed = tracker.get_ratio(model)
        if observed is not None and observed.sample_buckets >= _MIN_BUCKETS_FOR_OBSERVED:
            # Certainty scales with sample size: 3 buckets → 0.70, 50+ → 0.85
            sample_factor = min(1.0, math.log10(max(1, observed.sample_buckets)) / 1.7)
            certainty = 0.70 + (sample_factor * 0.15)

            # High variance (stddev > 0.15) reduces certainty
            if observed.stddev > 0.15:
                certainty -= 0.05

            return (
                observed.ratio,
                round(min(0.85, max(0.65, certainty)), 2),
                f"Reasoning ratio {observed.ratio:.1%} observed from "
                f"{observed.sample_buckets} API buckets for {observed.model} "
                f"(stddev={observed.stddev:.3f})",
            )

        if observed is not None:
            return (
                observed.ratio,
                0.60,
                f"Reasoning ratio {observed.ratio:.1%} from only "
                f"{observed.sample_buckets} bucket(s) for {observed.model} "
                f"— low confidence due to small sample",
            )

    return (
        _FALLBACK_REASONING_RATIO,
        0.55,
        f"Reasoning tokens estimated at {_FALLBACK_REASONING_RATIO:.0%} of "
        f"reported output (static fallback — no observed data available)",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    """Coerce to float, returning 0.0 on failure."""
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
