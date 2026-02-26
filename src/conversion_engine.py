"""Cross-model workload conversion engine.

Solves the core FinOps question: "If Model A consumed X tokens for Y queries
costing $Z, what would Model B consume and cost for the **same tasks**?"

The naive approach (apply Model B pricing to Model A token counts) is wrong
because:

1. **Tokenizer differences** — the same text produces different token counts
   across tokenizers (OpenAI o200k_base vs Anthropic vs Llama SentencePiece).
2. **Output length differences** — different models produce different length
   responses for the same prompt. Budget models are more concise; frontier
   models are more detailed.
3. **Reasoning overhead** — o3/o4 models add internal reasoning tokens (60-80%
   of billed output) that non-reasoning models don't produce.
4. **Quality gap** — a cheaper model may need more iterations (retries, human
   review) to achieve the same outcome, increasing effective cost.

This module applies empirically-derived conversion factors to produce a more
accurate "what-if" estimate. Every factor includes a documented confidence
level and assumption.

Sources for conversion ratios:
    - Tokenizer vocabulary sizes and published efficiency comparisons
    - OpenAI, Anthropic, and Meta documentation on tokenization
    - Observed output length patterns from public benchmarks
    - Quality scores from published benchmark results and leaderboards
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.pricing_registry import ModelPricingEntry, PricingRegistry, get_pricing_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer profiles
# ---------------------------------------------------------------------------


class TokenizerFamily(str, Enum):
    """Known tokenizer families across providers."""

    O200K_BASE = "o200k_base"
    CL100K_BASE = "cl100k_base"
    ANTHROPIC = "anthropic"
    LLAMA_SP = "llama_sentencepiece"
    QWEN = "qwen"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TokenizerProfile:
    """Characterization of a model's tokenizer for cross-model conversion.

    Attributes:
        family: Tokenizer family identifier.
        vocab_size: Approximate vocabulary size.
        efficiency_vs_o200k: Ratio of token count relative to o200k_base for
            the same English text. <1.0 means fewer tokens (more efficient),
            >1.0 means more tokens.
        confidence: How confident we are in this ratio (0.0–1.0).
        source: Where this estimate comes from.
    """

    family: TokenizerFamily
    vocab_size: int
    efficiency_vs_o200k: float
    confidence: float = 0.8
    source: str = ""


_TOKENIZER_PROFILES: dict[TokenizerFamily, TokenizerProfile] = {
    TokenizerFamily.O200K_BASE: TokenizerProfile(
        family=TokenizerFamily.O200K_BASE,
        vocab_size=200_000,
        efficiency_vs_o200k=1.0,
        confidence=1.0,
        source="Baseline — OpenAI o200k_base (GPT-4o, o3, GPT-4.1, GPT-5)",
    ),
    TokenizerFamily.CL100K_BASE: TokenizerProfile(
        family=TokenizerFamily.CL100K_BASE,
        vocab_size=100_256,
        efficiency_vs_o200k=1.05,
        confidence=0.90,
        source=(
            "Empirically measured: cl100k produces ~5% more tokens than o200k "
            "for the same English text due to smaller vocabulary"
        ),
    ),
    TokenizerFamily.ANTHROPIC: TokenizerProfile(
        family=TokenizerFamily.ANTHROPIC,
        vocab_size=100_000,
        efficiency_vs_o200k=1.03,
        confidence=0.75,
        source=(
            "Estimated from public tokenizer comparisons. Anthropic's tokenizer "
            "has similar efficiency to cl100k_base with minor differences"
        ),
    ),
    TokenizerFamily.LLAMA_SP: TokenizerProfile(
        family=TokenizerFamily.LLAMA_SP,
        vocab_size=128_000,
        efficiency_vs_o200k=1.10,
        confidence=0.70,
        source=(
            "Llama SentencePiece tokenizer produces ~10% more tokens than o200k "
            "for English text. Ratio varies more for non-English and code"
        ),
    ),
    TokenizerFamily.QWEN: TokenizerProfile(
        family=TokenizerFamily.QWEN,
        vocab_size=152_064,
        efficiency_vs_o200k=1.05,
        confidence=0.65,
        source="Estimated from Qwen2 tokenizer vocabulary analysis",
    ),
    TokenizerFamily.UNKNOWN: TokenizerProfile(
        family=TokenizerFamily.UNKNOWN,
        vocab_size=100_000,
        efficiency_vs_o200k=1.05,
        confidence=0.40,
        source="Unknown tokenizer — using conservative average estimate",
    ),
}

_MODEL_TOKENIZER_MAP: dict[str, TokenizerFamily] = {
    # OpenAI — o200k_base
    "gpt-4o": TokenizerFamily.O200K_BASE,
    "gpt-4o-mini": TokenizerFamily.O200K_BASE,
    "gpt-4.1": TokenizerFamily.O200K_BASE,
    "gpt-4.1-mini": TokenizerFamily.O200K_BASE,
    "gpt-4.1-nano": TokenizerFamily.O200K_BASE,
    "gpt-5": TokenizerFamily.O200K_BASE,
    "o3": TokenizerFamily.O200K_BASE,
    "o3-pro": TokenizerFamily.O200K_BASE,
    "o3-mini": TokenizerFamily.O200K_BASE,
    "o4-mini": TokenizerFamily.O200K_BASE,
    # OpenAI legacy — cl100k_base
    "gpt-4": TokenizerFamily.CL100K_BASE,
    "gpt-4-turbo": TokenizerFamily.CL100K_BASE,
    "gpt-3.5-turbo": TokenizerFamily.CL100K_BASE,
    # Anthropic
    "claude-sonnet-4": TokenizerFamily.ANTHROPIC,
    "claude-opus-4": TokenizerFamily.ANTHROPIC,
    "claude-3.7-sonnet": TokenizerFamily.ANTHROPIC,
    "claude-3.5-sonnet": TokenizerFamily.ANTHROPIC,
    "claude-3.5-haiku": TokenizerFamily.ANTHROPIC,
    "claude-3-opus": TokenizerFamily.ANTHROPIC,
    "claude-3-haiku": TokenizerFamily.ANTHROPIC,
    # Groq / Meta
    "llama-3.1": TokenizerFamily.LLAMA_SP,
    "llama-3.3": TokenizerFamily.LLAMA_SP,
    "llama-4": TokenizerFamily.LLAMA_SP,
    # Qwen
    "qwen3": TokenizerFamily.QWEN,
    "qwen": TokenizerFamily.QWEN,
}


def _resolve_tokenizer(provider: str, model_id: str) -> TokenizerProfile:
    """Resolve a model identifier to its tokenizer profile."""
    model_l = model_id.strip().lower()

    if model_l in _MODEL_TOKENIZER_MAP:
        return _TOKENIZER_PROFILES[_MODEL_TOKENIZER_MAP[model_l]]

    # Prefix matching
    best_key = ""
    for key in _MODEL_TOKENIZER_MAP:
        if model_l.startswith(key) and len(key) > len(best_key):
            best_key = key
    if best_key:
        return _TOKENIZER_PROFILES[_MODEL_TOKENIZER_MAP[best_key]]

    # Provider fallback
    provider_default = {
        "openai": TokenizerFamily.O200K_BASE,
        "anthropic": TokenizerFamily.ANTHROPIC,
        "groq": TokenizerFamily.LLAMA_SP,
    }
    return _TOKENIZER_PROFILES.get(
        provider_default.get(provider.lower(), TokenizerFamily.UNKNOWN),
        _TOKENIZER_PROFILES[TokenizerFamily.UNKNOWN],
    )


# ---------------------------------------------------------------------------
# Output behavior profiles
# ---------------------------------------------------------------------------


class OutputBehavior(str, Enum):
    """Categorization of model output length behavior."""

    CONCISE = "concise"
    STANDARD = "standard"
    DETAILED = "detailed"
    REASONING = "reasoning"


@dataclass(frozen=True)
class OutputProfile:
    """Characterization of a model's output generation behavior.

    Attributes:
        behavior: Output length category.
        output_length_factor: Multiplier vs. a standard model. <1.0 means
            shorter responses, >1.0 means longer.
        reasoning_ratio: Fraction of output tokens that are internal reasoning
            (0.0 for non-reasoning models, 0.6–0.8 for o-series).
        source: Documentation of where this estimate comes from.
    """

    behavior: OutputBehavior
    output_length_factor: float
    reasoning_ratio: float = 0.0
    source: str = ""


_OUTPUT_PROFILES: dict[str, OutputProfile] = {
    # Budget / nano — shorter, more concise responses
    "gpt-4.1-nano": OutputProfile(
        OutputBehavior.CONCISE, 0.75, source="Budget models produce ~25% fewer output tokens",
    ),
    "gpt-4o-mini": OutputProfile(
        OutputBehavior.CONCISE, 0.80, source="Mini models produce ~20% fewer output tokens",
    ),
    "gpt-4.1-mini": OutputProfile(
        OutputBehavior.CONCISE, 0.80, source="Mini models produce ~20% fewer output tokens",
    ),
    "claude-3-haiku": OutputProfile(
        OutputBehavior.CONCISE, 0.75, source="Haiku optimized for speed and brevity",
    ),
    "claude-3.5-haiku": OutputProfile(
        OutputBehavior.CONCISE, 0.80, source="Haiku produces shorter responses",
    ),
    # Standard tier — baseline output length
    "gpt-4o": OutputProfile(
        OutputBehavior.STANDARD, 1.0, source="Baseline standard output length",
    ),
    "gpt-4.1": OutputProfile(
        OutputBehavior.STANDARD, 0.95, source="4.1 is slightly more concise than 4o",
    ),
    "claude-3.5-sonnet": OutputProfile(
        OutputBehavior.STANDARD, 1.0, source="Sonnet produces comparable output length to GPT-4o",
    ),
    "claude-sonnet-4": OutputProfile(
        OutputBehavior.STANDARD, 1.0, source="Sonnet 4 comparable to GPT-4o in verbosity",
    ),
    "claude-3.7-sonnet": OutputProfile(
        OutputBehavior.STANDARD, 1.0, source="Sonnet 3.7 comparable output length",
    ),
    # Detailed tier — longer, more thorough responses
    "gpt-5": OutputProfile(
        OutputBehavior.DETAILED, 1.15, source="Frontier models produce ~15% more detailed output",
    ),
    "claude-3-opus": OutputProfile(
        OutputBehavior.DETAILED, 1.10, source="Opus produces more thorough, longer responses",
    ),
    "claude-opus-4": OutputProfile(
        OutputBehavior.DETAILED, 1.15, source="Opus 4 produces detailed responses",
    ),
    # Reasoning tier — output includes internal reasoning tokens
    "o3": OutputProfile(
        OutputBehavior.REASONING, 1.0, 0.65,
        source="o3 visible output is comparable, but 65% of billed output is reasoning",
    ),
    "o3-pro": OutputProfile(
        OutputBehavior.REASONING, 1.10, 0.75,
        source="o3-pro uses extended thinking — 75% reasoning overhead",
    ),
    "o3-mini": OutputProfile(
        OutputBehavior.REASONING, 0.85, 0.60,
        source="o3-mini is more concise with lighter reasoning",
    ),
    "o4-mini": OutputProfile(
        OutputBehavior.REASONING, 0.85, 0.60,
        source="o4-mini — efficient reasoning model",
    ),
    # Groq / open-source
    "llama-3.1-8b-instant": OutputProfile(
        OutputBehavior.CONCISE, 0.70, source="Small model — shorter, less detailed responses",
    ),
    "llama-3.3-70b-versatile": OutputProfile(
        OutputBehavior.STANDARD, 0.90, source="70B model — near-standard output length",
    ),
}


def _resolve_output_profile(model_id: str) -> OutputProfile:
    """Resolve a model identifier to its output behavior profile."""
    model_l = model_id.strip().lower()

    if model_l in _OUTPUT_PROFILES:
        return _OUTPUT_PROFILES[model_l]

    best_key = ""
    for key in _OUTPUT_PROFILES:
        if model_l.startswith(key) and len(key) > len(best_key):
            best_key = key
    if best_key:
        return _OUTPUT_PROFILES[best_key]

    return OutputProfile(OutputBehavior.STANDARD, 1.0, source="Unknown model — assuming standard behavior")


# ---------------------------------------------------------------------------
# Conversion result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionFactors:
    """The adjustment factors applied when converting from source to target model.

    Attributes:
        input_token_ratio: Target input tokens / source input tokens.
            Driven by tokenizer differences.
        output_token_ratio: Target visible-output tokens / source visible-output
            tokens. Driven by model output behavior.
        reasoning_overhead: Expected reasoning tokens on target model as a
            fraction of visible output. 0.0 for non-reasoning models.
        quality_adjustment: Effective call multiplier to achieve same outcome
            quality. >1.0 means target needs more calls.
        overall_confidence: Composite confidence in the conversion (0.0–1.0).
        assumptions: All assumptions made, with sources.
    """

    input_token_ratio: float = 1.0
    output_token_ratio: float = 1.0
    reasoning_overhead: float = 0.0
    quality_adjustment: float = 1.0
    overall_confidence: float = 0.5
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConvertedWorkload:
    """Result of converting a source workload to a target model.

    All values represent the **estimated** usage on the target model for the
    same set of tasks.

    Attributes:
        source_provider: Original provider.
        source_model: Original model.
        target_provider: Target provider.
        target_model: Target model.
        source_total_calls: Original call count.
        effective_calls: Calls needed on target (adjusted for quality).
        est_input_tokens_per_call: Estimated input tokens per call on target.
        est_output_tokens_per_call: Estimated visible output tokens on target.
        est_reasoning_tokens_per_call: Estimated reasoning tokens on target.
        est_total_tokens_per_call: Total billed tokens per call on target.
        est_cost_per_call: Estimated cost per call on target.
        est_monthly_cost: Estimated monthly cost on target.
        source_monthly_cost: Observed monthly cost on source.
        savings_usd: Monthly savings (positive = target is cheaper).
        savings_pct: Savings percentage.
        conversion_factors: The factors applied.
        conversion_confidence: Overall confidence in the conversion.
    """

    source_provider: str
    source_model: str
    target_provider: str
    target_model: str
    source_total_calls: float = 0.0
    effective_calls: float = 0.0
    est_input_tokens_per_call: float = 0.0
    est_output_tokens_per_call: float = 0.0
    est_reasoning_tokens_per_call: float = 0.0
    est_total_tokens_per_call: float = 0.0
    est_cost_per_call: float = 0.0
    est_monthly_cost: float = 0.0
    source_monthly_cost: float = 0.0
    savings_usd: float = 0.0
    savings_pct: float = 0.0
    conversion_factors: ConversionFactors = field(default_factory=ConversionFactors)
    conversion_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------


def compute_conversion_factors(
    source_provider: str,
    source_model: str,
    target_provider: str,
    target_model: str,
    *,
    observed_reasoning_ratio: float | None = None,
    target_entry: ModelPricingEntry | None = None,
) -> ConversionFactors:
    """Compute the conversion factors between a source and target model.

    Args:
        source_provider: Source provider key.
        source_model: Source model identifier.
        target_provider: Target provider key.
        target_model: Target model identifier.
        observed_reasoning_ratio: Dynamically observed reasoning ratio for
            the source model (overrides profile default).
        target_entry: Target model pricing entry (for reasoning capability check).

    Returns:
        ``ConversionFactors`` with all adjustment ratios and assumptions.
    """
    assumptions: list[str] = []
    confidences: list[float] = []

    # --- Tokenizer ratio ---
    src_tok = _resolve_tokenizer(source_provider, source_model)
    tgt_tok = _resolve_tokenizer(target_provider, target_model)

    if src_tok.family == tgt_tok.family:
        input_ratio = 1.0
        assumptions.append(
            f"Same tokenizer ({src_tok.family.value}) — input token count preserved exactly"
        )
        confidences.append(1.0)
    else:
        input_ratio = tgt_tok.efficiency_vs_o200k / src_tok.efficiency_vs_o200k
        assumptions.append(
            f"Tokenizer conversion: {src_tok.family.value} → {tgt_tok.family.value}, "
            f"ratio={input_ratio:.3f} ({src_tok.source}; {tgt_tok.source})"
        )
        confidences.append(min(src_tok.confidence, tgt_tok.confidence))

    # --- Output length ratio ---
    src_out = _resolve_output_profile(source_model)
    tgt_out = _resolve_output_profile(target_model)

    # Visible output = output tokens minus reasoning for reasoning models
    src_visible_factor = src_out.output_length_factor
    tgt_visible_factor = tgt_out.output_length_factor

    output_ratio = tgt_visible_factor / src_visible_factor if src_visible_factor > 0 else 1.0
    assumptions.append(
        f"Output length adjustment: source={src_out.behavior.value} "
        f"(×{src_visible_factor:.2f}), target={tgt_out.behavior.value} "
        f"(×{tgt_visible_factor:.2f}), ratio={output_ratio:.3f}"
    )
    confidences.append(0.65)

    # --- Reasoning overhead ---
    src_reasoning = observed_reasoning_ratio if observed_reasoning_ratio is not None else src_out.reasoning_ratio
    tgt_reasoning = tgt_out.reasoning_ratio

    target_supports_reasoning = False
    if target_entry is not None:
        target_supports_reasoning = target_entry.supports_reasoning
    elif tgt_out.reasoning_ratio > 0:
        target_supports_reasoning = True

    if src_reasoning > 0 and not target_supports_reasoning:
        reasoning_overhead = 0.0
        assumptions.append(
            f"Source model uses reasoning tokens ({src_reasoning:.0%} of output). "
            f"Target model does NOT reason — reasoning overhead removed. "
            f"Quality may decrease for complex tasks."
        )
        confidences.append(0.70)
    elif src_reasoning == 0 and target_supports_reasoning:
        reasoning_overhead = tgt_reasoning if tgt_reasoning > 0 else 0.65
        assumptions.append(
            f"Source model does NOT produce reasoning tokens. Target model is a "
            f"reasoning model — estimated {reasoning_overhead:.0%} reasoning overhead "
            f"will be added to output tokens."
        )
        confidences.append(0.55)
    elif src_reasoning > 0 and target_supports_reasoning:
        reasoning_overhead = tgt_reasoning if tgt_reasoning > 0 else src_reasoning
        assumptions.append(
            f"Both models use reasoning. Source ratio={src_reasoning:.0%}, "
            f"target ratio={reasoning_overhead:.0%}."
        )
        confidences.append(0.70)
    else:
        reasoning_overhead = 0.0
        confidences.append(0.90)

    # --- Quality adjustment ---
    from src.recommendation_engine import _get_model_meta

    src_meta = _get_model_meta(source_provider, source_model)
    tgt_meta = _get_model_meta(target_provider, target_model)

    src_quality = src_meta.get("quality_score", 0.70)
    tgt_quality = tgt_meta.get("quality_score", 0.70)

    if tgt_quality >= src_quality:
        quality_adj = 1.0
        assumptions.append(
            f"Target quality ({tgt_quality:.2f}) ≥ source ({src_quality:.2f}) — "
            f"no additional iterations needed."
        )
    else:
        gap = src_quality - tgt_quality
        # Graduated penalty: small gap → minimal adjustment, large gap → significant
        quality_adj = 1.0 + (gap * 1.5)
        quality_adj = min(quality_adj, 2.0)  # cap at 2x
        assumptions.append(
            f"Target quality ({tgt_quality:.2f}) < source ({src_quality:.2f}) — "
            f"estimated {quality_adj:.2f}× effective calls needed for equivalent outcome. "
            f"This is task-dependent: simple tasks may not need extra calls."
        )
    confidences.append(0.50)

    overall_confidence = _geometric_mean(confidences)

    return ConversionFactors(
        input_token_ratio=round(input_ratio, 4),
        output_token_ratio=round(output_ratio, 4),
        reasoning_overhead=round(reasoning_overhead, 4),
        quality_adjustment=round(quality_adj, 4),
        overall_confidence=round(overall_confidence, 3),
        assumptions=tuple(assumptions),
    )


def estimate_converted_workload(
    source_provider: str,
    source_model: str,
    target_provider: str,
    target_model: str,
    *,
    avg_input_tokens: float,
    avg_output_tokens: float,
    avg_reasoning_tokens: float = 0.0,
    total_calls: float,
    monthly_calls: float,
    source_monthly_cost: float,
    cached_ratio: float = 0.0,
    observed_reasoning_ratio: float | None = None,
    registry: PricingRegistry | None = None,
) -> ConvertedWorkload:
    """Estimate what a source workload would cost on a target model.

    This is the core conversion function. Unlike ``estimate_model_cost`` which
    naively transfers token counts, this function adjusts for:

    - Tokenizer differences (different token counts for same text)
    - Output length behavior (budget models are more concise)
    - Reasoning overhead (added or removed depending on target)
    - Quality gap (cheaper model may need more iterations)

    Args:
        source_provider: Provider of the observed workload.
        source_model: Model of the observed workload.
        target_provider: Target provider.
        target_model: Target model.
        avg_input_tokens: Average input tokens per call (source model).
        avg_output_tokens: Average output tokens per call (source model,
            excluding reasoning).
        avg_reasoning_tokens: Average reasoning tokens per call (source model).
        total_calls: Total calls in observation window.
        monthly_calls: Projected monthly calls.
        source_monthly_cost: Observed monthly cost on source model.
        cached_ratio: Fraction of input tokens served from cache.
        observed_reasoning_ratio: Dynamically observed reasoning ratio.
        registry: Pricing registry.

    Returns:
        ``ConvertedWorkload`` with estimated usage and cost on target model.
    """
    reg = registry or get_pricing_registry()
    target_entry = reg.lookup(target_provider, target_model)

    factors = compute_conversion_factors(
        source_provider, source_model,
        target_provider, target_model,
        observed_reasoning_ratio=observed_reasoning_ratio,
        target_entry=target_entry,
    )

    # For reasoning source models, compute the visible (non-reasoning) output
    src_total_output = avg_output_tokens + avg_reasoning_tokens
    if avg_reasoning_tokens > 0 and src_total_output > 0:
        src_visible_output = avg_output_tokens
    else:
        src_visible_output = avg_output_tokens

    # Apply conversion factors
    est_input = avg_input_tokens * factors.input_token_ratio
    est_visible_output = src_visible_output * factors.output_token_ratio

    # Apply reasoning overhead to target
    if factors.reasoning_overhead > 0:
        est_reasoning = est_visible_output * (factors.reasoning_overhead / (1.0 - factors.reasoning_overhead))
    else:
        est_reasoning = 0.0

    # Quality-adjusted effective calls
    effective_monthly = monthly_calls * factors.quality_adjustment

    # Cost calculation
    est_cost_per_call = 0.0
    if target_entry:
        p = target_entry.pricing
        eff_input = est_input * (1.0 - cached_ratio)
        cached_input = est_input * cached_ratio
        cached_rate = p.cached_input if p.cached_input is not None else p.input
        reasoning_rate = p.reasoning_output if p.reasoning_output is not None else p.output

        est_cost_per_call = (
            (eff_input / 1_000_000) * p.input
            + (cached_input / 1_000_000) * cached_rate
            + (est_visible_output / 1_000_000) * p.output
            + (est_reasoning / 1_000_000) * reasoning_rate
        )

    est_monthly = est_cost_per_call * effective_monthly
    savings = source_monthly_cost - est_monthly
    savings_pct = (savings / source_monthly_cost * 100) if source_monthly_cost > 0 else 0.0

    return ConvertedWorkload(
        source_provider=source_provider,
        source_model=source_model,
        target_provider=target_provider,
        target_model=target_model,
        source_total_calls=total_calls,
        effective_calls=effective_monthly,
        est_input_tokens_per_call=round(est_input, 1),
        est_output_tokens_per_call=round(est_visible_output, 1),
        est_reasoning_tokens_per_call=round(est_reasoning, 1),
        est_total_tokens_per_call=round(est_input + est_visible_output + est_reasoning, 1),
        est_cost_per_call=round(est_cost_per_call, 8),
        est_monthly_cost=round(est_monthly, 2),
        source_monthly_cost=round(source_monthly_cost, 2),
        savings_usd=round(savings, 2),
        savings_pct=round(savings_pct, 1),
        conversion_factors=factors,
        conversion_confidence=factors.overall_confidence,
    )


# ---------------------------------------------------------------------------
# Batch conversion across all registry models
# ---------------------------------------------------------------------------


def convert_workload_across_models(
    source_provider: str,
    source_model: str,
    *,
    avg_input_tokens: float,
    avg_output_tokens: float,
    avg_reasoning_tokens: float = 0.0,
    total_calls: float,
    monthly_calls: float,
    source_monthly_cost: float,
    cached_ratio: float = 0.0,
    observed_reasoning_ratio: float | None = None,
    registry: PricingRegistry | None = None,
    target_providers: tuple[str, ...] | None = None,
) -> list[ConvertedWorkload]:
    """Convert the source workload across all models in the registry.

    Args:
        source_provider: Provider of the observed workload.
        source_model: Model of the observed workload.
        avg_input_tokens: Average input tokens per call.
        avg_output_tokens: Average output tokens per call.
        avg_reasoning_tokens: Average reasoning tokens per call.
        total_calls: Total calls in observation window.
        monthly_calls: Projected monthly calls.
        source_monthly_cost: Observed monthly cost.
        cached_ratio: Fraction of input from cache.
        observed_reasoning_ratio: Dynamically observed reasoning ratio.
        registry: Pricing registry.
        target_providers: Filter to specific providers.

    Returns:
        List of ``ConvertedWorkload`` sorted by estimated monthly cost.
    """
    reg = registry or get_pricing_registry()
    entries = reg.all_entries()

    if target_providers:
        allowed = {p.lower() for p in target_providers}
        entries = [e for e in entries if e.provider in allowed]

    results: list[ConvertedWorkload] = []
    for entry in entries:
        result = estimate_converted_workload(
            source_provider, source_model,
            entry.provider, entry.model_id,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            avg_reasoning_tokens=avg_reasoning_tokens,
            total_calls=total_calls,
            monthly_calls=monthly_calls,
            source_monthly_cost=source_monthly_cost,
            cached_ratio=cached_ratio,
            observed_reasoning_ratio=observed_reasoning_ratio,
            registry=reg,
        )
        results.append(result)

    results.sort(key=lambda r: r.est_monthly_cost)
    return results


# ---------------------------------------------------------------------------
# Comparison report: naive vs. converted
# ---------------------------------------------------------------------------


@dataclass
class ConversionComparison:
    """Side-by-side comparison of naive transfer vs. converted estimate.

    Shows how much the conversion factors change the cost projection.
    """

    target_provider: str
    target_model: str
    naive_monthly_cost: float
    converted_monthly_cost: float
    difference_usd: float
    difference_pct: float
    key_adjustments: list[str] = field(default_factory=list)


def build_naive_vs_converted_comparison(
    converted: ConvertedWorkload,
    *,
    naive_monthly_cost: float,
) -> ConversionComparison:
    """Build a comparison showing the impact of conversion factors.

    Args:
        converted: The conversion-adjusted estimate.
        naive_monthly_cost: The naive estimate (source tokens × target pricing).

    Returns:
        ``ConversionComparison`` with the difference.
    """
    diff = converted.est_monthly_cost - naive_monthly_cost
    diff_pct = (diff / naive_monthly_cost * 100) if naive_monthly_cost > 0 else 0.0

    adjustments: list[str] = []
    f = converted.conversion_factors
    if abs(f.input_token_ratio - 1.0) > 0.01:
        direction = "more" if f.input_token_ratio > 1.0 else "fewer"
        adjustments.append(
            f"Tokenizer: target uses {abs(f.input_token_ratio - 1.0):.0%} {direction} "
            f"input tokens for the same text"
        )
    if abs(f.output_token_ratio - 1.0) > 0.05:
        direction = "longer" if f.output_token_ratio > 1.0 else "shorter"
        adjustments.append(
            f"Output length: target produces {abs(f.output_token_ratio - 1.0):.0%} "
            f"{direction} responses"
        )
    if f.reasoning_overhead > 0:
        adjustments.append(
            f"Reasoning: target adds {f.reasoning_overhead:.0%} reasoning token overhead"
        )
    if f.quality_adjustment > 1.01:
        adjustments.append(
            f"Quality gap: ~{f.quality_adjustment:.0%} effective calls needed "
            f"for equivalent outcome"
        )

    return ConversionComparison(
        target_provider=converted.target_provider,
        target_model=converted.target_model,
        naive_monthly_cost=round(naive_monthly_cost, 2),
        converted_monthly_cost=round(converted.est_monthly_cost, 2),
        difference_usd=round(diff, 2),
        difference_pct=round(diff_pct, 1),
        key_adjustments=adjustments,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geometric_mean(values: list[float]) -> float:
    """Geometric mean of confidence values — penalizes low outliers."""
    if not values:
        return 0.5
    product = 1.0
    for v in values:
        product *= max(0.01, v)
    return product ** (1.0 / len(values))
