"""Version-controlled pricing registry for multi-provider LLM cost analysis.

Provides a single source of truth for model pricing across OpenAI, Anthropic,
Groq, and future providers. Supports fuzzy model-family matching, pricing
freshness tracking, and configurable overrides.

AC-3.2 compliance:
- Configurable pricing registry (this module)
- Version-controlled pricing file (PRICING_VERSION + per-entry updated_at)
- Approved pricing API placeholder (_fetch_remote_pricing)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

PRICING_VERSION = "2026-02-24"
PRICING_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class TokenPricing:
    """Per-million-token pricing for a single model.

    Attributes:
        input: USD per 1M input tokens.
        output: USD per 1M output tokens.
        cached_input: USD per 1M cached input tokens (if discounted).
        reasoning_output: USD per 1M reasoning/thinking output tokens
            (o3/o4 family). Falls back to ``output`` when None.
        input_audio: USD per 1M input audio tokens.
        output_audio: USD per 1M output audio tokens.
    """

    input: float
    output: float
    cached_input: float | None = None
    reasoning_output: float | None = None
    input_audio: float | None = None
    output_audio: float | None = None


@dataclass(frozen=True)
class ModelPricingEntry:
    """Full pricing metadata for a single model.

    Attributes:
        provider: Lowercase provider key (``openai``, ``anthropic``, ``groq``).
        model_id: Canonical model identifier as reported by the provider API.
        family: Model family for fuzzy matching (e.g. ``gpt-4o``, ``o3``).
        pricing: Token pricing rates.
        context_window: Maximum context length in tokens.
        supports_reasoning: Whether the model emits reasoning/thinking tokens.
        supports_vision: Whether the model accepts image inputs.
        supports_audio: Whether the model accepts audio inputs.
        deprecation_date: ISO date string if model is deprecated/scheduled.
        updated_at: ISO timestamp of last pricing update.
        notes: Free-form notes about pricing assumptions or caveats.
    """

    provider: str
    model_id: str
    family: str
    pricing: TokenPricing
    context_window: int = 128_000
    supports_reasoning: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    deprecation_date: str | None = None
    updated_at: str = PRICING_VERSION
    notes: str = ""


# ---------------------------------------------------------------------------
# Canonical pricing data — maintained manually from official pricing pages.
# Update ``PRICING_VERSION`` at the top whenever entries change.
# ---------------------------------------------------------------------------

_OPENAI_MODELS: list[ModelPricingEntry] = [
    # ---- GPT-4.1 family ----
    ModelPricingEntry(
        provider="openai",
        model_id="gpt-4.1",
        family="gpt-4.1",
        pricing=TokenPricing(input=2.00, output=8.00, cached_input=0.50),
        context_window=1_000_000,
        supports_vision=True,
        notes="1M context, code-heavy workloads",
    ),
    ModelPricingEntry(
        provider="openai",
        model_id="gpt-4.1-mini",
        family="gpt-4.1",
        pricing=TokenPricing(input=0.40, output=1.60, cached_input=0.10),
        context_window=1_000_000,
        supports_vision=True,
        notes="Balanced cost/performance in 4.1 family",
    ),
    ModelPricingEntry(
        provider="openai",
        model_id="gpt-4.1-nano",
        family="gpt-4.1",
        pricing=TokenPricing(input=0.10, output=0.40, cached_input=0.025),
        context_window=1_000_000,
        supports_vision=True,
        notes="Cheapest 4.1 variant for high-volume routing",
    ),
    # ---- GPT-4o family ----
    ModelPricingEntry(
        provider="openai",
        model_id="gpt-4o",
        family="gpt-4o",
        pricing=TokenPricing(
            input=2.50,
            output=10.00,
            cached_input=1.25,
            input_audio=40.00,
            output_audio=80.00,
        ),
        context_window=128_000,
        supports_vision=True,
        supports_audio=True,
        notes="Flagship multimodal model",
    ),
    ModelPricingEntry(
        provider="openai",
        model_id="gpt-4o-mini",
        family="gpt-4o",
        pricing=TokenPricing(
            input=0.15,
            output=0.60,
            cached_input=0.075,
            input_audio=10.00,
            output_audio=20.00,
        ),
        context_window=128_000,
        supports_vision=True,
        supports_audio=True,
        notes="Cost-effective 4o variant",
    ),
    # ---- o-series reasoning models ----
    ModelPricingEntry(
        provider="openai",
        model_id="o3",
        family="o3",
        pricing=TokenPricing(input=10.00, output=40.00, cached_input=2.50, reasoning_output=40.00),
        context_window=200_000,
        supports_reasoning=True,
        supports_vision=True,
        notes="Deep reasoning model — output tokens include internal reasoning tokens",
    ),
    ModelPricingEntry(
        provider="openai",
        model_id="o3-pro",
        family="o3",
        pricing=TokenPricing(input=20.00, output=80.00, cached_input=5.00, reasoning_output=80.00),
        context_window=200_000,
        supports_reasoning=True,
        supports_vision=True,
        notes="Extended-thinking variant of o3",
    ),
    ModelPricingEntry(
        provider="openai",
        model_id="o4-mini",
        family="o4",
        pricing=TokenPricing(input=1.10, output=4.40, cached_input=0.275, reasoning_output=4.40),
        context_window=200_000,
        supports_reasoning=True,
        supports_vision=True,
        notes="Cost-efficient reasoning model",
    ),
    ModelPricingEntry(
        provider="openai",
        model_id="o3-mini",
        family="o3",
        pricing=TokenPricing(input=1.10, output=4.40, cached_input=0.55, reasoning_output=4.40),
        context_window=200_000,
        supports_reasoning=True,
        notes="Lightweight reasoning (being replaced by o4-mini)",
        deprecation_date="2025-07-31",
    ),
    # ---- GPT-5 frontier ----
    ModelPricingEntry(
        provider="openai",
        model_id="gpt-5",
        family="gpt-5",
        pricing=TokenPricing(input=15.00, output=60.00, cached_input=3.75),
        context_window=256_000,
        supports_vision=True,
        supports_reasoning=True,
        notes="Frontier model — premium quality and agentic capabilities",
    ),
    ModelPricingEntry(
        provider="openai",
        model_id="gpt-5.2",
        family="gpt-5",
        pricing=TokenPricing(input=15.00, output=60.00, cached_input=3.75),
        context_window=256_000,
        supports_vision=True,
        supports_reasoning=True,
        notes="GPT-5.2 frontier variant — assumed same pricing as gpt-5 until confirmed",
    ),
]

_ANTHROPIC_MODELS: list[ModelPricingEntry] = [
    ModelPricingEntry(
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        family="claude-sonnet",
        pricing=TokenPricing(input=3.00, output=15.00, cached_input=0.30),
        context_window=200_000,
        supports_vision=True,
        notes="Claude Sonnet 4 — strong reasoning + coding",
    ),
    ModelPricingEntry(
        provider="anthropic",
        model_id="claude-3.5-sonnet",
        family="claude-sonnet",
        pricing=TokenPricing(input=3.00, output=15.00, cached_input=0.30),
        context_window=200_000,
        supports_vision=True,
        notes="Claude 3.5 Sonnet",
    ),
    ModelPricingEntry(
        provider="anthropic",
        model_id="claude-3.7-sonnet",
        family="claude-sonnet",
        pricing=TokenPricing(input=3.00, output=15.00, cached_input=0.30),
        context_window=200_000,
        supports_vision=True,
        supports_reasoning=True,
        notes="Claude 3.7 Sonnet — extended thinking support",
    ),
    ModelPricingEntry(
        provider="anthropic",
        model_id="claude-opus-4-20250514",
        family="claude-opus",
        pricing=TokenPricing(input=15.00, output=75.00, cached_input=1.50),
        context_window=200_000,
        supports_vision=True,
        supports_reasoning=True,
        notes="Claude Opus 4 — frontier quality",
    ),
    ModelPricingEntry(
        provider="anthropic",
        model_id="claude-3-opus",
        family="claude-opus",
        pricing=TokenPricing(input=15.00, output=75.00, cached_input=1.50),
        context_window=200_000,
        supports_vision=True,
        notes="Claude 3 Opus",
    ),
    ModelPricingEntry(
        provider="anthropic",
        model_id="claude-3.5-haiku",
        family="claude-haiku",
        pricing=TokenPricing(input=0.80, output=4.00, cached_input=0.08),
        context_window=200_000,
        supports_vision=True,
        notes="Claude 3.5 Haiku — fast + affordable",
    ),
    ModelPricingEntry(
        provider="anthropic",
        model_id="claude-3-haiku",
        family="claude-haiku",
        pricing=TokenPricing(input=0.25, output=1.25, cached_input=0.03),
        context_window=200_000,
        supports_vision=True,
        notes="Claude 3 Haiku — legacy fast model",
    ),
]

_GROQ_MODELS: list[ModelPricingEntry] = [
    ModelPricingEntry(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        family="llama-3.1",
        pricing=TokenPricing(input=0.05, output=0.08),
        context_window=131_072,
        notes="Groq-hosted Llama 3.1 8B",
    ),
    ModelPricingEntry(
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        family="llama-3.3",
        pricing=TokenPricing(input=0.59, output=0.79),
        context_window=131_072,
        notes="Groq-hosted Llama 3.3 70B",
    ),
    ModelPricingEntry(
        provider="groq",
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        family="llama-4",
        pricing=TokenPricing(input=0.11, output=0.34),
        context_window=131_072,
        notes="Llama 4 Scout on Groq",
    ),
    ModelPricingEntry(
        provider="groq",
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct",
        family="llama-4",
        pricing=TokenPricing(input=0.20, output=0.60),
        context_window=131_072,
        notes="Llama 4 Maverick on Groq",
    ),
    ModelPricingEntry(
        provider="groq",
        model_id="qwen/qwen3-32b",
        family="qwen3",
        pricing=TokenPricing(input=0.29, output=0.59),
        context_window=131_072,
        notes="Qwen 3 32B on Groq",
    ),
]

_ALL_ENTRIES: list[ModelPricingEntry] = _OPENAI_MODELS + _ANTHROPIC_MODELS + _GROQ_MODELS


@dataclass
class PricingRegistry:
    """Central pricing registry with lookup, freshness, and override support.

    Usage::

        registry = PricingRegistry()
        entry = registry.lookup("openai", "gpt-4o-2024-08-06")
        if entry:
            cost = entry.pricing.input * input_tokens / 1_000_000
    """

    _entries: dict[tuple[str, str], ModelPricingEntry] = field(default_factory=dict)
    _family_index: dict[tuple[str, str], list[ModelPricingEntry]] = field(default_factory=dict)
    _overrides: dict[tuple[str, str], ModelPricingEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._entries:
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load all built-in pricing entries."""
        for entry in _ALL_ENTRIES:
            key = (entry.provider.lower(), entry.model_id.lower())
            self._entries[key] = entry

            family_key = (entry.provider.lower(), entry.family.lower())
            self._family_index.setdefault(family_key, []).append(entry)

    def lookup(self, provider: str, model_id: str) -> ModelPricingEntry | None:
        """Look up pricing by exact model id, then by family prefix matching.

        Args:
            provider: Provider key (e.g. ``openai``).
            model_id: Model identifier as reported by provider API.

        Returns:
            Best-matching ``ModelPricingEntry`` or ``None`` if no match found.
        """
        provider_l = provider.strip().lower()
        model_l = model_id.strip().lower()

        if (provider_l, model_l) in self._overrides:
            return self._overrides[(provider_l, model_l)]

        if (provider_l, model_l) in self._entries:
            return self._entries[(provider_l, model_l)]

        return self._fuzzy_match(provider_l, model_l)

    def _fuzzy_match(self, provider: str, model_id: str) -> ModelPricingEntry | None:
        """Match by family prefix: e.g. 'gpt-4o-2024-08-06' → 'gpt-4o' family."""
        best_match: ModelPricingEntry | None = None
        best_prefix_len = 0

        for (prov, family), entries in self._family_index.items():
            if prov != provider:
                continue
            if model_id.startswith(family) and len(family) > best_prefix_len:
                best_prefix_len = len(family)
                non_deprecated = [e for e in entries if e.deprecation_date is None]
                best_match = non_deprecated[0] if non_deprecated else entries[0]

        if best_match:
            logger.debug(
                "Fuzzy pricing match: %s/%s → %s/%s (family: %s)",
                provider,
                model_id,
                best_match.provider,
                best_match.model_id,
                best_match.family,
            )
        return best_match

    def register_override(self, entry: ModelPricingEntry) -> None:
        """Register a pricing override for a specific provider/model pair."""
        key = (entry.provider.lower(), entry.model_id.lower())
        self._overrides[key] = entry

    def all_entries(self, *, provider: str | None = None) -> list[ModelPricingEntry]:
        """Return all pricing entries, optionally filtered by provider."""
        entries = list(self._entries.values())
        if provider:
            provider_l = provider.strip().lower()
            entries = [e for e in entries if e.provider == provider_l]
        return sorted(entries, key=lambda e: (e.provider, e.family, e.model_id))

    def get_pricing_per_million(
        self,
        provider: str,
        model_id: str,
    ) -> dict[str, float]:
        """Backward-compatible dict format: ``{"input": X, "output": Y}``.

        Returns empty dict if model is not found.
        """
        entry = self.lookup(provider, model_id)
        if not entry:
            return {}
        result: dict[str, float] = {
            "input": entry.pricing.input,
            "output": entry.pricing.output,
        }
        if entry.pricing.cached_input is not None:
            result["cached_input"] = entry.pricing.cached_input
        if entry.pricing.reasoning_output is not None:
            result["reasoning_output"] = entry.pricing.reasoning_output
        return result

    def pricing_freshness_days(self) -> int:
        """Days since the pricing data was last updated."""
        try:
            updated = datetime.strptime(PRICING_VERSION, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            delta = datetime.now(tz=timezone.utc) - updated
            return max(0, delta.days)
        except ValueError:
            return -1

    def is_pricing_stale(self, *, threshold_days: int = 30) -> bool:
        """Return True if pricing data is older than ``threshold_days``."""
        age = self.pricing_freshness_days()
        return age < 0 or age > threshold_days

    def export_as_legacy_map(self, provider: str) -> dict[str, dict[str, float]]:
        """Export as the legacy ``MODEL_PRICING_PER_MILLION`` dict format.

        Keyed by lowercase model_id → ``{"input": ..., "output": ...}``.
        """
        result: dict[str, dict[str, float]] = {}
        for entry in self.all_entries(provider=provider):
            result[entry.model_id.lower()] = {
                "input": entry.pricing.input,
                "output": entry.pricing.output,
            }
        return result

    def merge_entries(self, entries: list[ModelPricingEntry]) -> int:
        """Bulk-add entries into the registry (not as overrides).

        Existing entries with the same (provider, model_id) are replaced.

        Args:
            entries: List of pricing entries to merge.

        Returns:
            Number of entries added/updated.
        """
        count = 0
        for entry in entries:
            key = (entry.provider.lower(), entry.model_id.lower())
            self._entries[key] = entry
            family_key = (entry.provider.lower(), entry.family.lower())
            fam_list = self._family_index.setdefault(family_key, [])
            fam_list[:] = [e for e in fam_list if e.model_id.lower() != entry.model_id.lower()]
            fam_list.append(entry)
            count += 1
        return count

    def summary(self) -> dict[str, Any]:
        """Return registry metadata for display/logging."""
        providers = set()
        models_count = 0
        for entry in self._entries.values():
            providers.add(entry.provider)
            models_count += 1

        source = getattr(self, "_data_source", "hardcoded")
        return {
            "pricing_version": PRICING_VERSION,
            "schema_version": PRICING_SCHEMA_VERSION,
            "total_models": models_count + len(self._overrides),
            "providers": sorted(providers),
            "freshness_days": self.pricing_freshness_days(),
            "is_stale": self.is_pricing_stale(),
            "overrides_count": len(self._overrides),
            "data_source": source,
        }


# Module-level singleton for convenience
_default_registry: PricingRegistry | None = None
_default_source: str = "hardcoded"


def get_pricing_registry() -> PricingRegistry:
    """Return the module-level singleton ``PricingRegistry``."""
    global _default_registry  # noqa: PLW0603
    if _default_registry is None:
        _default_registry = PricingRegistry()
    return _default_registry


def set_pricing_registry(registry: PricingRegistry, *, source: str = "dynamic") -> None:
    """Replace the module-level singleton with a new registry.

    Called by the dynamic pricing loader after building a registry from
    remote/YAML sources.
    """
    global _default_registry, _default_source  # noqa: PLW0603
    registry._data_source = source  # type: ignore[attr-defined]
    _default_registry = registry
    _default_source = source
