"""Dynamic pricing fetcher — eliminates hardcoded pricing staleness.

Fetches current model pricing from the **LiteLLM community pricing index**
(https://github.com/BerriAI/litellm) which tracks 2,500+ models across all
major providers and is updated within hours of pricing changes.

Fallback chain (highest priority first):
    1. **Remote** — LiteLLM ``model_prices_and_context_window.json`` (live)
    2. **Local YAML** — ``data/pricing.yaml`` (version-controlled, editable)
    3. **Hardcoded** — Python defaults in ``pricing_registry.py`` (emergency)

Each source is self-contained: if the remote fetch fails (network down,
GitHub outage), the system transparently falls back to the next layer
without user intervention.

Cache behaviour:
    Remote data is cached at ``data/.cache/litellm_pricing.json`` with a
    configurable TTL (default 24 hours). Within the TTL the cached file is
    used instead of re-fetching.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from src.pricing_registry import (
    ModelPricingEntry,
    PricingRegistry,
    TokenPricing,
)

logger = logging.getLogger(__name__)

_LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / ".cache"
_CACHE_FILE = _CACHE_DIR / "litellm_pricing.json"
_YAML_FILE = _PROJECT_ROOT / "data" / "pricing.yaml"
_METADATA_YAML = _PROJECT_ROOT / "data" / "model_metadata.yaml"

_DEFAULT_TTL_HOURS = 24
_FETCH_TIMEOUT_SECONDS = 20

_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic", "groq"})

_PER_TOKEN_TO_PER_MILLION = 1_000_000


@dataclass
class PricingSource:
    """Metadata about where the active pricing data came from."""

    source: str  # "litellm_remote", "litellm_cache", "yaml", "hardcoded"
    model_count: int = 0
    fetch_timestamp: float = 0.0
    url: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Remote fetch (LiteLLM)
# ---------------------------------------------------------------------------


def _fetch_litellm_raw(*, timeout: int = _FETCH_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Fetch the LiteLLM pricing JSON from GitHub.

    Returns:
        Raw parsed JSON dict keyed by model name.

    Raises:
        httpx.HTTPError: On any network failure.
    """
    resp = httpx.get(_LITELLM_URL, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def _is_cache_fresh(*, ttl_hours: float = _DEFAULT_TTL_HOURS) -> bool:
    """Check if the local cache file exists and is within TTL."""
    if not _CACHE_FILE.exists():
        return False
    age_seconds = time.time() - _CACHE_FILE.stat().st_mtime
    return age_seconds < (ttl_hours * 3600)


def _read_cache() -> dict[str, Any]:
    return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))


def _write_cache(data: dict[str, Any]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(data), encoding="utf-8")


def fetch_litellm_pricing(
    *,
    ttl_hours: float = _DEFAULT_TTL_HOURS,
    force_refresh: bool = False,
) -> tuple[dict[str, Any], PricingSource]:
    """Fetch LiteLLM pricing with local caching.

    Args:
        ttl_hours: How many hours the cache is considered fresh.
        force_refresh: Bypass cache and always fetch from remote.

    Returns:
        Tuple of (raw_data, PricingSource metadata).
    """
    if not force_refresh and _is_cache_fresh(ttl_hours=ttl_hours):
        try:
            data = _read_cache()
            return data, PricingSource(
                source="litellm_cache",
                model_count=len(data),
                fetch_timestamp=_CACHE_FILE.stat().st_mtime,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache read failed: %s — fetching from remote.", exc)

    try:
        data = _fetch_litellm_raw()
        _write_cache(data)
        source = PricingSource(
            source="litellm_remote",
            model_count=len(data),
            fetch_timestamp=time.time(),
            url=_LITELLM_URL,
        )
        logger.info(
            "Fetched %d models from LiteLLM pricing index.", len(data),
        )
        return data, source
    except Exception as exc:
        logger.warning("Remote LiteLLM fetch failed: %s", exc)

        if _CACHE_FILE.exists():
            try:
                data = _read_cache()
                return data, PricingSource(
                    source="litellm_cache",
                    model_count=len(data),
                    fetch_timestamp=_CACHE_FILE.stat().st_mtime,
                    error=f"Remote failed ({exc}); using stale cache",
                )
            except (json.JSONDecodeError, OSError):
                pass

        return {}, PricingSource(
            source="litellm_remote",
            error=f"Remote fetch failed: {exc}",
        )


# ---------------------------------------------------------------------------
# LiteLLM → ModelPricingEntry mapping
# ---------------------------------------------------------------------------

_LITELLM_PROVIDER_MAP: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "groq": "groq",
}

_PROVIDER_KEY_PREFIXES: dict[str, str] = {
    "groq": "groq/",
}


def _strip_provider_prefix(key: str, provider: str) -> str:
    prefix = _PROVIDER_KEY_PREFIXES.get(provider, "")
    if prefix and key.startswith(prefix):
        return key[len(prefix):]
    return key


def _infer_family(model_id: str, provider: str) -> str:
    """Best-effort family inference from model ID."""
    m = model_id.lower()

    if provider == "openai":
        for fam in ("gpt-5", "gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5",
                     "o4-mini", "o4", "o3-pro", "o3-mini", "o3", "o1-pro", "o1-mini", "o1"):
            if m.startswith(fam):
                return fam
    elif provider == "anthropic":
        for fam in ("claude-opus-4", "claude-sonnet-4", "claude-3.7-sonnet",
                     "claude-3.5-sonnet", "claude-3.5-haiku", "claude-3-opus",
                     "claude-3-sonnet", "claude-3-haiku"):
            if fam in m:
                return fam
    elif provider == "groq":
        for fam in ("llama-4", "llama-3.3", "llama-3.1", "qwen3", "mixtral", "gemma"):
            if fam in m:
                return fam

    parts = m.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else m


def _litellm_entry_to_pricing_entry(
    key: str,
    data: dict[str, Any],
) -> ModelPricingEntry | None:
    """Map a single LiteLLM entry to our ``ModelPricingEntry``."""
    provider_raw = data.get("litellm_provider", "")
    provider = _LITELLM_PROVIDER_MAP.get(provider_raw, "")
    if not provider or provider not in _SUPPORTED_PROVIDERS:
        return None

    input_cpt = data.get("input_cost_per_token")
    output_cpt = data.get("output_cost_per_token")
    if input_cpt is None or output_cpt is None:
        return None

    model_id = _strip_provider_prefix(key, provider)

    # Skip fine-tuned, batch-only, and deep-research variants to keep catalog manageable
    if model_id.startswith("ft:") or "-batch-" in model_id:
        return None

    cached_rate = data.get("cache_read_input_token_cost")
    reasoning_rate = data.get("output_cost_per_token")  # same rate for reasoning models

    pricing = TokenPricing(
        input=round(float(input_cpt) * _PER_TOKEN_TO_PER_MILLION, 4),
        output=round(float(output_cpt) * _PER_TOKEN_TO_PER_MILLION, 4),
        cached_input=round(float(cached_rate) * _PER_TOKEN_TO_PER_MILLION, 4) if cached_rate else None,
        reasoning_output=(
            round(float(reasoning_rate) * _PER_TOKEN_TO_PER_MILLION, 4)
            if data.get("supports_reasoning")
            else None
        ),
        input_audio=(
            round(float(data["input_cost_per_audio_token"]) * _PER_TOKEN_TO_PER_MILLION, 4)
            if data.get("input_cost_per_audio_token")
            else None
        ),
        output_audio=(
            round(float(data["output_cost_per_audio_token"]) * _PER_TOKEN_TO_PER_MILLION, 4)
            if data.get("output_cost_per_audio_token")
            else None
        ),
    )

    context_window = int(data.get("max_input_tokens") or data.get("max_tokens") or 128_000)
    family = _infer_family(model_id, provider)

    return ModelPricingEntry(
        provider=provider,
        model_id=model_id,
        family=family,
        pricing=pricing,
        context_window=context_window,
        supports_reasoning=bool(data.get("supports_reasoning")),
        supports_vision=bool(data.get("supports_vision")),
        supports_audio=bool(
            data.get("supports_audio_input") or data.get("input_cost_per_audio_token")
        ),
        deprecation_date=data.get("deprecation_date"),
        updated_at="dynamic",
        notes="Sourced from LiteLLM pricing index",
    )


def parse_litellm_to_entries(
    raw: dict[str, Any],
    *,
    providers: frozenset[str] | None = None,
) -> list[ModelPricingEntry]:
    """Convert the full LiteLLM JSON into a list of ``ModelPricingEntry``.

    Args:
        raw: Raw LiteLLM pricing dict.
        providers: Restrict to these providers. Defaults to all supported.

    Returns:
        List of ``ModelPricingEntry`` objects.
    """
    allowed = providers or _SUPPORTED_PROVIDERS
    entries: list[ModelPricingEntry] = []
    seen: set[tuple[str, str]] = set()

    for key, data in raw.items():
        if not isinstance(data, dict):
            continue
        entry = _litellm_entry_to_pricing_entry(key, data)
        if entry is None:
            continue
        if entry.provider not in allowed:
            continue

        dedup_key = (entry.provider, entry.model_id)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# YAML pricing loader
# ---------------------------------------------------------------------------


def load_yaml_pricing(path: Path | str | None = None) -> list[ModelPricingEntry]:
    """Load pricing entries from a YAML file.

    Args:
        path: Path to YAML file. Defaults to ``data/pricing.yaml``.

    Returns:
        List of ``ModelPricingEntry``. Empty list if file missing/invalid.
    """
    yaml_path = Path(path) if path else _YAML_FILE
    if not yaml_path.exists():
        return []

    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse pricing YAML at %s: %s", yaml_path, exc)
        return []

    if not isinstance(raw, dict) or "providers" not in raw:
        return []

    entries: list[ModelPricingEntry] = []
    for provider_key, provider_data in raw["providers"].items():
        models = provider_data.get("models", {})
        for model_id, m in models.items():
            p = m.get("pricing", {})
            pricing = TokenPricing(
                input=float(p.get("input", 0)),
                output=float(p.get("output", 0)),
                cached_input=float(p["cached_input"]) if "cached_input" in p else None,
                reasoning_output=float(p["reasoning_output"]) if "reasoning_output" in p else None,
                input_audio=float(p["input_audio"]) if "input_audio" in p else None,
                output_audio=float(p["output_audio"]) if "output_audio" in p else None,
            )
            entries.append(ModelPricingEntry(
                provider=provider_key,
                model_id=model_id,
                family=m.get("family", _infer_family(model_id, provider_key)),
                pricing=pricing,
                context_window=int(m.get("context_window", 128_000)),
                supports_reasoning=bool(m.get("supports_reasoning")),
                supports_vision=bool(m.get("supports_vision")),
                supports_audio=bool(m.get("supports_audio")),
                deprecation_date=m.get("deprecation_date"),
                updated_at=str(raw.get("updated_at", "yaml")),
                notes=m.get("notes", "From pricing YAML"),
            ))

    return entries


# ---------------------------------------------------------------------------
# Model metadata loader (latency + quality)
# ---------------------------------------------------------------------------


def load_model_metadata(
    path: Path | str | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Load model metadata (latency_ms, quality_score) from YAML.

    Args:
        path: Path to metadata YAML. Defaults to ``data/model_metadata.yaml``.

    Returns:
        Dict keyed by ``(provider, model_id)`` → ``{"latency_ms": ..., "quality_score": ...}``.
    """
    yaml_path = Path(path) if path else _METADATA_YAML
    if not yaml_path.exists():
        return {}

    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse model metadata YAML at %s: %s", yaml_path, exc)
        return {}

    if not isinstance(raw, dict) or "providers" not in raw:
        return {}

    result: dict[tuple[str, str], dict[str, float]] = {}
    for provider_key, provider_data in raw["providers"].items():
        models = provider_data.get("models", {})
        for model_id, m in models.items():
            result[(provider_key, model_id)] = {
                "latency_ms": float(m.get("latency_ms", 500)),
                "quality_score": float(m.get("quality_score", 0.70)),
            }

    return result


# ---------------------------------------------------------------------------
# Orchestration — build a fully-loaded PricingRegistry
# ---------------------------------------------------------------------------


def build_dynamic_registry(
    *,
    ttl_hours: float = _DEFAULT_TTL_HOURS,
    force_refresh: bool = False,
    enable_remote: bool = True,
) -> tuple[PricingRegistry, PricingSource]:
    """Build a ``PricingRegistry`` using the multi-source fallback chain.

    Resolution order:
        1. LiteLLM remote (if ``enable_remote``)
        2. ``data/pricing.yaml``
        3. Hardcoded defaults

    Remote entries override YAML entries, which override hardcoded defaults.

    Args:
        ttl_hours: Cache TTL for remote fetch.
        force_refresh: Bypass cache.
        enable_remote: If False, skip remote fetch entirely.

    Returns:
        Tuple of (populated PricingRegistry, PricingSource metadata).
    """
    registry = PricingRegistry()

    source = PricingSource(source="hardcoded", model_count=len(registry.all_entries()))

    yaml_entries = load_yaml_pricing()
    if yaml_entries:
        for entry in yaml_entries:
            registry.register_override(entry)
        source = PricingSource(
            source="yaml",
            model_count=len(registry.all_entries()) + len(yaml_entries),
        )
        logger.info("Loaded %d entries from pricing YAML.", len(yaml_entries))

    if enable_remote:
        raw, remote_source = fetch_litellm_pricing(
            ttl_hours=ttl_hours, force_refresh=force_refresh,
        )
        if raw:
            entries = parse_litellm_to_entries(raw)
            for entry in entries:
                registry.register_override(entry)
            source = PricingSource(
                source=remote_source.source,
                model_count=len(entries),
                fetch_timestamp=remote_source.fetch_timestamp,
                url=remote_source.url,
                error=remote_source.error,
            )
            logger.info(
                "Loaded %d entries from LiteLLM (%s). Overriding hardcoded/YAML pricing.",
                len(entries),
                remote_source.source,
            )
        elif remote_source.error:
            source.error = remote_source.error

    return registry, source
