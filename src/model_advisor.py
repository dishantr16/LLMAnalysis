"""AI Model Advisor heuristics for workload-aware model recommendations."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

CATEGORY_LABELS = {
    "slm": "SLM",
    "llm": "LLM",
    "frontier": "Frontier",
}

OBJECTIVE_OPTIONS = {"balanced", "min_cost", "max_quality"}

TASK_CAPABILITY_HINTS: dict[str, set[str]] = {
    "general_assistant": {"reasoning"},
    "classification_routing": set(),
    "coding_assistant": {"code", "reasoning"},
    "document_analysis": {"long_context", "reasoning"},
    "multimodal": {"vision", "reasoning"},
}

# Pricing and performance values are intentionally heuristic so the advisor can
# rank candidate models with a consistent scoring approach.
ADVISOR_MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "category": "slm",
        "input_cost_per_1m": 0.10,
        "output_cost_per_1m": 0.40,
        "latency_ms": 120,
        "quality_score": 0.66,
        "max_context": 128000,
        "capabilities": {"reasoning", "classification", "summarization"},
        "best_for": "High-volume low-complexity tasks and routing",
    },
    {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "category": "slm",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "latency_ms": 180,
        "quality_score": 0.73,
        "max_context": 128000,
        "capabilities": {"reasoning", "code", "vision", "classification"},
        "best_for": "General low-cost production assistants",
    },
    {
        "provider": "anthropic",
        "model": "claude-haiku",
        "category": "slm",
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
        "latency_ms": 220,
        "quality_score": 0.75,
        "max_context": 200000,
        "capabilities": {"reasoning", "code", "long_context", "classification"},
        "best_for": "Fast responses with larger context windows",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "category": "llm",
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
        "latency_ms": 360,
        "quality_score": 0.88,
        "max_context": 128000,
        "capabilities": {"reasoning", "code", "vision", "long_context"},
        "best_for": "Balanced reasoning quality for broad production workloads",
    },
    {
        "provider": "openai",
        "model": "gpt-4.1",
        "category": "llm",
        "input_cost_per_1m": 2.00,
        "output_cost_per_1m": 8.00,
        "latency_ms": 340,
        "quality_score": 0.87,
        "max_context": 1000000,
        "capabilities": {"reasoning", "code", "long_context"},
        "best_for": "Long-context and coding-heavy enterprise use cases",
    },
    {
        "provider": "anthropic",
        "model": "claude-sonnet",
        "category": "llm",
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "latency_ms": 440,
        "quality_score": 0.90,
        "max_context": 200000,
        "capabilities": {"reasoning", "code", "long_context"},
        "best_for": "Complex writing, coding, and high-quality reasoning",
    },
    {
        "provider": "openai",
        "model": "gpt-5",
        "category": "frontier",
        "input_cost_per_1m": 15.00,
        "output_cost_per_1m": 60.00,
        "latency_ms": 850,
        "quality_score": 0.96,
        "max_context": 256000,
        "capabilities": {"reasoning", "code", "vision", "long_context", "research"},
        "best_for": "High-end agentic reasoning and premium quality output",
    },
    {
        "provider": "openai",
        "model": "o3",
        "category": "frontier",
        "input_cost_per_1m": 10.00,
        "output_cost_per_1m": 40.00,
        "latency_ms": 1500,
        "quality_score": 0.97,
        "max_context": 128000,
        "capabilities": {"reasoning", "code", "research"},
        "best_for": "Deep reasoning and difficult technical problem solving",
    },
    {
        "provider": "anthropic",
        "model": "claude-opus",
        "category": "frontier",
        "input_cost_per_1m": 18.00,
        "output_cost_per_1m": 90.00,
        "latency_ms": 1200,
        "quality_score": 0.95,
        "max_context": 200000,
        "capabilities": {"reasoning", "code", "long_context", "research"},
        "best_for": "Safety-sensitive and advanced reasoning workflows",
    },
]


def advisor_model_catalog(*, providers: tuple[str, ...] = ("openai", "anthropic")) -> pd.DataFrame:
    allowed = {p.lower() for p in providers}
    rows = [row for row in ADVISOR_MODEL_CATALOG if str(row["provider"]).lower() in allowed]
    if not rows:
        return pd.DataFrame(
            columns=[
                "provider",
                "model",
                "category",
                "input_cost_per_1m",
                "output_cost_per_1m",
                "latency_ms",
                "quality_score",
                "max_context",
                "capabilities",
                "best_for",
            ]
        )
    catalog_df = pd.DataFrame(rows)
    catalog_df["provider"] = catalog_df["provider"].astype(str).str.lower()
    catalog_df["model"] = catalog_df["model"].astype(str)
    catalog_df["category"] = catalog_df["category"].astype(str).str.lower()
    return catalog_df


def build_usage_workload_profile(unified_df: pd.DataFrame) -> dict[str, Any]:
    if unified_df.empty:
        return {
            "total_calls": 0.0,
            "total_input_tokens": 0.0,
            "total_output_tokens": 0.0,
            "total_cost_usd": 0.0,
            "window_days": 0.0,
            "avg_calls_per_day": 0.0,
            "avg_input_tokens_per_call": 0.0,
            "avg_output_tokens_per_call": 0.0,
            "avg_total_tokens_per_call": 0.0,
            "estimated_monthly_calls": 0.0,
            "estimated_monthly_spend_usd": 0.0,
            "current_provider": "unknown",
            "current_model": "unknown",
            "current_cpi_usd": 0.0,
            "active_providers": [],
            "active_models": [],
        }

    df = _prepare_unified_df(unified_df)
    total_calls = float(df["calls"].sum())
    total_input = float(df["input_tokens"].sum())
    total_output = float(df["output_tokens"].sum())
    total_tokens = float(df["total_tokens"].sum())
    total_cost = float(df["cost_usd"].sum())

    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    if pd.isna(start_ts) or pd.isna(end_ts):
        window_days = 1.0
    else:
        window_days = float(max(1, (end_ts.normalize() - start_ts.normalize()).days + 1))

    avg_calls_per_day = total_calls / window_days if window_days > 0 else 0.0
    estimated_monthly_calls = avg_calls_per_day * 30.0
    estimated_monthly_spend = (total_cost / window_days) * 30.0 if window_days > 0 else 0.0

    current = (
        df.groupby(["provider", "model"], as_index=False)
        .agg(total_cost=("cost_usd", "sum"), total_calls=("calls", "sum"))
        .sort_values(["total_cost", "total_calls"], ascending=False)
    )
    current_provider = "unknown"
    current_model = "unknown"
    current_cpi = 0.0
    if not current.empty:
        row0 = current.iloc[0]
        current_provider = str(row0["provider"])
        current_model = str(row0["model"])
        current_cost = float(row0["total_cost"])
        current_calls = float(row0["total_calls"])
        current_cpi = current_cost / current_calls if current_calls > 0 else 0.0

    return {
        "total_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": total_cost,
        "window_days": window_days,
        "avg_calls_per_day": avg_calls_per_day,
        "avg_input_tokens_per_call": total_input / total_calls if total_calls > 0 else 0.0,
        "avg_output_tokens_per_call": total_output / total_calls if total_calls > 0 else 0.0,
        "avg_total_tokens_per_call": total_tokens / total_calls if total_calls > 0 else 0.0,
        "estimated_monthly_calls": estimated_monthly_calls,
        "estimated_monthly_spend_usd": estimated_monthly_spend,
        "current_provider": current_provider,
        "current_model": current_model,
        "current_cpi_usd": current_cpi,
        "active_providers": sorted(df["provider"].dropna().astype(str).unique().tolist()),
        "active_models": sorted(df["model"].dropna().astype(str).unique().tolist()),
    }


def run_ai_model_advisor(
    unified_df: pd.DataFrame,
    *,
    objective: str = "balanced",
    complexity_level: str = "medium",
    max_latency_ms: int = 800,
    monthly_budget_usd: float | None = None,
    required_capabilities: tuple[str, ...] = (),
    primary_task: str = "general_assistant",
    preferred_providers: tuple[str, ...] = ("openai", "anthropic"),
    allow_cross_provider: bool = True,
    top_n: int = 10,
) -> dict[str, Any]:
    if unified_df.empty:
        return {"error": "No usage rows available to run AI Model Advisor."}

    objective_key = objective.strip().lower()
    if objective_key not in OBJECTIVE_OPTIONS:
        objective_key = "balanced"

    complexity_key = complexity_level.strip().lower()
    if complexity_key not in {"low", "medium", "high", "very_high"}:
        complexity_key = "medium"

    profile = build_usage_workload_profile(unified_df)
    if profile["total_calls"] <= 0:
        return {"error": "No request volume detected in current data window.", "profile": profile}

    observed_stats_df = _build_observed_model_stats(unified_df)
    capabilities = set(required_capabilities)
    capabilities.update(TASK_CAPABILITY_HINTS.get(primary_task, set()))

    category_scores = _compute_category_scores(
        profile=profile,
        complexity_level=complexity_key,
        max_latency_ms=max_latency_ms,
        monthly_budget_usd=monthly_budget_usd,
        required_capabilities=capabilities,
        objective=objective_key,
        providers=preferred_providers,
    )

    candidates_df = _build_candidate_scores(
        profile=profile,
        observed_stats_df=observed_stats_df,
        category_scores=category_scores,
        objective=objective_key,
        max_latency_ms=max_latency_ms,
        monthly_budget_usd=monthly_budget_usd,
        required_capabilities=capabilities,
        preferred_providers=preferred_providers,
        allow_cross_provider=allow_cross_provider,
        top_n=top_n,
    )

    if candidates_df.empty:
        return {
            "error": "No candidate models available for selected provider filters.",
            "profile": profile,
        }

    primary = candidates_df.iloc[0]
    second_score = float(candidates_df.iloc[1]["advisor_score"]) if len(candidates_df) > 1 else 0.0
    confidence = _estimate_confidence(
        top_score=float(primary["advisor_score"]),
        second_score=second_score,
        total_calls=float(profile["total_calls"]),
        cost_source=str(primary["cost_source"]),
    )

    baseline_cost = _estimate_baseline_monthly_cost(
        candidates_df=candidates_df,
        profile=profile,
        current_provider=str(profile["current_provider"]),
        current_model=str(profile["current_model"]),
    )
    rec_cost = float(primary["estimated_monthly_cost_usd"])
    savings_usd = baseline_cost - rec_cost
    savings_pct = (savings_usd / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

    primary_reasoning = _build_primary_reasoning(
        primary=primary,
        objective=objective_key,
        category_scores=category_scores,
        required_capabilities=capabilities,
        savings_usd=savings_usd,
    )
    primary_tradeoffs = _build_tradeoffs(
        primary=primary,
        max_latency_ms=max_latency_ms,
        savings_usd=savings_usd,
        allow_cross_provider=allow_cross_provider,
        current_provider=str(profile["current_provider"]),
    )

    category_scores_pct = {
        CATEGORY_LABELS[key]: round(max(0.0, min(1.0, value)) * 100.0, 1)
        for key, value in category_scores.items()
    }
    primary_payload = {
        "recommended_category": CATEGORY_LABELS.get(str(primary["category"]), str(primary["category"]).upper()),
        "recommended_model": str(primary["model"]),
        "provider": _provider_label(str(primary["provider"])),
        "confidence_score": round(confidence, 1),
        "estimated_monthly_cost_usd": round(rec_cost, 2),
        "estimated_monthly_savings_usd": round(savings_usd, 2),
        "estimated_monthly_savings_pct": round(savings_pct, 1),
        "cost_source": str(primary["cost_source"]),
        "reasoning": primary_reasoning,
        "trade_offs": primary_tradeoffs,
    }

    alternatives_payload: list[dict[str, Any]] = []
    for _, row in candidates_df.iloc[1:6].iterrows():
        alternatives_payload.append(
            {
                "Category": CATEGORY_LABELS.get(str(row["category"]), str(row["category"]).upper()),
                "Model": str(row["model"]),
                "Provider": _provider_label(str(row["provider"])),
                "Advisor Score (%)": round(float(row["advisor_score"]) * 100.0, 1),
                "Estimated Monthly Cost (USD)": round(float(row["estimated_monthly_cost_usd"]), 2),
                "Estimated CPI (USD)": round(float(row["estimated_cpi_usd"]), 6),
                "Cost Source": str(row["cost_source"]),
                "Notes": str(row["best_for"]),
            }
        )

    candidate_view = candidates_df.copy()
    candidate_view["provider"] = candidate_view["provider"].map(lambda value: _provider_label(str(value)))
    candidate_view["category"] = candidate_view["category"].map(
        lambda value: CATEGORY_LABELS.get(str(value), str(value).upper())
    )
    candidate_view["advisor_score"] = (candidate_view["advisor_score"] * 100.0).round(1)
    candidate_view["capability_fit"] = (candidate_view["capability_fit"] * 100.0).round(1)
    candidate_view["latency_fit"] = (candidate_view["latency_fit"] * 100.0).round(1)
    candidate_view["quality_score"] = (candidate_view["quality_score"] * 100.0).round(1)
    candidate_view["migration_score"] = (candidate_view["migration_score"] * 100.0).round(1)
    candidate_view["cost_score"] = (candidate_view["cost_score"] * 100.0).round(1)
    candidate_view["estimated_monthly_cost_usd"] = candidate_view["estimated_monthly_cost_usd"].round(2)
    candidate_view["estimated_cpi_usd"] = candidate_view["estimated_cpi_usd"].round(6)

    return {
        "profile": profile,
        "category_scores": category_scores_pct,
        "primary_recommendation": primary_payload,
        "alternatives": alternatives_payload,
        "candidates_df": candidate_view,
        "inputs_used": {
            "objective": objective_key,
            "complexity_level": complexity_key,
            "max_latency_ms": int(max_latency_ms),
            "monthly_budget_usd": monthly_budget_usd,
            "required_capabilities": sorted(capabilities),
            "primary_task": primary_task,
            "preferred_providers": sorted({p.lower() for p in preferred_providers}),
            "allow_cross_provider": bool(allow_cross_provider),
        },
    }


def _prepare_unified_df(unified_df: pd.DataFrame) -> pd.DataFrame:
    df = unified_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    for col in ["calls", "input_tokens", "output_tokens", "total_tokens", "cost_usd"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "provider" not in df.columns:
        df["provider"] = "unknown"
    if "model" not in df.columns:
        df["model"] = "unknown"
    df["provider"] = df["provider"].astype(str).str.lower()
    df["model"] = df["model"].astype(str)

    token_mask = df["total_tokens"] <= 0
    df.loc[token_mask, "total_tokens"] = df.loc[token_mask, "input_tokens"] + df.loc[token_mask, "output_tokens"]

    return df


def _build_observed_model_stats(unified_df: pd.DataFrame) -> pd.DataFrame:
    if unified_df.empty:
        return pd.DataFrame(
            columns=[
                "provider",
                "model",
                "calls",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
                "observed_cpi_usd",
                "avg_tokens_per_call",
            ]
        )
    df = _prepare_unified_df(unified_df)
    grouped = (
        df.groupby(["provider", "model"], as_index=False)
        .agg(
            calls=("calls", "sum"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
        )
        .sort_values("cost_usd", ascending=False)
    )
    grouped["observed_cpi_usd"] = grouped["cost_usd"] / grouped["calls"].replace(0, pd.NA)
    grouped["observed_cpi_usd"] = grouped["observed_cpi_usd"].fillna(0.0)
    grouped["avg_tokens_per_call"] = grouped["total_tokens"] / grouped["calls"].replace(0, pd.NA)
    grouped["avg_tokens_per_call"] = grouped["avg_tokens_per_call"].fillna(0.0)
    return grouped


def _compute_category_scores(
    *,
    profile: dict[str, Any],
    complexity_level: str,
    max_latency_ms: int,
    monthly_budget_usd: float | None,
    required_capabilities: set[str],
    objective: str,
    providers: tuple[str, ...],
) -> dict[str, float]:
    complexity_map = {
        "low": {"slm": 0.92, "llm": 0.58, "frontier": 0.22},
        "medium": {"slm": 0.58, "llm": 0.88, "frontier": 0.62},
        "high": {"slm": 0.25, "llm": 0.74, "frontier": 0.90},
        "very_high": {"slm": 0.10, "llm": 0.50, "frontier": 1.00},
    }
    base = complexity_map.get(complexity_level, complexity_map["medium"]).copy()

    latency_targets = {"slm": 180.0, "llm": 420.0, "frontier": 1200.0}
    latency_scores = {
        category: max(0.0, min(1.0, float(max_latency_ms) / target))
        for category, target in latency_targets.items()
    }

    cap_strength = {
        "slm": {"reasoning": 0.55, "code": 0.55, "vision": 0.45, "long_context": 0.50, "research": 0.20},
        "llm": {"reasoning": 0.82, "code": 0.85, "vision": 0.72, "long_context": 0.84, "research": 0.65},
        "frontier": {"reasoning": 0.97, "code": 0.95, "vision": 0.88, "long_context": 0.90, "research": 0.93},
    }
    if required_capabilities:
        capability_scores: dict[str, float] = {}
        for category, cap_map in cap_strength.items():
            values = [float(cap_map.get(capability, 0.5)) for capability in required_capabilities]
            capability_scores[category] = sum(values) / len(values)
    else:
        capability_scores = {"slm": 0.70, "llm": 0.82, "frontier": 0.86}

    monthly_calls = float(profile.get("estimated_monthly_calls", 0.0))
    if monthly_calls > 3_000_000:
        volume_scores = {"slm": 1.00, "llm": 0.72, "frontier": 0.30}
    elif monthly_calls > 300_000:
        volume_scores = {"slm": 0.86, "llm": 0.80, "frontier": 0.48}
    else:
        volume_scores = {"slm": 0.62, "llm": 0.82, "frontier": 0.72}

    quality_prior = {"slm": 0.64, "llm": 0.86, "frontier": 0.96}

    budget_scores = _category_budget_scores(
        profile=profile,
        monthly_budget_usd=monthly_budget_usd,
        providers=providers,
    )

    weight_sets = {
        "balanced": {
            "complexity": 0.25,
            "budget": 0.20,
            "latency": 0.15,
            "capability": 0.18,
            "volume": 0.10,
            "quality": 0.12,
        },
        "min_cost": {
            "complexity": 0.15,
            "budget": 0.35,
            "latency": 0.18,
            "capability": 0.10,
            "volume": 0.15,
            "quality": 0.07,
        },
        "max_quality": {
            "complexity": 0.25,
            "budget": 0.05,
            "latency": 0.10,
            "capability": 0.23,
            "volume": 0.05,
            "quality": 0.32,
        },
    }
    weights = weight_sets[objective]

    scores: dict[str, float] = {}
    for category in CATEGORY_LABELS:
        scores[category] = (
            base[category] * weights["complexity"]
            + budget_scores[category] * weights["budget"]
            + latency_scores[category] * weights["latency"]
            + capability_scores[category] * weights["capability"]
            + volume_scores[category] * weights["volume"]
            + quality_prior[category] * weights["quality"]
        )
    return scores


def _category_budget_scores(
    *,
    profile: dict[str, Any],
    monthly_budget_usd: float | None,
    providers: tuple[str, ...],
) -> dict[str, float]:
    if monthly_budget_usd is None or monthly_budget_usd <= 0:
        return {"slm": 0.75, "llm": 0.70, "frontier": 0.55}

    catalog = advisor_model_catalog(providers=providers)
    if catalog.empty:
        return {"slm": 0.0, "llm": 0.0, "frontier": 0.0}

    monthly_calls = float(profile.get("estimated_monthly_calls", 0.0))
    avg_input = float(profile.get("avg_input_tokens_per_call", 0.0))
    avg_output = float(profile.get("avg_output_tokens_per_call", 0.0))

    scores = {"slm": 0.0, "llm": 0.0, "frontier": 0.0}
    for category in CATEGORY_LABELS:
        subset = catalog[catalog["category"] == category]
        if subset.empty:
            continue
        estimated_costs = []
        for _, row in subset.iterrows():
            cpi = _estimate_pricing_cpi(
                input_tokens_per_call=avg_input,
                output_tokens_per_call=avg_output,
                input_price=float(row["input_cost_per_1m"]),
                output_price=float(row["output_cost_per_1m"]),
            )
            estimated_costs.append(cpi * monthly_calls)
        category_cost = min(estimated_costs) if estimated_costs else 0.0
        if category_cost <= 0:
            scores[category] = 1.0
        else:
            ratio = float(monthly_budget_usd) / category_cost
            scores[category] = max(0.05, min(1.0, ratio))
    return scores


def _build_candidate_scores(
    *,
    profile: dict[str, Any],
    observed_stats_df: pd.DataFrame,
    category_scores: dict[str, float],
    objective: str,
    max_latency_ms: int,
    monthly_budget_usd: float | None,
    required_capabilities: set[str],
    preferred_providers: tuple[str, ...],
    allow_cross_provider: bool,
    top_n: int,
) -> pd.DataFrame:
    catalog = advisor_model_catalog(providers=preferred_providers)
    if catalog.empty:
        return pd.DataFrame()

    observed_lookup = {
        (str(row["provider"]).lower(), str(row["model"])): row
        for _, row in observed_stats_df.iterrows()
    }

    current_provider = str(profile.get("current_provider", "unknown")).lower()
    monthly_calls = float(profile.get("estimated_monthly_calls", 0.0))
    avg_input = float(profile.get("avg_input_tokens_per_call", 0.0))
    avg_output = float(profile.get("avg_output_tokens_per_call", 0.0))

    rows: list[dict[str, Any]] = []
    for _, model_row in catalog.iterrows():
        provider = str(model_row["provider"]).lower()
        model = str(model_row["model"])
        category = str(model_row["category"]).lower()

        if not allow_cross_provider and current_provider not in {"", "unknown"} and provider != current_provider:
            continue

        observed_row = observed_lookup.get((provider, model))
        estimated_cpi = 0.0
        cost_source = "pricing_estimate"
        observed_calls = 0.0
        observed_avg_tokens = 0.0
        if observed_row is not None:
            observed_calls = float(observed_row.get("calls", 0.0))
            observed_avg_tokens = float(observed_row.get("avg_tokens_per_call", 0.0))
            observed_cpi = float(observed_row.get("observed_cpi_usd", 0.0))
            if observed_calls >= 20 and observed_cpi > 0:
                estimated_cpi = observed_cpi
                cost_source = "observed_history"

        if estimated_cpi <= 0:
            estimated_cpi = _estimate_pricing_cpi(
                input_tokens_per_call=avg_input,
                output_tokens_per_call=avg_output,
                input_price=float(model_row["input_cost_per_1m"]),
                output_price=float(model_row["output_cost_per_1m"]),
            )
            if estimated_cpi <= 0:
                estimated_cpi = 0.0

        monthly_cost = estimated_cpi * monthly_calls

        capabilities = model_row["capabilities"]
        if isinstance(capabilities, str):
            capabilities_set = {cap.strip() for cap in capabilities.split(",") if cap.strip()}
        elif isinstance(capabilities, set):
            capabilities_set = capabilities
        elif isinstance(capabilities, list):
            capabilities_set = {str(item) for item in capabilities}
        else:
            capabilities_set = set()

        if required_capabilities:
            cap_hits = sum(1 for cap in required_capabilities if cap in capabilities_set)
            capability_fit = cap_hits / float(len(required_capabilities))
        else:
            capability_fit = 0.85

        latency_ms = float(model_row["latency_ms"])
        latency_fit = max(0.0, min(1.0, float(max_latency_ms) / latency_ms)) if latency_ms > 0 else 0.0
        migration_score = (
            1.0 if provider == current_provider else (0.82 if allow_cross_provider else 0.0)
        )

        rows.append(
            {
                "provider": provider,
                "model": model,
                "category": category,
                "best_for": str(model_row["best_for"]),
                "estimated_cpi_usd": estimated_cpi,
                "estimated_monthly_cost_usd": monthly_cost,
                "cost_source": cost_source,
                "latency_ms": latency_ms,
                "latency_fit": latency_fit,
                "quality_score": float(model_row["quality_score"]),
                "capability_fit": capability_fit,
                "migration_score": migration_score,
                "category_fit": max(0.0, min(1.0, float(category_scores.get(category, 0.0)))),
                "observed_calls": observed_calls,
                "observed_avg_tokens": observed_avg_tokens,
            }
        )

    candidates_df = pd.DataFrame(rows)
    if candidates_df.empty:
        return candidates_df

    min_cost = float(candidates_df["estimated_monthly_cost_usd"].min())
    max_cost = float(candidates_df["estimated_monthly_cost_usd"].max())
    if max_cost > min_cost:
        candidates_df["cost_score"] = 1.0 - (
            (candidates_df["estimated_monthly_cost_usd"] - min_cost) / (max_cost - min_cost)
        )
    else:
        candidates_df["cost_score"] = 1.0

    weight_sets = {
        "balanced": {
            "category_fit": 0.22,
            "cost_score": 0.26,
            "quality_score": 0.20,
            "latency_fit": 0.14,
            "capability_fit": 0.13,
            "migration_score": 0.05,
        },
        "min_cost": {
            "category_fit": 0.15,
            "cost_score": 0.45,
            "quality_score": 0.10,
            "latency_fit": 0.13,
            "capability_fit": 0.07,
            "migration_score": 0.10,
        },
        "max_quality": {
            "category_fit": 0.22,
            "cost_score": 0.08,
            "quality_score": 0.35,
            "latency_fit": 0.10,
            "capability_fit": 0.20,
            "migration_score": 0.05,
        },
    }
    weights = weight_sets[objective]

    candidates_df["advisor_score"] = 0.0
    for metric, weight in weights.items():
        candidates_df["advisor_score"] += pd.to_numeric(candidates_df[metric], errors="coerce").fillna(0.0) * weight

    if monthly_budget_usd is not None and monthly_budget_usd > 0:
        over_budget_mask = candidates_df["estimated_monthly_cost_usd"] > float(monthly_budget_usd)
        candidates_df.loc[over_budget_mask, "advisor_score"] *= 0.85

    candidates_df = candidates_df.sort_values(
        ["advisor_score", "quality_score", "cost_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return candidates_df.head(max(1, int(top_n)))


def _estimate_pricing_cpi(
    *,
    input_tokens_per_call: float,
    output_tokens_per_call: float,
    input_price: float,
    output_price: float,
) -> float:
    return (max(0.0, input_tokens_per_call) / 1_000_000) * max(0.0, input_price) + (
        max(0.0, output_tokens_per_call) / 1_000_000
    ) * max(0.0, output_price)


def _estimate_baseline_monthly_cost(
    *,
    candidates_df: pd.DataFrame,
    profile: dict[str, Any],
    current_provider: str,
    current_model: str,
) -> float:
    current_match = candidates_df[
        (candidates_df["provider"] == current_provider.lower())
        & (candidates_df["model"] == current_model)
    ]
    if not current_match.empty:
        return float(current_match.iloc[0]["estimated_monthly_cost_usd"])

    observed_monthly = float(profile.get("estimated_monthly_spend_usd", 0.0))
    if observed_monthly > 0:
        return observed_monthly

    return float(candidates_df["estimated_monthly_cost_usd"].median())


def _estimate_confidence(
    *,
    top_score: float,
    second_score: float,
    total_calls: float,
    cost_source: str,
) -> float:
    gap = max(0.0, top_score - second_score)
    volume_factor = min(1.0, math.log10(max(1.0, total_calls)) / 4.0)
    source_bonus = 0.08 if cost_source == "observed_history" else 0.03
    confidence = 0.44 + (gap * 0.45) + (volume_factor * 0.22) + source_bonus
    return max(35.0, min(98.0, confidence * 100.0))


def _build_primary_reasoning(
    *,
    primary: pd.Series,
    objective: str,
    category_scores: dict[str, float],
    required_capabilities: set[str],
    savings_usd: float,
) -> list[str]:
    reasons: list[str] = []

    category = str(primary["category"])
    category_label = CATEGORY_LABELS.get(category, category.upper())
    category_fit = float(category_scores.get(category, 0.0)) * 100.0
    reasons.append(f"{category_label} category scored {category_fit:.1f}% for this workload profile.")

    if objective == "min_cost":
        reasons.append("Objective prioritized spend reduction with acceptable quality/latency fit.")
    elif objective == "max_quality":
        reasons.append("Objective prioritized quality and reasoning strength over pure cost minimization.")
    else:
        reasons.append("Objective balanced cost, quality, latency, and migration friction.")

    if required_capabilities:
        matched_caps = int(round(float(primary["capability_fit"]) * len(required_capabilities)))
        reasons.append(
            f"Capability match: {matched_caps}/{len(required_capabilities)} requested capabilities."
        )
    else:
        reasons.append("No strict capability constraints were set, enabling broader optimization.")

    if savings_usd > 0:
        reasons.append(f"Estimated monthly savings vs baseline: ${savings_usd:.2f}.")
    elif savings_usd < 0:
        reasons.append(f"Estimated monthly cost increase vs baseline: ${abs(savings_usd):.2f}.")
    else:
        reasons.append("Estimated monthly spend is near baseline parity.")

    reasons.append(f"Model fit summary: {str(primary['best_for'])}.")
    return reasons


def _build_tradeoffs(
    *,
    primary: pd.Series,
    max_latency_ms: int,
    savings_usd: float,
    allow_cross_provider: bool,
    current_provider: str,
) -> list[str]:
    trade_offs: list[str] = []
    latency = float(primary["latency_ms"])
    if latency > float(max_latency_ms):
        trade_offs.append(
            "Estimated latency may exceed the configured threshold; validate with a pilot workload."
        )

    if savings_usd < 0:
        trade_offs.append("Recommendation increases spend and is justified only if quality gains are required.")

    provider = str(primary["provider"]).lower()
    if allow_cross_provider and current_provider not in {"", "unknown"} and provider != current_provider:
        trade_offs.append("Migration crosses providers and may require SDK/runtime integration adjustments.")

    if float(primary["quality_score"]) < 0.75:
        trade_offs.append("Lower capability tier may underperform for high-complexity reasoning tasks.")

    if not trade_offs:
        trade_offs.append("No major trade-offs detected for the selected constraints; validate with A/B tests.")

    return trade_offs


def _provider_label(provider: str) -> str:
    mapping = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
    }
    key = provider.strip().lower()
    return mapping.get(key, key.title())
