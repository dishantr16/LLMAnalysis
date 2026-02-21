"""Aggregation logic for dashboard metrics and summaries."""

from __future__ import annotations

import calendar
from typing import Any

import numpy as np
import pandas as pd

from src.config import MODEL_PRICING_PER_MILLION


def aggregate_usage(usage_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if usage_df.empty:
        return pd.DataFrame(
            columns=["period", "input_tokens", "output_tokens", "total_tokens", "requests"]
        )

    period_source = pd.to_datetime(usage_df["bucket_start"], utc=True).dt.tz_localize(None)
    grouped = (
        usage_df.assign(period=period_source.dt.floor("D").dt.to_period(freq).dt.to_timestamp())
        .groupby("period", as_index=False)
        .agg(
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            requests=("requests", "sum"),
        )
        .sort_values("period")
    )
    return grouped


def aggregate_cost(cost_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if cost_df.empty:
        return pd.DataFrame(columns=["period", "amount"])

    period_source = pd.to_datetime(cost_df["bucket_start"], utc=True).dt.tz_localize(None)
    grouped = (
        cost_df.assign(period=period_source.dt.floor("D").dt.to_period(freq).dt.to_timestamp())
        .groupby("period", as_index=False)
        .agg(amount=("amount", "sum"))
        .sort_values("period")
    )
    return grouped


def build_model_summary(
    usage_df: pd.DataFrame,
    *,
    pricing_map: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    if usage_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "requests",
                "estimated_cost_usd",
                "avg_estimated_cost_per_request",
                "has_pricing",
            ]
        )

    pricing = pricing_map or MODEL_PRICING_PER_MILLION

    grouped = (
        usage_df.groupby("model", as_index=False)
        .agg(
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            requests=("requests", "sum"),
        )
        .sort_values("total_tokens", ascending=False)
    )

    estimated_costs: list[float] = []
    has_pricing: list[bool] = []

    for _, row in grouped.iterrows():
        model = str(row["model"]).lower()
        model_pricing = pricing.get(model)
        if not model_pricing:
            estimated_costs.append(float("nan"))
            has_pricing.append(False)
            continue

        estimated_cost = (
            (row["input_tokens"] / 1_000_000) * float(model_pricing.get("input", 0))
            + (row["output_tokens"] / 1_000_000) * float(model_pricing.get("output", 0))
        )
        estimated_costs.append(estimated_cost)
        has_pricing.append(True)

    grouped["estimated_cost_usd"] = estimated_costs
    grouped["has_pricing"] = has_pricing
    grouped["avg_estimated_cost_per_request"] = grouped["estimated_cost_usd"] / grouped["requests"].replace(
        0, pd.NA
    )
    return grouped


def build_dimension_summary(
    usage_df: pd.DataFrame,
    dimension: str,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    if usage_df.empty or dimension not in usage_df.columns:
        return pd.DataFrame(columns=[dimension, "total_tokens", "requests"])

    grouped = (
        usage_df.groupby(dimension, as_index=False)
        .agg(total_tokens=("total_tokens", "sum"), requests=("requests", "sum"))
        .sort_values("total_tokens", ascending=False)
    )
    return grouped.head(top_n)


def build_project_cost_summary(cost_df: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    if cost_df.empty:
        return pd.DataFrame(columns=["project_id", "amount"])

    grouped = (
        cost_df.groupby("project_id", as_index=False)
        .agg(amount=("amount", "sum"))
        .sort_values("amount", ascending=False)
    )
    return grouped.head(top_n)


def build_line_item_cost_summary(cost_df: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    if cost_df.empty:
        return pd.DataFrame(columns=["line_item", "amount"])

    grouped = (
        cost_df.groupby("line_item", as_index=False)
        .agg(amount=("amount", "sum"))
        .sort_values("amount", ascending=False)
    )
    return grouped.head(top_n)


def build_token_distribution(model_summary: pd.DataFrame) -> pd.DataFrame:
    if model_summary.empty:
        return pd.DataFrame(columns=["token_type", "tokens"])

    totals = {
        "Input Tokens": float(model_summary["input_tokens"].sum()),
        "Output Tokens": float(model_summary["output_tokens"].sum()),
    }
    return pd.DataFrame(
        [{"token_type": token_type, "tokens": tokens} for token_type, tokens in totals.items()]
    )


def compute_kpis(
    usage_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> dict[str, Any]:
    total_input_tokens = int(usage_df["input_tokens"].sum()) if not usage_df.empty else 0
    total_output_tokens = int(usage_df["output_tokens"].sum()) if not usage_df.empty else 0
    total_tokens = int(usage_df["total_tokens"].sum()) if not usage_df.empty else 0
    total_requests = int(usage_df["requests"].sum()) if not usage_df.empty else 0

    reported_cost = float(cost_df["amount"].sum()) if not cost_df.empty else 0.0
    estimated_cost = (
        float(model_summary["estimated_cost_usd"].dropna().sum()) if not model_summary.empty else 0.0
    )

    currency = "usd"
    if not cost_df.empty and "currency" in cost_df.columns:
        currencies = [c for c in cost_df["currency"].dropna().unique() if c]
        if currencies:
            currency = str(currencies[0]).lower()

    avg_reported_cost_per_request = reported_cost / total_requests if total_requests else 0.0

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_requests": total_requests,
        "active_models": int(usage_df["model"].nunique()) if not usage_df.empty else 0,
        "reported_cost": reported_cost,
        "estimated_cost": estimated_cost,
        "reconciliation_delta": estimated_cost - reported_cost,
        "avg_reported_cost_per_request": avg_reported_cost_per_request,
        "currency": currency,
    }


def extract_metric_columns(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []

    non_metric_columns = {
        "dataset",
        "bucket_start",
        "bucket_end",
        "model",
        "project_id",
        "user_id",
        "api_key_id",
        "batch",
        "size",
        "source",
        "service_tier",
    }
    return [
        col
        for col in df.columns
        if col not in non_metric_columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def aggregate_generic_usage(generic_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if generic_df.empty:
        return pd.DataFrame(columns=["period"])

    metric_columns = extract_metric_columns(generic_df)
    if not metric_columns:
        return pd.DataFrame(columns=["period"])

    period_source = pd.to_datetime(generic_df["bucket_start"], utc=True).dt.tz_localize(None)
    grouped = (
        generic_df.assign(period=period_source.dt.floor("D").dt.to_period(freq).dt.to_timestamp())
        .groupby("period", as_index=False)[metric_columns]
        .sum()
        .sort_values("period")
    )
    return grouped


def build_generic_metric_dimension_summary(
    generic_df: pd.DataFrame,
    *,
    metric: str,
    dimension: str,
    top_n: int = 10,
) -> pd.DataFrame:
    if generic_df.empty or metric not in generic_df.columns or dimension not in generic_df.columns:
        return pd.DataFrame(columns=[dimension, metric])

    grouped = (
        generic_df.groupby(dimension, as_index=False)
        .agg(**{metric: (metric, "sum")})
        .sort_values(metric, ascending=False)
    )
    return grouped.head(top_n)


def build_baseline_forecast(
    series_df: pd.DataFrame,
    *,
    value_column: str,
    horizon_days: int = 30,
) -> pd.DataFrame:
    if series_df.empty or value_column not in series_df.columns:
        return pd.DataFrame(columns=["period", value_column])

    history = (
        series_df[["period", value_column]]
        .copy()
        .dropna(subset=["period"])
        .sort_values("period")
    )
    history[value_column] = pd.to_numeric(history[value_column], errors="coerce").fillna(0.0)
    if history.empty:
        return pd.DataFrame(columns=["period", value_column])

    y = history[value_column].to_numpy(dtype=float)
    count = len(y)
    if count < 3:
        slope = 0.0
        intercept = float(y.mean())
    else:
        slope, intercept = np.polyfit(np.arange(count), y, 1)

    idx = np.arange(count, count + horizon_days)
    predictions = np.maximum(0.0, slope * idx + intercept)
    first_future_date = pd.to_datetime(history["period"].iloc[-1]) + pd.Timedelta(days=1)

    return pd.DataFrame(
        {
            "period": pd.date_range(first_future_date, periods=horizon_days, freq="D"),
            value_column: predictions,
        }
    )


def build_actual_vs_forecast(
    series_df: pd.DataFrame,
    *,
    value_column: str,
    horizon_days: int = 30,
) -> pd.DataFrame:
    if series_df.empty or value_column not in series_df.columns:
        return pd.DataFrame(columns=["period", value_column, "series"])

    actual = (
        series_df[["period", value_column]]
        .copy()
        .dropna(subset=["period"])
        .sort_values("period")
        .assign(series="actual")
    )
    forecast = build_baseline_forecast(
        series_df,
        value_column=value_column,
        horizon_days=horizon_days,
    ).assign(series="forecast")

    return pd.concat([actual, forecast], ignore_index=True)


def project_current_month_total(
    series_df: pd.DataFrame,
    *,
    value_column: str,
) -> dict[str, float]:
    if series_df.empty or value_column not in series_df.columns:
        return {
            "actual_to_date": 0.0,
            "avg_daily_projection": 0.0,
            "linear_projection": 0.0,
            "days_elapsed": 0.0,
            "days_remaining": 0.0,
        }

    values = (
        series_df[["period", value_column]]
        .copy()
        .dropna(subset=["period"])
        .sort_values("period")
    )
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce").fillna(0.0)
    values["period"] = pd.to_datetime(values["period"]).dt.normalize()

    latest_period = values["period"].max()
    month_mask = (
        (values["period"].dt.month == latest_period.month)
        & (values["period"].dt.year == latest_period.year)
    )
    month_data = values[month_mask]
    actual_to_date = float(month_data[value_column].sum())

    days_in_month = calendar.monthrange(latest_period.year, latest_period.month)[1]
    days_elapsed = float(int(month_data["period"].dt.day.max())) if not month_data.empty else 0.0
    days_remaining = float(max(0, days_in_month - int(days_elapsed)))
    avg_daily_projection = (actual_to_date / days_elapsed) * days_in_month if days_elapsed else 0.0

    if month_data.empty or days_remaining <= 0:
        linear_projection = actual_to_date
    else:
        forecast = build_baseline_forecast(
            month_data.rename(columns={value_column: "metric"}),
            value_column="metric",
            horizon_days=int(days_remaining),
        )
        linear_projection = actual_to_date + float(forecast["metric"].sum())

    return {
        "actual_to_date": actual_to_date,
        "avg_daily_projection": avg_daily_projection,
        "linear_projection": linear_projection,
        "days_elapsed": days_elapsed,
        "days_remaining": days_remaining,
    }


def monthly_spend_trend(unified_df: pd.DataFrame) -> pd.DataFrame:
    if unified_df.empty:
        return pd.DataFrame(columns=["period", "cost_usd"])

    df = unified_df.copy()
    df["period"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    return (
        df.groupby("period", as_index=False)
        .agg(cost_usd=("cost_usd", "sum"))
        .sort_values("period")
    )


def spend_by_provider(unified_df: pd.DataFrame) -> pd.DataFrame:
    if unified_df.empty:
        return pd.DataFrame(columns=["provider", "cost_usd"])

    return (
        unified_df.groupby("provider", as_index=False)
        .agg(cost_usd=("cost_usd", "sum"))
        .sort_values("cost_usd", ascending=False)
    )


def top_models_by_cost(unified_df: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    if unified_df.empty:
        return pd.DataFrame(columns=["model", "provider", "cost_usd"])

    grouped = (
        unified_df.groupby(["model", "provider"], as_index=False)
        .agg(cost_usd=("cost_usd", "sum"))
        .sort_values("cost_usd", ascending=False)
    )
    return grouped.head(top_n)


def model_cost_breakdown(unified_df: pd.DataFrame, *, top_n: int = 25) -> pd.DataFrame:
    if unified_df.empty:
        return pd.DataFrame(
            columns=[
                "Rank",
                "Model",
                "Provider",
                "Calls (24h)",
                "Avg Tokens",
                "CPI",
                "7-Day Trend",
                "Status",
            ]
        )

    df = unified_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    latest_ts = df["timestamp"].max()

    grouped = (
        df.groupby(["model", "provider"], as_index=False)
        .agg(
            total_cost=("cost_usd", "sum"),
            total_calls=("calls", "sum"),
            total_tokens=("total_tokens", "sum"),
        )
        .sort_values("total_cost", ascending=False)
    )

    last_24h = df[df["timestamp"] >= (latest_ts - pd.Timedelta(hours=24))]
    calls_24h = (
        last_24h.groupby(["model", "provider"], as_index=False)
        .agg(calls_24h=("calls", "sum"))
    )
    grouped = grouped.merge(calls_24h, on=["model", "provider"], how="left")
    grouped["calls_24h"] = grouped["calls_24h"].fillna(0)

    grouped["avg_tokens"] = grouped["total_tokens"] / grouped["total_calls"].replace(0, np.nan)
    grouped["avg_tokens"] = grouped["avg_tokens"].fillna(0)
    grouped["cpi"] = grouped["total_cost"] / grouped["total_calls"].replace(0, np.nan)
    grouped["cpi"] = grouped["cpi"].fillna(0)

    trends: list[float] = []
    statuses: list[str] = []
    for _, row in grouped.iterrows():
        model = row["model"]
        provider = row["provider"]
        mdf = df[(df["model"] == model) & (df["provider"] == provider)]
        recent = mdf[mdf["timestamp"] >= (latest_ts - pd.Timedelta(days=7))]["cost_usd"].sum()
        previous = mdf[
            (mdf["timestamp"] < (latest_ts - pd.Timedelta(days=7)))
            & (mdf["timestamp"] >= (latest_ts - pd.Timedelta(days=14)))
        ]["cost_usd"].sum()

        if previous <= 0:
            trend_pct = 100.0 if recent > 0 else 0.0
        else:
            trend_pct = ((recent - previous) / previous) * 100.0
        trends.append(trend_pct)

        if trend_pct > 15:
            statuses.append("Rising")
        elif trend_pct < -15:
            statuses.append("Declining")
        else:
            statuses.append("Stable")

    grouped["trend_pct"] = trends
    grouped["status"] = statuses
    grouped = grouped.head(top_n).reset_index(drop=True)
    grouped["rank"] = grouped.index + 1

    result = pd.DataFrame(
        {
            "Rank": grouped["rank"],
            "Model": grouped["model"],
            "Provider": grouped["provider"].str.replace("_", " ").str.title(),
            "Calls (24h)": grouped["calls_24h"].round(0).astype(int),
            "Avg Tokens": grouped["avg_tokens"].round(2),
            "CPI": grouped["cpi"].map(lambda x: f"${x:.6f}"),
            "7-Day Trend": grouped["trend_pct"].map(lambda x: f"{x:+.1f}%"),
            "Status": grouped["status"],
        }
    )
    return result
