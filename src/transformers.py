"""Payload normalization into pandas DataFrames."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

COMMON_DIMENSION_COLUMNS = [
    "model",
    "project_id",
    "user_id",
    "api_key_id",
    "batch",
    "size",
    "source",
    "service_tier",
]
GENERIC_USAGE_BASE_COLUMNS = ["dataset", "bucket_start", "bucket_end", *COMMON_DIMENSION_COLUMNS]

USAGE_COLUMNS = [
    "bucket_start",
    "bucket_end",
    "model",
    "project_id",
    "user_id",
    "api_key_id",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "input_audio_tokens",
    "output_audio_tokens",
    "requests",
]

COST_COLUMNS = [
    "bucket_start",
    "bucket_end",
    "project_id",
    "line_item",
    "amount",
    "currency",
]


def build_usage_df(buckets: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for bucket in buckets:
        start_time = bucket.get("start_time")
        end_time = bucket.get("end_time")
        results = bucket.get("results") or []

        for result in results:
            rows.append(
                {
                    "bucket_start": _to_utc_datetime(start_time),
                    "bucket_end": _to_utc_datetime(end_time),
                    "model": result.get("model") or "unknown",
                    "project_id": result.get("project_id") or "unknown",
                    "user_id": result.get("user_id") or "unknown",
                    "api_key_id": result.get("api_key_id") or "unknown",
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "cached_input_tokens": result.get("input_cached_tokens", 0),
                    "reasoning_tokens": result.get(
                        "reasoning_tokens",
                        result.get("output_tokens_details", {}).get("reasoning_tokens", 0)
                        if isinstance(result.get("output_tokens_details"), dict) else 0,
                    ),
                    "input_audio_tokens": result.get("input_audio_tokens", 0),
                    "output_audio_tokens": result.get("output_audio_tokens", 0),
                    "requests": result.get("num_model_requests", result.get("num_requests", 0)),
                }
            )

    df = pd.DataFrame(rows, columns=USAGE_COLUMNS)
    if df.empty:
        return df

    numeric_columns = [
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "input_audio_tokens",
        "output_audio_tokens",
        "requests",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
    return df


def build_generic_usage_df(dataset: str, buckets: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Normalize non-completions usage endpoints to a metrics table."""
    rows: list[dict[str, Any]] = []

    for bucket in buckets:
        start_time = bucket.get("start_time")
        end_time = bucket.get("end_time")
        results = bucket.get("results") or []

        for result in results:
            row: dict[str, Any] = {
                "dataset": dataset,
                "bucket_start": _to_utc_datetime(start_time),
                "bucket_end": _to_utc_datetime(end_time),
            }
            for col in COMMON_DIMENSION_COLUMNS:
                row[col] = result.get(col) or "unknown"

            for key, value in result.items():
                if key in set(COMMON_DIMENSION_COLUMNS) | {"object"}:
                    continue
                if _is_number(value):
                    row[key] = value

            if "requests" not in row:
                if "num_model_requests" in row:
                    row["requests"] = row["num_model_requests"]
                elif "num_requests" in row:
                    row["requests"] = row["num_requests"]

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=GENERIC_USAGE_BASE_COLUMNS)

    for col in GENERIC_USAGE_BASE_COLUMNS:
        if col not in df.columns:
            df[col] = "unknown" if col not in {"bucket_start", "bucket_end"} else pd.NaT

    metric_columns = [
        col
        for col in df.columns
        if col not in GENERIC_USAGE_BASE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in metric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "input_tokens" in df.columns and "output_tokens" in df.columns:
        df["total_tokens"] = df["input_tokens"] + df["output_tokens"]

    ordered_cols = GENERIC_USAGE_BASE_COLUMNS + [
        col for col in df.columns if col not in GENERIC_USAGE_BASE_COLUMNS
    ]
    return df[ordered_cols].sort_values("bucket_start")


def build_cost_df(buckets: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for bucket in buckets:
        start_time = bucket.get("start_time")
        end_time = bucket.get("end_time")
        results = bucket.get("results") or []

        for result in results:
            amount_payload = result.get("amount") or {}
            amount_value = 0
            currency = "usd"

            if isinstance(amount_payload, dict):
                amount_value = amount_payload.get("value", 0)
                currency = str(amount_payload.get("currency", "usd")).lower()

            rows.append(
                {
                    "bucket_start": _to_utc_datetime(start_time),
                    "bucket_end": _to_utc_datetime(end_time),
                    "project_id": result.get("project_id") or "unassigned",
                    "line_item": result.get("line_item") or "unattributed",
                    "amount": amount_value,
                    "currency": currency,
                }
            )

    df = pd.DataFrame(rows, columns=COST_COLUMNS)
    if df.empty:
        return df

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def _to_utc_datetime(unix_seconds: Any) -> pd.Timestamp:
    if unix_seconds is None:
        return pd.NaT
    return pd.to_datetime(int(unix_seconds), unit="s", utc=True)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
