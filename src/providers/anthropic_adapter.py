"""Anthropic provider adapter implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from src.config import (
    ANTHROPIC_API_BASE_URL,
    ANTHROPIC_COST_REPORT_ENDPOINT,
    ANTHROPIC_USAGE_REPORT_MESSAGES_ENDPOINT,
    ANTHROPIC_VERSION,
    MAX_PAGES,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
)
from src.providers.base import ProviderAdapter, ProviderFetchResult, empty_unified_df

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class AnthropicAPIError(Exception):
    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"{self.message} (status={self.status_code})"


class AnthropicProviderAdapter(ProviderAdapter):
    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = ANTHROPIC_API_BASE_URL,
        version: str = ANTHROPIC_VERSION,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def fetch(
        self,
        *,
        start_time: int,
        end_time: int,
        bucket_width: str,
        group_by: tuple[str, ...],
    ) -> ProviderFetchResult:
        if not self.is_configured():
            return ProviderFetchResult(
                provider=self.provider_name,
                unified_df=empty_unified_df(),
                notices=["Anthropic key not configured."],
            )

        usage_group_dimensions = _anthropic_usage_group_dimensions(group_by)
        usage_params = {
            "starting_at": _unix_to_iso8601(start_time),
            "ending_at": _unix_to_iso8601(end_time),
            "bucket_width": _anthropic_bucket_width(bucket_width),
            "group_by[]": usage_group_dimensions,
            "limit": _anthropic_limit(bucket_width),
        }
        cost_params = {
            "starting_at": _unix_to_iso8601(start_time),
            "ending_at": _unix_to_iso8601(end_time),
            "bucket_width": "1d",
            "group_by[]": ["workspace_id", "description"],
            "limit": 31,
        }

        endpoint_errors: dict[str, str] = {}
        notices: list[str] = []
        if bucket_width == "1h":
            notices.append(
                "Anthropic cost report only supports daily buckets (`1d`); hourly selection applies to usage only."
            )

        usage_rows: list[dict[str, Any]] = []
        cost_rows: list[dict[str, Any]] = []

        with requests.Session() as session:
            session.headers.update(
                {
                    "x-api-key": self.api_key,
                    "anthropic-version": self.version,
                    "content-type": "application/json",
                }
            )

            try:
                usage_rows = _get_paginated(
                    session=session,
                    base_url=self.base_url,
                    path=ANTHROPIC_USAGE_REPORT_MESSAGES_ENDPOINT,
                    params=usage_params,
                    timeout_seconds=self.timeout_seconds,
                    max_retries=self.max_retries,
                )
            except AnthropicAPIError as exc:
                endpoint_errors["usage_report_messages"] = str(exc)

            try:
                cost_rows = _get_paginated(
                    session=session,
                    base_url=self.base_url,
                    path=ANTHROPIC_COST_REPORT_ENDPOINT,
                    params=cost_params,
                    timeout_seconds=self.timeout_seconds,
                    max_retries=self.max_retries,
                )
            except AnthropicAPIError as exc:
                endpoint_errors["cost_report"] = str(exc)

        usage_df = build_anthropic_usage_df(usage_rows)
        cost_df = build_anthropic_cost_df(cost_rows)
        unified_df = build_anthropic_unified_df(usage_df, cost_df)

        if usage_df.empty and cost_df.empty:
            notices.append("Anthropic returned no usage/cost rows for selected range.")
        if not usage_df.empty and cost_df.empty:
            notices.append(
                "Anthropic usage was returned, but cost report returned no rows. Cost columns are zero."
            )

        return ProviderFetchResult(
            provider=self.provider_name,
            unified_df=unified_df,
            raw_payload={
                "usage_df": usage_df,
                "cost_df": cost_df,
                "usage_rows": usage_rows,
                "cost_rows": cost_rows,
            },
            endpoint_errors=endpoint_errors,
            notices=notices,
        )


def build_anthropic_usage_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []

    for item in rows:
        bucket_start = _to_utc_timestamp(item.get("starting_at") or item.get("start_time"))
        bucket_end = _to_utc_timestamp(item.get("ending_at") or item.get("end_time"))
        record_list = item.get("results") if isinstance(item.get("results"), list) else [item]

        for record in record_list:
            row_start = _fallback_timestamp(
                _to_utc_timestamp(record.get("starting_at") or record.get("start_time")),
                bucket_start,
            )
            row_end = _fallback_timestamp(
                _to_utc_timestamp(record.get("ending_at") or record.get("end_time")),
                bucket_end,
            )

            input_tokens = _extract_input_tokens(record)
            output_tokens = _to_float(record.get("output_tokens"))
            requests_count = _to_float(
                record.get("requests")
                or record.get("num_requests")
                or record.get("message_requests")
                or record.get("num_model_requests")
            )

            normalized.append(
                {
                    "bucket_start": row_start,
                    "bucket_end": row_end,
                    "model": record.get("model") or item.get("model") or "unknown",
                    "project_id": (
                        record.get("workspace_id")
                        or record.get("project_id")
                        or item.get("workspace_id")
                        or item.get("project_id")
                        or "unknown"
                    ),
                    "user_id": record.get("user_id") or item.get("user_id") or "unknown",
                    "api_key_id": record.get("api_key_id") or item.get("api_key_id") or "unknown",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "requests": requests_count,
                }
            )

    columns = [
        "bucket_start",
        "bucket_end",
        "model",
        "project_id",
        "user_id",
        "api_key_id",
        "input_tokens",
        "output_tokens",
        "requests",
    ]
    df = pd.DataFrame(normalized, columns=columns)
    if df.empty:
        return df

    numeric_columns = ["input_tokens", "output_tokens", "requests"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["total_tokens"] = df["input_tokens"] + df["output_tokens"]

    return df.sort_values("bucket_start")


def build_anthropic_cost_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []

    for item in rows:
        bucket_start = _to_utc_timestamp(item.get("starting_at") or item.get("start_time"))
        bucket_end = _to_utc_timestamp(item.get("ending_at") or item.get("end_time"))
        record_list = item.get("results") if isinstance(item.get("results"), list) else [item]

        for record in record_list:
            row_start = _fallback_timestamp(
                _to_utc_timestamp(record.get("starting_at") or record.get("start_time")),
                bucket_start,
            )
            row_end = _fallback_timestamp(
                _to_utc_timestamp(record.get("ending_at") or record.get("end_time")),
                bucket_end,
            )
            amount_usd = _extract_cost_usd(record)

            normalized.append(
                {
                    "bucket_start": row_start,
                    "bucket_end": row_end,
                    "project_id": (
                        record.get("workspace_id")
                        or record.get("project_id")
                        or item.get("workspace_id")
                        or item.get("project_id")
                        or "unknown"
                    ),
                    "model": record.get("model") or item.get("model") or "unknown",
                    "amount": amount_usd,
                    "currency": str(record.get("currency") or item.get("currency") or "usd").lower(),
                }
            )

    columns = ["bucket_start", "bucket_end", "project_id", "model", "amount", "currency"]
    df = pd.DataFrame(normalized, columns=columns)
    if df.empty:
        return df

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df.sort_values("bucket_start")


def build_anthropic_unified_df(usage_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    if usage_df.empty and cost_df.empty:
        return empty_unified_df()

    if usage_df.empty and not cost_df.empty:
        return _cost_only_unified(cost_df)

    usage = usage_df.copy()
    usage["timestamp"] = pd.to_datetime(usage["bucket_start"], utc=True)
    usage["provider"] = "anthropic"
    usage["project_id"] = usage["project_id"].fillna("unknown")
    usage["model"] = usage["model"].fillna("unknown")
    usage["calls"] = pd.to_numeric(usage["requests"], errors="coerce").fillna(0.0)
    usage["input_tokens"] = pd.to_numeric(usage["input_tokens"], errors="coerce").fillna(0.0)
    usage["output_tokens"] = pd.to_numeric(usage["output_tokens"], errors="coerce").fillna(0.0)
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    usage["cost_usd"] = 0.0
    usage["currency"] = "usd"
    usage["cost_source"] = "unavailable"

    if not cost_df.empty:
        usage = _assign_reported_costs(usage, cost_df)
        usage["cost_source"] = "reported_allocated"

    return usage[
        [
            "timestamp",
            "provider",
            "model",
            "project_id",
            "calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "currency",
            "cost_source",
        ]
    ].sort_values("timestamp")


def _assign_reported_costs(usage_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    usage = usage_df.copy()
    cost = cost_df.copy()

    usage["date"] = pd.to_datetime(usage["timestamp"], utc=True).dt.floor("D")
    cost["date"] = pd.to_datetime(cost["bucket_start"], utc=True).dt.floor("D")
    cost["project_id"] = cost["project_id"].fillna("unknown")
    cost["model"] = cost["model"].fillna("unknown")

    direct_cost = (
        cost[cost["model"] != "unknown"]
        .groupby(["date", "project_id", "model"], as_index=False)
        .agg(amount=("amount", "sum"))
        .rename(columns={"amount": "direct_amount"})
    )
    usage = usage.merge(direct_cost, on=["date", "project_id", "model"], how="left")
    usage["cost_usd"] = usage["direct_amount"].fillna(0.0)
    usage = usage.drop(columns=["direct_amount"])

    unattributed = (
        cost[cost["model"] == "unknown"]
        .groupby(["date", "project_id"], as_index=False)
        .agg(unattributed_amount=("amount", "sum"))
    )
    if not unattributed.empty:
        for _, row in unattributed.iterrows():
            mask = (usage["date"] == row["date"]) & (usage["project_id"] == row["project_id"])
            idx = usage[mask].index.tolist()
            amount = float(row["unattributed_amount"])
            if not idx:
                extra = {
                    "timestamp": row["date"],
                    "provider": "anthropic",
                    "model": "unattributed",
                    "project_id": row["project_id"],
                    "calls": 0.0,
                    "input_tokens": 0.0,
                    "output_tokens": 0.0,
                    "total_tokens": 0.0,
                    "cost_usd": amount,
                    "currency": "usd",
                    "cost_source": "reported_allocated",
                    "date": row["date"],
                }
                usage = pd.concat([usage, pd.DataFrame([extra])], ignore_index=True)
                continue

            weights = usage.loc[idx, "total_tokens"].to_numpy(dtype=float)
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                weights = usage.loc[idx, "calls"].to_numpy(dtype=float)
                weight_sum = float(weights.sum())

            if weight_sum <= 0:
                share = amount / max(1, len(idx))
                usage.loc[idx, "cost_usd"] += share
            else:
                usage.loc[idx, "cost_usd"] += (weights / weight_sum) * amount

    unmatched = _cost_rows_without_matching_usage(usage, cost)
    if not unmatched.empty:
        usage = pd.concat([usage, unmatched], ignore_index=True)

    return usage.drop(columns=["date"])


def _cost_rows_without_matching_usage(usage_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    usage_keys = usage_df[["timestamp", "project_id", "model"]].copy()
    usage_keys["date"] = pd.to_datetime(usage_keys["timestamp"], utc=True).dt.floor("D")
    usage_key_set = {
        (row["date"], str(row["project_id"]), str(row["model"])) for _, row in usage_keys.iterrows()
    }

    extras: list[dict[str, Any]] = []
    grouped = (
        cost_df[cost_df["model"] != "unknown"]
        .copy()
        .assign(date=lambda d: pd.to_datetime(d["bucket_start"], utc=True).dt.floor("D"))
        .groupby(["date", "project_id", "model"], as_index=False)
        .agg(amount=("amount", "sum"))
    )
    for _, row in grouped.iterrows():
        key = (row["date"], str(row["project_id"]), str(row["model"]))
        if key in usage_key_set:
            continue
        extras.append(
            {
                "timestamp": row["date"],
                "provider": "anthropic",
                "model": row["model"],
                "project_id": row["project_id"],
                "calls": 0.0,
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "total_tokens": 0.0,
                "cost_usd": float(row["amount"]),
                "currency": "usd",
                "cost_source": "reported_allocated",
            }
        )

    if not extras:
        return pd.DataFrame(columns=usage_df.columns)
    return pd.DataFrame(extras)


def _cost_only_unified(cost_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        cost_df.copy()
        .assign(timestamp=lambda d: pd.to_datetime(d["bucket_start"], utc=True).dt.floor("D"))
        .groupby(["timestamp", "project_id", "model"], as_index=False)
        .agg(cost_usd=("amount", "sum"))
    )
    grouped["provider"] = "anthropic"
    grouped["calls"] = 0.0
    grouped["input_tokens"] = 0.0
    grouped["output_tokens"] = 0.0
    grouped["total_tokens"] = 0.0
    grouped["currency"] = "usd"
    grouped["cost_source"] = "reported_allocated"
    return grouped[
        [
            "timestamp",
            "provider",
            "model",
            "project_id",
            "calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "currency",
            "cost_source",
        ]
    ].sort_values("timestamp")


def _get_paginated(
    *,
    session: requests.Session,
    base_url: str,
    path: str,
    params: dict[str, Any],
    timeout_seconds: int,
    max_retries: int,
    max_pages: int = MAX_PAGES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_page: str | None = None
    page_count = 0

    while page_count < max_pages:
        query = dict(params)
        if next_page:
            query["page"] = next_page

        response_payload = _get_json(
            session=session,
            url=f"{base_url}/{path.lstrip('/')}",
            params=query,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        data = response_payload.get("data", [])
        if isinstance(data, list):
            rows.extend([item for item in data if isinstance(item, dict)])

        next_page = str(
            response_payload.get("next_page")
            or response_payload.get("next")
            or response_payload.get("next_cursor")
        )
        if next_page in {"None", ""}:
            next_page = None
        has_more = bool(response_payload.get("has_more")) or bool(next_page)
        page_count += 1

        if not has_more or not next_page:
            return rows

    raise AnthropicAPIError(f"Pagination exceeded {max_pages} pages. Narrow date range and retry.")


def _get_json(
    *,
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout_seconds)
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise AnthropicAPIError(f"Request failed: {exc}") from exc
            time.sleep(0.5 * (2**attempt))
            continue

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            time.sleep(0.5 * (2**attempt))
            continue

        if response.status_code >= 400:
            raise AnthropicAPIError(_error_message(response), status_code=response.status_code)

        try:
            payload = response.json()
        except ValueError as exc:
            raise AnthropicAPIError("Anthropic API returned non-JSON response") from exc
        if not isinstance(payload, dict):
            raise AnthropicAPIError("Unexpected Anthropic payload: root is not an object")
        return payload

    raise AnthropicAPIError("Request failed after retries")


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        err = payload.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(payload.get("message"), str):
            return str(payload["message"])
    except ValueError:
        pass
    return response.text.strip() or "Anthropic API request failed"


def _anthropic_usage_group_dimensions(group_by: tuple[str, ...]) -> list[str]:
    mapping = {
        "model": "model",
        "project_id": "workspace_id",
        "api_key_id": "api_key_id",
    }
    mapped = [mapping[key] for key in group_by if key in mapping]
    if "model" not in mapped:
        mapped.append("model")
    if "workspace_id" not in mapped:
        mapped.append("workspace_id")
    return mapped


def _anthropic_bucket_width(bucket_width: str) -> str:
    return "1h" if bucket_width == "1h" else "1d"


def _anthropic_limit(bucket_width: str) -> int:
    if bucket_width == "1h":
        return 168
    return 31


def _extract_input_tokens(record: dict[str, Any]) -> float:
    if record.get("input_tokens") is not None:
        return _to_float(record.get("input_tokens"))
    cache_creation = record.get("cache_creation")
    cache_creation_1h = 0.0
    cache_creation_5m = 0.0
    if isinstance(cache_creation, dict):
        cache_creation_1h = _to_float(cache_creation.get("ephemeral_1h_input_tokens"))
        cache_creation_5m = _to_float(cache_creation.get("ephemeral_5m_input_tokens"))
    return (
        _to_float(record.get("uncached_input_tokens"))
        + _to_float(record.get("cache_read_input_tokens"))
        + _to_float(record.get("cache_creation_input_tokens"))
        + _to_float(record.get("ephemeral_1h_input_tokens"))
        + _to_float(record.get("ephemeral_5m_input_tokens"))
        + cache_creation_1h
        + cache_creation_5m
    )


def _extract_cost_usd(record: dict[str, Any]) -> float:
    # Anthropic cost report `amount` is in cents.
    if record.get("amount") is not None:
        amount = record.get("amount")
        if isinstance(amount, dict):
            value = amount.get("value")
            return _to_float(value) / 100.0
        return _to_float(amount) / 100.0

    if record.get("cost") is not None:
        return _to_float(record.get("cost"))
    if record.get("cost_usd") is not None:
        return _to_float(record.get("cost_usd"))
    if record.get("amount_usd") is not None:
        return _to_float(record.get("amount_usd"))
    return 0.0


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    if isinstance(value, (int, float)):
        return pd.to_datetime(int(value), unit="s", utc=True)
    try:
        parsed = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return pd.NaT
    return parsed


def _fallback_timestamp(value: pd.Timestamp, fallback: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(value):
        return fallback
    return value


def _to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _unix_to_iso8601(unix_seconds: int) -> str:
    return datetime.fromtimestamp(unix_seconds, tz=UTC).isoformat().replace("+00:00", "Z")
