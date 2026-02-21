"""Groq provider adapter implementation via metrics API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from src.config import (
    GROQ_METRICS_BASE_URL,
    GROQ_MODEL_PRICING_PER_MILLION,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
)
from src.providers.base import ProviderAdapter, ProviderFetchResult, empty_unified_df

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

GROQ_METRIC_QUERIES = {
    "calls": [
        "model_project_id_status_code:requests:rate5m",
        "model_project_id:requests:rate5m",
        "requests:rate5m",
    ],
    "input_tokens": ["model_project_id:tokens_in:rate5m", "tokens_in:rate5m"],
    "output_tokens": ["model_project_id:tokens_out:rate5m", "tokens_out:rate5m"],
}


@dataclass
class GroqAPIError(Exception):
    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"{self.message} (status={self.status_code})"


class GroqProviderAdapter(ProviderAdapter):
    provider_name = "groq"

    def __init__(
        self,
        api_key: str,
        *,
        metrics_base_url: str = GROQ_METRICS_BASE_URL,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.api_key = api_key.strip()
        self.metrics_base_url = metrics_base_url.rstrip("/")
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
        del group_by  # Groq metrics labels drive grouping.

        if not self.is_configured():
            return ProviderFetchResult(
                provider=self.provider_name,
                unified_df=empty_unified_df(),
                notices=["Groq key not configured."],
            )

        step = "1h" if bucket_width == "1h" else "1d"
        step_seconds = 3600 if step == "1h" else 86400

        endpoint_errors: dict[str, str] = {}
        notices: list[str] = []
        metric_frames: dict[str, pd.DataFrame] = {}
        metric_query_used: dict[str, str] = {}

        with requests.Session() as session:
            session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            for metric_name, query_candidates in GROQ_METRIC_QUERIES.items():
                frame, used_query, error = _fetch_metric_with_fallback(
                    session=session,
                    base_url=self.metrics_base_url,
                    query_candidates=query_candidates,
                    start_time=start_time,
                    end_time=end_time,
                    step=step,
                    step_seconds=step_seconds,
                    output_column=metric_name,
                    timeout_seconds=self.timeout_seconds,
                    max_retries=self.max_retries,
                )
                metric_frames[metric_name] = frame
                if used_query:
                    metric_query_used[metric_name] = used_query
                if error:
                    endpoint_errors[metric_name] = error

        unified_df = build_groq_unified_df(metric_frames)

        if unified_df.empty:
            notices.append(
                "Groq returned no metrics rows. Verify metrics entitlement/API access in your Groq org."
            )
            if any("status=404" in str(msg) for msg in endpoint_errors.values()):
                notices.append(
                    "Groq metrics endpoint returned 404. This usually means Prometheus Metrics is not "
                    "enabled for your org (Enterprise feature) or the metrics base URL is incorrect."
                )
        else:
            missing_pricing_models = _models_without_pricing(unified_df)
            if missing_pricing_models:
                sample_models = ", ".join(missing_pricing_models[:5])
                notices.append(
                    "Cost for some Groq models is set to 0 because they are missing in the local pricing map: "
                    f"{sample_models}."
                )
            notices.append(
                "Groq billing totals are estimated from token metrics using the local model pricing map."
            )

        return ProviderFetchResult(
            provider=self.provider_name,
            unified_df=unified_df,
            raw_payload={
                "metric_frames": metric_frames,
                "metric_query_used": metric_query_used,
            },
            endpoint_errors=endpoint_errors,
            notices=notices,
        )


def build_groq_unified_df(metric_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ["calls", "input_tokens", "output_tokens"]
    available_frames = [metric_frames[key] for key in keys if key in metric_frames and not metric_frames[key].empty]
    if not available_frames:
        return empty_unified_df()

    merged = available_frames[0].copy()
    for frame in available_frames[1:]:
        merged = merged.merge(frame, on=["timestamp", "model", "project_id"], how="outer")

    for metric_name in keys:
        if metric_name not in merged.columns:
            merged[metric_name] = 0.0
        merged[metric_name] = pd.to_numeric(merged[metric_name], errors="coerce").fillna(0.0)

    merged["provider"] = "groq"
    merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
    merged["cost_usd"] = merged.apply(_estimate_groq_cost, axis=1)
    merged["currency"] = "usd"
    merged["cost_source"] = "estimated"

    return merged[
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


def _fetch_metric_with_fallback(
    *,
    session: requests.Session,
    base_url: str,
    query_candidates: list[str],
    start_time: int,
    end_time: int,
    step: str,
    step_seconds: int,
    output_column: str,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[pd.DataFrame, str | None, str | None]:
    last_error: str | None = None

    for query in query_candidates:
        try:
            payload = _query_range(
                session=session,
                base_url=base_url,
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=step,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        except GroqAPIError as exc:
            last_error = str(exc)
            continue

        frame = _prometheus_matrix_to_frame(
            payload=payload,
            output_column=output_column,
            step_seconds=step_seconds,
        )
        if not frame.empty:
            return frame, query, None

    return pd.DataFrame(columns=["timestamp", "model", "project_id", output_column]), None, last_error


def _query_range(
    *,
    session: requests.Session,
    base_url: str,
    query: str,
    start_time: int,
    end_time: int,
    step: str,
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    params = {"query": query, "start": start_time, "end": end_time, "step": step}
    candidate_urls = _build_query_range_urls(base_url)
    last_not_found: GroqAPIError | None = None

    for url in candidate_urls:
        for attempt in range(max_retries + 1):
            try:
                response = session.get(url, params=params, timeout=timeout_seconds)
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise GroqAPIError(f"Request failed: {exc}") from exc
                time.sleep(0.5 * (2**attempt))
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                time.sleep(0.5 * (2**attempt))
                continue

            if response.status_code == 404:
                last_not_found = GroqAPIError(_error_message(response), status_code=response.status_code)
                break

            if response.status_code >= 400:
                raise GroqAPIError(_error_message(response), status_code=response.status_code)

            try:
                payload = response.json()
            except ValueError as exc:
                raise GroqAPIError("Groq metrics endpoint returned non-JSON response") from exc

            if not isinstance(payload, dict):
                raise GroqAPIError("Unexpected Groq payload: root is not an object")
            status = payload.get("status")
            if status != "success":
                raise GroqAPIError(f"Groq query failed for `{query}`: {payload}")
            return payload

    if last_not_found is not None:
        raise last_not_found
    raise GroqAPIError("Request failed after retries")


def _build_query_range_urls(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    candidates: list[str] = []
    if base.endswith("/api/v1"):
        candidates.append(f"{base}/query_range")
    else:
        candidates.append(f"{base}/api/v1/query_range")
        candidates.append(f"{base}/query_range")
    return candidates


def _prometheus_matrix_to_frame(
    *,
    payload: dict[str, Any],
    output_column: str,
    step_seconds: int,
) -> pd.DataFrame:
    data = payload.get("data", {})
    result = data.get("result", []) if isinstance(data, dict) else []
    if not isinstance(result, list):
        return pd.DataFrame(columns=["timestamp", "model", "project_id", output_column])

    rows: list[dict[str, Any]] = []
    for series in result:
        if not isinstance(series, dict):
            continue
        metric = series.get("metric", {})
        if not isinstance(metric, dict):
            metric = {}

        model = (
            metric.get("model")
            or metric.get("model_id")
            or metric.get("model_name")
            or metric.get("model_project_id")
            or "unknown"
        )
        project = (
            metric.get("project_id")
            or metric.get("project")
            or metric.get("workspace_id")
            or metric.get("organization_id")
            or "unknown"
        )

        values = series.get("values", [])
        if not isinstance(values, list):
            continue

        for pair in values:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            try:
                ts = float(pair[0])
                rate_value = float(pair[1])
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "timestamp": pd.to_datetime(ts, unit="s", utc=True),
                    "model": str(model),
                    "project_id": str(project),
                    output_column: max(0.0, rate_value) * step_seconds,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "model", "project_id", output_column])

    return (
        pd.DataFrame(rows)
        .groupby(["timestamp", "model", "project_id"], as_index=False)
        .agg(**{output_column: (output_column, "sum")})
        .sort_values("timestamp")
    )


def _estimate_groq_cost(row: pd.Series) -> float:
    model_key = str(row.get("model", "")).lower()
    pricing = GROQ_MODEL_PRICING_PER_MILLION.get(model_key)
    if not pricing:
        return 0.0

    input_tokens = float(row.get("input_tokens", 0.0))
    output_tokens = float(row.get("output_tokens", 0.0))
    return (input_tokens / 1_000_000) * float(pricing["input"]) + (
        output_tokens / 1_000_000
    ) * float(pricing["output"])


def _models_without_pricing(unified_df: pd.DataFrame) -> list[str]:
    if unified_df.empty:
        return []
    frame = unified_df.copy()
    frame["model_key"] = frame["model"].astype(str).str.lower()
    missing = frame[
        (frame["total_tokens"] > 0)
        & (~frame["model_key"].isin(set(GROQ_MODEL_PRICING_PER_MILLION.keys())))
    ]["model"].dropna()
    return sorted(missing.astype(str).unique().tolist())


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, str) and err:
                return err
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if isinstance(payload.get("message"), str):
                return str(payload["message"])
    except ValueError:
        pass
    return response.text.strip() or "Groq API request failed"
