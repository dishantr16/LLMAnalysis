"""OpenAI provider adapter implementation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import MODEL_PRICING_PER_MILLION, USAGE_ENDPOINTS
from src.fetchers import CostQuery, UsageQuery, fetch_cost_buckets, fetch_usage_buckets
from src.openai_client import OpenAIAPIError, OpenAIAdminClient
from src.providers.base import ProviderAdapter, ProviderFetchResult, empty_unified_df
from src.transformers import build_cost_df, build_generic_usage_df, build_usage_df


class OpenAIProviderAdapter(ProviderAdapter):
    provider_name = "openai"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()

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
                notices=["OpenAI key not configured."],
            )

        usage_query = UsageQuery(
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=list(group_by),
        )
        cost_query = CostQuery(
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
        )

        usage_bucket_map: dict[str, list[dict[str, Any]]] = {}
        endpoint_errors: dict[str, str] = {}
        project_rows: list[dict[str, Any]] = []
        project_rate_limit_rows: list[dict[str, Any]] = []
        project_rate_limit_errors: list[str] = []

        with OpenAIAdminClient(api_key=self.api_key) as client:
            for dataset, endpoint in USAGE_ENDPOINTS.items():
                try:
                    usage_bucket_map[dataset] = fetch_usage_buckets(
                        client,
                        usage_query,
                        endpoint=endpoint,
                    )
                except OpenAIAPIError as exc:
                    usage_bucket_map[dataset] = []
                    endpoint_errors[dataset] = str(exc)

            try:
                cost_buckets = fetch_cost_buckets(client, cost_query)
            except OpenAIAPIError as exc:
                cost_buckets = []
                endpoint_errors["costs"] = str(exc)

            try:
                project_rows = client.list_projects()
            except OpenAIAPIError as exc:
                project_rows = []
                endpoint_errors["openai_project_rate_limits"] = str(exc)
            else:
                for project in project_rows:
                    project_id = str(project.get("id") or "").strip()
                    if not project_id:
                        continue
                    try:
                        limits = client.get_project_rate_limits(project_id)
                    except OpenAIAPIError as exc:
                        project_rate_limit_errors.append(f"{project_id}: {exc}")
                        continue

                    for limit in limits:
                        if not isinstance(limit, dict):
                            continue
                        row = dict(limit)
                        row["project_id"] = project_id
                        row["project_name"] = project.get("name") or project_id
                        project_rate_limit_rows.append(row)

        if project_rate_limit_errors and not project_rate_limit_rows:
            endpoint_errors["openai_project_rate_limits"] = "; ".join(project_rate_limit_errors[:3])

        usage_df = build_usage_df(usage_bucket_map.get("completions", []))
        cost_df = build_cost_df(cost_buckets)

        auxiliary_usage_frames: dict[str, pd.DataFrame] = {}
        for dataset, buckets in usage_bucket_map.items():
            if dataset == "completions":
                continue
            auxiliary_usage_frames[dataset] = build_generic_usage_df(dataset, buckets)

        unified_df = build_openai_unified_df(usage_df, cost_df)
        project_rate_limits_df = build_openai_project_rate_limits_df(project_rate_limit_rows)

        notices: list[str] = []
        if unified_df.empty:
            notices.append("OpenAI returned no model-level usage rows for selected range.")
        if project_rate_limit_errors:
            notices.append(
                "OpenAI rate limits were partially fetched. Some projects returned errors."
            )

        return ProviderFetchResult(
            provider=self.provider_name,
            unified_df=unified_df,
            raw_payload={
                "usage_df": usage_df,
                "cost_df": cost_df,
                "aux_usage_frames": auxiliary_usage_frames,
                "project_rate_limits_df": project_rate_limits_df,
            },
            endpoint_errors=endpoint_errors,
            notices=notices,
        )



def build_openai_unified_df(usage_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    if usage_df.empty:
        return empty_unified_df()

    df = usage_df.copy()
    df["timestamp"] = pd.to_datetime(df["bucket_start"], utc=True)
    df["provider"] = "openai"
    df["project_id"] = df["project_id"].fillna("unknown")
    df["model"] = df["model"].fillna("unknown")
    df["calls"] = pd.to_numeric(df["requests"], errors="coerce").fillna(0)
    df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0)
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0)
    df["total_tokens"] = df["input_tokens"] + df["output_tokens"]

    df["estimated_cost_usd"] = df.apply(_estimate_row_cost_usd, axis=1)
    df["cost_usd"] = _reconcile_daily_model_cost(df, cost_df)
    df["currency"] = "usd"

    has_reported_cost = not cost_df.empty and float(cost_df["amount"].sum()) > 0
    df["cost_source"] = "reconciled_estimate" if has_reported_cost else "estimated"

    return df[
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



def _estimate_row_cost_usd(row: pd.Series) -> float:
    model_key = str(row.get("model", "")).lower()
    pricing = MODEL_PRICING_PER_MILLION.get(model_key)
    if not pricing:
        return 0.0

    input_cost = (float(row.get("input_tokens", 0)) / 1_000_000) * float(pricing.get("input", 0))
    output_cost = (float(row.get("output_tokens", 0)) / 1_000_000) * float(pricing.get("output", 0))
    return input_cost + output_cost



def _reconcile_daily_model_cost(usage_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.Series:
    if usage_df.empty:
        return pd.Series(dtype=float)

    reconciled = pd.Series(0.0, index=usage_df.index)

    daily_reported: dict[pd.Timestamp, float] = {}
    if not cost_df.empty:
        cdf = cost_df.copy()
        cdf["date"] = pd.to_datetime(cdf["bucket_start"], utc=True).dt.floor("D")
        daily_reported = cdf.groupby("date")["amount"].sum().to_dict()

    udf = usage_df.copy()
    udf["date"] = pd.to_datetime(udf["timestamp"], utc=True).dt.floor("D")

    for day, idx in udf.groupby("date").groups.items():
        index_list = list(idx)
        day_estimated = float(udf.loc[index_list, "estimated_cost_usd"].sum())
        day_calls = float(udf.loc[index_list, "calls"].sum())
        day_reported = float(daily_reported.get(day, 0.0))

        if day_reported <= 0:
            reconciled.loc[index_list] = udf.loc[index_list, "estimated_cost_usd"].to_numpy()
            continue

        if day_estimated > 0:
            scale = day_reported / day_estimated
            reconciled.loc[index_list] = udf.loc[index_list, "estimated_cost_usd"].to_numpy() * scale
            continue

        if day_calls > 0:
            proportions = udf.loc[index_list, "calls"].to_numpy() / day_calls
            reconciled.loc[index_list] = proportions * day_reported
        else:
            reconciled.loc[index_list] = day_reported / max(1, len(index_list))

    return reconciled.fillna(0.0)


def build_openai_project_rate_limits_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "provider",
                "project_id",
                "project_name",
                "model",
                "rpm_limit",
                "tpm_limit",
                "rpd_limit",
                "tpd_limit",
            ]
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    if "model" not in df.columns:
        df["model"] = "unknown"
    if "project_id" not in df.columns:
        df["project_id"] = "unknown"
    if "project_name" not in df.columns:
        df["project_name"] = df["project_id"]

    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    result = pd.DataFrame(
        {
            "provider": "openai",
            "project_id": df["project_id"].astype(str),
            "project_name": df["project_name"].astype(str),
            "model": df["model"].astype(str),
            "rpm_limit": _num("max_requests_per_1_minute"),
            "tpm_limit": _num("max_tokens_per_1_minute"),
            "rpd_limit": _num("max_requests_per_1_day"),
            "tpd_limit": _num("max_tokens_per_1_day"),
        }
    )
    return result.sort_values(["project_name", "model"]).reset_index(drop=True)
