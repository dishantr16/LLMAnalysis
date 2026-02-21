"""Data fetch orchestration for usage and cost endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from src.config import (
    COSTS_ENDPOINT,
    DEFAULT_BUCKET_WIDTH,
    USAGE_COMPLETIONS_ENDPOINT,
)
from src.openai_client import OpenAIAdminClient


@dataclass(frozen=True)
class UsageQuery:
    start_time: int
    end_time: int
    bucket_width: str = DEFAULT_BUCKET_WIDTH
    group_by: list[str] = field(default_factory=lambda: ["model"])


@dataclass(frozen=True)
class CostQuery:
    start_time: int
    end_time: int
    bucket_width: str = DEFAULT_BUCKET_WIDTH


def build_time_window(date_from: date, date_to: date) -> tuple[int, int]:
    """Convert inclusive date range to Unix timestamps (UTC)."""
    if date_from > date_to:
        raise ValueError("Start date must be before or equal to end date.")

    start_dt = datetime.combine(date_from, time.min, tzinfo=timezone.utc)
    end_dt_exclusive = datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return int(start_dt.timestamp()), int(end_dt_exclusive.timestamp())


def fetch_usage_buckets(
    client: OpenAIAdminClient,
    query: UsageQuery,
    *,
    endpoint: str = USAGE_COMPLETIONS_ENDPOINT,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "start_time": query.start_time,
        "end_time": query.end_time,
        "bucket_width": query.bucket_width,
    }
    if query.group_by:
        params["group_by"] = query.group_by

    return list(client.paginate(endpoint, params=params))


def fetch_cost_buckets(client: OpenAIAdminClient, query: CostQuery) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "start_time": query.start_time,
        "end_time": query.end_time,
        "bucket_width": query.bucket_width,
    }
    return list(client.paginate(COSTS_ENDPOINT, params=params))
