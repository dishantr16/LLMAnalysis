"""Provider adapter interfaces and unified schema definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

UNIFIED_SCHEMA_COLUMNS = [
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


@dataclass
class ProviderFetchResult:
    provider: str
    unified_df: pd.DataFrame
    raw_payload: dict[str, Any] = field(default_factory=dict)
    endpoint_errors: dict[str, str] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)


class ProviderAdapter(ABC):
    """Abstract interface that all LLM provider adapters must implement."""

    provider_name: str

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if adapter has enough credentials to attempt fetching data."""

    @abstractmethod
    def fetch(
        self,
        *,
        start_time: int,
        end_time: int,
        bucket_width: str,
        group_by: tuple[str, ...],
    ) -> ProviderFetchResult:
        """Fetch provider data and return normalized unified frame + provider raw payload."""



def empty_unified_df() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIFIED_SCHEMA_COLUMNS)
