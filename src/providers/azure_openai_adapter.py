"""Azure OpenAI provider adapter scaffold."""

from __future__ import annotations

from src.providers.base import ProviderAdapter, ProviderFetchResult, empty_unified_df


class AzureOpenAIProviderAdapter(ProviderAdapter):
    provider_name = "azure_openai"

    def __init__(self, api_key: str, endpoint: str) -> None:
        self.api_key = api_key.strip()
        self.endpoint = endpoint.strip()

    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint)

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
                notices=["Azure OpenAI key/endpoint not configured."],
            )

        return ProviderFetchResult(
            provider=self.provider_name,
            unified_df=empty_unified_df(),
            notices=[
                "Azure OpenAI adapter scaffold is in place. "
                "Azure usage and billing API mapping will be added next."
            ],
        )
