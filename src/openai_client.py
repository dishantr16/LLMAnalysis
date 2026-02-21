"""Thin OpenAI admin API client for usage and costs endpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

from src.config import API_BASE_URL, MAX_PAGES, MAX_RETRIES, REQUEST_TIMEOUT_SECONDS

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class OpenAIAPIError(Exception):
    """Represents a failed API request."""

    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"{self.message} (status={self.status_code})"


class OpenAIAdminClient:
    """Minimal client that supports paginated GET calls."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = API_BASE_URL,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        if not api_key:
            raise ValueError("An OpenAI Admin API key is required.")

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "OpenAIAdminClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def paginate(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        max_pages: int = MAX_PAGES,
    ) -> Iterator[dict[str, Any]]:
        """Yield each bucket from a paginated endpoint."""
        page_count = 0
        next_page: str | None = None

        while page_count < max_pages:
            query = dict(params or {})
            if next_page:
                query["page"] = next_page

            payload = self._get(path, params=query)
            data = payload.get("data", [])
            if not isinstance(data, list):
                raise OpenAIAPIError("Unexpected response payload: data is not a list")

            for bucket in data:
                if isinstance(bucket, dict):
                    yield bucket

            next_page = payload.get("next_page")
            page_count += 1
            if not next_page:
                return

        raise OpenAIAPIError(
            f"Pagination exceeded {max_pages} pages. Narrow date range and retry."
        )

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise OpenAIAPIError(f"Request failed: {exc}") from exc
                time.sleep(0.5 * (2**attempt))
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                time.sleep(0.5 * (2**attempt))
                continue

            if response.status_code >= 400:
                raise OpenAIAPIError(
                    message=self._error_message(response),
                    status_code=response.status_code,
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise OpenAIAPIError("OpenAI API returned non-JSON response") from exc

            if not isinstance(payload, dict):
                raise OpenAIAPIError("Unexpected response payload: root is not an object")
            return payload

        raise OpenAIAPIError("Request failed after retries")

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
            err = payload.get("error", {})
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
        except ValueError:
            pass
        return response.text.strip() or "OpenAI API request failed"
