from datetime import date

from src.config import COSTS_ENDPOINT, USAGE_COMPLETIONS_ENDPOINT
from src.fetchers import CostQuery, UsageQuery, build_time_window, fetch_cost_buckets, fetch_usage_buckets


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def paginate(self, path: str, *, params: dict | None = None):
        self.calls.append((path, params or {}))
        return iter([{"start_time": 1, "end_time": 2, "results": []}])


def test_build_time_window_uses_inclusive_end_date() -> None:
    start, end = build_time_window(date(2026, 1, 1), date(2026, 1, 1))
    assert end - start == 86400


def test_fetch_usage_and_cost_buckets_delegate_to_client() -> None:
    client = FakeClient()

    usage_query = UsageQuery(start_time=1, end_time=2, bucket_width="1d", group_by=["model"])
    cost_query = CostQuery(start_time=1, end_time=2, bucket_width="1d")

    usage_rows = fetch_usage_buckets(client, usage_query)
    cost_rows = fetch_cost_buckets(client, cost_query)

    assert len(usage_rows) == 1
    assert len(cost_rows) == 1
    assert client.calls[0][0] == USAGE_COMPLETIONS_ENDPOINT
    assert client.calls[1][0] == COSTS_ENDPOINT
