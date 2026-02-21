import pandas as pd

from src.providers.openai_adapter import build_openai_project_rate_limits_df


def test_build_openai_project_rate_limits_df() -> None:
    rows = [
        {
            "project_id": "proj_1",
            "project_name": "Project 1",
            "model": "gpt-4o",
            "max_requests_per_1_minute": 300,
            "max_tokens_per_1_minute": 6000,
            "max_requests_per_1_day": 100000,
            "max_tokens_per_1_day": 2000000,
        }
    ]
    df = build_openai_project_rate_limits_df(rows)
    assert not df.empty
    assert set(
        ["provider", "project_id", "project_name", "model", "rpm_limit", "tpm_limit", "rpd_limit", "tpd_limit"]
    ).issubset(df.columns)
    row = df.iloc[0]
    assert row["provider"] == "openai"
    assert row["rpm_limit"] == 300
    assert row["tpm_limit"] == 6000
