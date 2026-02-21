from src.transformers import build_cost_df, build_generic_usage_df, build_usage_df


def test_build_usage_df_parses_expected_fields() -> None:
    usage_buckets = [
        {
            "start_time": 1704067200,
            "end_time": 1704153600,
            "results": [
                {
                    "model": "gpt-4o",
                    "project_id": "proj_a",
                    "user_id": "user_1",
                    "api_key_id": "key_1",
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "input_cached_tokens": 200,
                    "num_model_requests": 8,
                },
                {
                    "model": "gpt-4o-mini",
                    "project_id": "proj_b",
                    "input_tokens": 400,
                    "output_tokens": 600,
                    "num_model_requests": 4,
                },
            ],
        }
    ]

    df = build_usage_df(usage_buckets)

    assert not df.empty
    assert set(["model", "input_tokens", "output_tokens", "requests", "total_tokens"]).issubset(df.columns)
    assert df["input_tokens"].sum() == 1400
    assert df["output_tokens"].sum() == 1100
    assert df["requests"].sum() == 12
    assert df["total_tokens"].sum() == 2500


def test_build_cost_df_parses_amount_currency() -> None:
    cost_buckets = [
        {
            "start_time": 1704067200,
            "end_time": 1704153600,
            "results": [
                {
                    "project_id": "proj_a",
                    "line_item": "completions",
                    "amount": {"value": 3.25, "currency": "usd"},
                },
                {
                    "project_id": "proj_b",
                    "line_item": "images",
                    "amount": {"value": 1.75, "currency": "usd"},
                },
            ],
        }
    ]

    df = build_cost_df(cost_buckets)

    assert not df.empty
    assert list(df["currency"].unique()) == ["usd"]
    assert df["amount"].sum() == 5.0


def test_build_generic_usage_df_extracts_numeric_metrics() -> None:
    buckets = [
        {
            "start_time": 1704067200,
            "end_time": 1704153600,
            "results": [
                {
                    "project_id": "proj_a",
                    "num_requests": 10,
                    "num_images": 24,
                    "size": "1024x1024",
                }
            ],
        }
    ]

    df = build_generic_usage_df("images", buckets)

    assert not df.empty
    assert df["dataset"].iloc[0] == "images"
    assert df["num_requests"].sum() == 10
    assert df["num_images"].sum() == 24
