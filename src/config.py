"""Application configuration for the LLM Cost Analysis POC."""

from __future__ import annotations

API_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_API_BASE_URL = "https://api.anthropic.com/v1"
GROQ_METRICS_BASE_URL = "https://api.groq.com/v1/metrics/prometheus"
ANTHROPIC_VERSION = "2023-06-01"
USAGE_COMPLETIONS_ENDPOINT = "/organization/usage/completions"
USAGE_IMAGES_ENDPOINT = "/organization/usage/images"
USAGE_MODERATIONS_ENDPOINT = "/organization/usage/moderations"
USAGE_AUDIO_SPEECHES_ENDPOINT = "/organization/usage/audio_speeches"
USAGE_AUDIO_TRANSCRIPTIONS_ENDPOINT = "/organization/usage/audio_transcriptions"
USAGE_VECTOR_STORES_ENDPOINT = "/organization/usage/vector_stores"
USAGE_CODE_INTERPRETER_ENDPOINT = "/organization/usage/code_interpreter_sessions"
COSTS_ENDPOINT = "/organization/costs"
OPENAI_PROJECTS_ENDPOINT = "/organization/projects"
OPENAI_PROJECT_RATE_LIMITS_ENDPOINT_TEMPLATE = "/organization/projects/{project_id}/rate_limits"
ANTHROPIC_USAGE_REPORT_MESSAGES_ENDPOINT = "/organizations/usage_report/messages"
ANTHROPIC_COST_REPORT_ENDPOINT = "/organizations/cost_report"

USAGE_ENDPOINTS = {
    "completions": USAGE_COMPLETIONS_ENDPOINT,
    "images": USAGE_IMAGES_ENDPOINT,
    "moderations": USAGE_MODERATIONS_ENDPOINT,
    "audio_speeches": USAGE_AUDIO_SPEECHES_ENDPOINT,
    "audio_transcriptions": USAGE_AUDIO_TRANSCRIPTIONS_ENDPOINT,
    "vector_stores": USAGE_VECTOR_STORES_ENDPOINT,
    "code_interpreter_sessions": USAGE_CODE_INTERPRETER_ENDPOINT,
}

USAGE_ENDPOINT_LABELS = {
    "completions": "Completions",
    "images": "Images",
    "moderations": "Moderations",
    "audio_speeches": "Audio Speech",
    "audio_transcriptions": "Audio Transcription",
    "vector_stores": "Vector Stores",
    "code_interpreter_sessions": "Code Interpreter Sessions",
    "openai_project_rate_limits": "OpenAI Project Rate Limits",
    "usage_report_messages": "Usage Report Messages",
    "cost_report": "Cost Report",
}

DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_BUCKET_WIDTH = "1d"
CACHE_TTL_SECONDS = 300
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
MAX_PAGES = 100
DUMMY_GPU_UTILIZATION_PCT = 62.0

ENV_OPENAI_ADMIN_KEY = "OPENAI_ADMIN_KEY"
ENV_ANTHROPIC_ADMIN_KEY = "ANTHROPIC_ADMIN_KEY"
ENV_GROQ_API_KEY = "GROQ_API_KEY"

# ---------------------------------------------------------------------------
# Legacy pricing maps — kept for backward compatibility.
# The authoritative pricing source is now src.pricing_registry.PricingRegistry.
# These maps are auto-populated from the registry at import time.
# ---------------------------------------------------------------------------
from src.pricing_registry import get_pricing_registry as _get_registry  # noqa: E402

def _build_legacy_pricing_map(provider: str) -> dict[str, dict[str, float]]:
    return _get_registry().export_as_legacy_map(provider)

MODEL_PRICING_PER_MILLION: dict[str, dict[str, float]] = _build_legacy_pricing_map("openai")
GROQ_MODEL_PRICING_PER_MILLION: dict[str, dict[str, float]] = _build_legacy_pricing_map("groq")

DIMENSION_OPTIONS = ["model", "project_id", "user_id", "api_key_id"]
