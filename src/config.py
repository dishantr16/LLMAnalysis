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

ENV_OPENAI_ADMIN_KEY = "OPENAI_ADMIN_KEY"
ENV_ANTHROPIC_ADMIN_KEY = "ANTHROPIC_ADMIN_KEY"
ENV_GROQ_API_KEY = "GROQ_API_KEY"

# USD price per 1M tokens. Keep updated manually from official pricing.
MODEL_PRICING_PER_MILLION = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
}

# USD price per 1M tokens for common Groq models.
# Keep this map updated from https://console.groq.com/docs/models.
GROQ_MODEL_PRICING_PER_MILLION = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "qwen/qwen3-32b": {"input": 0.29, "output": 0.59},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.20, "output": 0.60},
}

DIMENSION_OPTIONS = ["model", "project_id", "user_id", "api_key_id"]
