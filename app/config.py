"""Environment-based configuration for the application."""

import os

from dotenv import load_dotenv


load_dotenv()


def _optional_float(name: str) -> float | None:
    """Read an optional floating-point environment variable."""
    value = os.getenv(name, "").strip()
    return float(value) if value else None


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "")
INPUT_COST_PER_MILLION_TOKENS = _optional_float(
    "INPUT_COST_PER_MILLION_TOKENS"
)
OUTPUT_COST_PER_MILLION_TOKENS = _optional_float(
    "OUTPUT_COST_PER_MILLION_TOKENS"
)
MANUAL_TASK_BASELINE_MINUTES = int(
    os.getenv("MANUAL_TASK_BASELINE_MINUTES", "15")
)

