"""Per-caller learning settings (PINBALL_OPT_CALLER_*), kept apart from settings.py."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class CallerSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_samples: int = Field(default=30, ge=1)  # judged non-agentic samples per caller
    per_tier: int = Field(default=15, ge=0)  # extra per-caller samples per tier and night
    shrink_k: float = Field(default=30.0, ge=0)  # weight = n / (n + shrink_k)
    max_profiles: int = Field(default=20, ge=0)
    max_changes: int = Field(default=5, ge=0)  # profiles added or modified per night
    stale_days: float = Field(default=14.0, gt=0)
    # Light-tier served replies judged unacceptable this much more often than the
    # global light rate lower the caller's standard threshold by one point.
    light_fail_gap: float = Field(default=0.15, ge=0, le=1)
    revert_min_requests: int = Field(default=20, ge=1)  # per caller and window


_ENV: Final = {
    "PINBALL_OPT_CALLER_MIN_SAMPLES": "min_samples",
    "PINBALL_OPT_CALLER_PER_TIER": "per_tier",
    "PINBALL_OPT_CALLER_SHRINK_K": "shrink_k",
    "PINBALL_OPT_CALLER_MAX_PROFILES": "max_profiles",
    "PINBALL_OPT_CALLER_MAX_CHANGES": "max_changes",
    "PINBALL_OPT_CALLER_STALE_DAYS": "stale_days",
    "PINBALL_OPT_CALLER_LIGHT_FAIL_GAP": "light_fail_gap",
    "PINBALL_OPT_CALLER_REVERT_MIN_REQUESTS": "revert_min_requests",
}


def caller_settings_from_env(env: Mapping[str, str]) -> CallerSettings:
    return CallerSettings.model_validate({f: env[v] for v, f in _ENV.items() if env.get(v)})
