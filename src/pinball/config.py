"""pinball.yaml: the model pool, tier preferences and scoring knobs."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Tier(StrEnum):
    LIGHT = "light"
    STANDARD = "standard"
    HEAVY = "heavy"


TIER_ORDER: tuple[Tier, ...] = (Tier.LIGHT, Tier.STANDARD, Tier.HEAVY)


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(gt=0)
    vision: bool = False


class Thresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    standard: int
    heavy: int

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.heavy <= self.standard:
            raise ValueError("thresholds.heavy must be greater than thresholds.standard")
        return self


class Scoring(BaseModel):
    """Points per signal and the size cutoffs; every key is optional."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: int = 1
    heavy_keyword: int = 1
    max_heavy_keywords: int = Field(default=2, ge=0)
    light_keyword: int = -1
    long_chars: int = Field(default=4000, gt=0)
    long: int = 1
    very_long_chars: int = Field(default=12000, gt=0)
    very_long: int = 2
    short_chars: int = Field(default=160, ge=0)
    short: int = -1
    tools: int = 1
    effort_high: int = 2
    effort_low: int = -1
    small_output_tokens: int = Field(default=512, ge=0)
    small_output: int = -1
    structured: int = -1
    # Whole-request size (estimated prompt tokens); unset = off.
    large_context_tokens: int | None = Field(default=None, gt=0)
    large_context: int = 1
    very_large_context_tokens: int | None = Field(default=None, gt=0)
    very_large_context: int = 2

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        small, big = self.large_context_tokens, self.very_large_context_tokens
        if big is not None and (small is None or big <= small):
            raise ValueError(
                "scoring.very_large_context_tokens needs a smaller large_context_tokens"
            )
        if self.very_long_chars <= self.long_chars:
            raise ValueError("scoring.very_long_chars must be greater than scoring.long_chars")
        return self


class WrappedUserMessage(BaseModel):
    """A user-role message that is mostly client scaffolding (e.g. an injected skill file).

    Only the text after the last `instruction_markers` hit (first marker that
    matches wins), cut at `instruction_end`, is scored; with no marker the
    message is not scored at all. It still counts toward prompt size.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefix: str = Field(min_length=1)
    instruction_markers: tuple[str, ...] = ()
    instruction_end: str = ""


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    models: dict[str, ModelSpec] = Field(min_length=1)
    tiers: dict[Tier, tuple[str, ...]]
    thresholds: Thresholds
    scoring: Scoring = Scoring()
    wrapped_user_messages: tuple[WrappedUserMessage, ...] = ()
    headroom: float = Field(default=0.9, gt=0, le=1)
    heavy_keywords: tuple[str, ...] = ()
    light_keywords: tuple[str, ...] = ()
    reply_signals: bool = False  # opt-in: add signal names (no content) to the /check reply

    @model_validator(mode="after")
    def _tiers_reference_pool(self) -> Self:
        for tier in TIER_ORDER:
            names = self.tiers.get(tier)
            if not names:
                raise ValueError(f"tiers.{tier} must list at least one model")
            unknown = [n for n in names if n not in self.models]
            if unknown:
                raise ValueError(f"tiers.{tier} names models not in `models`: {unknown}")
        return self

    def tier_models(self, tier: Tier) -> tuple[str, ...]:
        return self.tiers[tier]


def keyword_pattern(words: tuple[str, ...]) -> re.Pattern[str] | None:
    """One case-insensitive alternation, longest first, on word boundaries."""
    if not words:
        return None
    alts = "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE)


def default_config_path() -> Path:
    """Packaged pinball.yaml (wheel), else the repository root copy."""
    packaged = Path(__file__).with_name("pinball.yaml")
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "pinball.yaml"


def load_config(path: Path | None = None) -> Config:
    with (path or default_config_path()).open(encoding="utf-8") as fh:
        return Config.model_validate(yaml.safe_load(fh))
