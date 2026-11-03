"""Learned overlay: data written by the tuning job, merged onto the Git base config.

The base config (pinball.yaml) owns the model pool and every guardrail: prompt
windows, vision, thresholds, scoring and which tier a model belongs to. The
overlay may only reorder models inside a tier and shift a caller's thresholds,
so a bad overlay can change preference but never capability. Any overlay that
tries more is rejected whole; the caller keeps its previous state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pinball.config import Config, Tier
from pinball.decide import Decider

# The caller header is truncated to this many characters before lookup.
CALLER_KEY_MAX: Final = 64


class ThresholdOffsets(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    standard: int = 0
    heavy: int = 0


class CallerProfile(BaseModel):
    """Per-caller learned behaviour.

    `weight` is the shrinkage toward global behaviour: the effective threshold
    offset is round(weight * offset), so 0 means "behave like everyone else"
    and 1 means "trust this caller's data fully". The tuning job is expected
    to set it from sample size (e.g. n / (n + k)). `tier_order` is applied as
    written; it is still bounded to a permutation of the base tier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier_order: dict[Tier, tuple[str, ...]] = {}
    threshold_offsets: ThresholdOffsets = ThresholdOffsets()
    weight: float = Field(default=1.0, ge=0, le=1)


class Overlay(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1, max_length=128)
    generated_at: datetime
    tier_order: dict[Tier, tuple[str, ...]] = {}
    callers: dict[str, CallerProfile] = {}


class OverlayRejected(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail

    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Active:
    """One immutable routing state; swapped as a single reference."""

    decider: Decider
    callers: dict[str, Decider]
    version: str

    def select(self, caller: str) -> tuple[Decider, str]:
        """The decider for `caller` and the profile name applied ("" for global)."""
        d = self.callers.get(caller)
        return (self.decider, "") if d is None else (d, caller)


def base_only(base: Config) -> Active:
    return Active(Decider(base), {}, "")


def _reorder(
    base: Config, order: dict[Tier, tuple[str, ...]], where: str
) -> dict[Tier, tuple[str, ...]]:
    tiers = dict(base.tiers)
    for tier, names in order.items():
        allowed = base.tiers[tier]
        for n in names:
            if n not in base.models:
                raise OverlayRejected("unknown_model", f"{where}.{tier}: {n!r} not in base pool")
            if n not in allowed:
                raise OverlayRejected("cross_tier", f"{where}.{tier}: {n!r} not in base tier")
        if len(names) != len(allowed) or set(names) != set(allowed):
            raise OverlayRejected(
                "not_permutation", f"{where}.{tier} must be a permutation of {list(allowed)}"
            )
        tiers[tier] = names
    return tiers


def _validated(
    base: Config, tiers: dict[Tier, tuple[str, ...]], thresholds: dict[str, int], where: str
) -> Config:
    try:
        return Config.model_validate(
            {**base.model_dump(), "tiers": tiers, "thresholds": thresholds}
        )
    except ValidationError as e:
        raise OverlayRejected("guardrail", f"{where}: {e.errors()[0]['msg']}") from e


def merge(base: Config, overlay: Overlay) -> Active:
    """Base + overlay -> Active, or OverlayRejected if any rule is broken."""
    tiers = _reorder(base, overlay.tier_order, "tier_order")
    global_cfg = _validated(base, tiers, base.thresholds.model_dump(), "tier_order")
    callers: dict[str, Decider] = {}
    for name, p in overlay.callers.items():
        where = f"callers.{name}"
        if not name or len(name) > CALLER_KEY_MAX:
            raise OverlayRejected("caller_key", f"{where}: key must be 1-{CALLER_KEY_MAX} chars")
        c_tiers = {**tiers, **_reorder(global_cfg, p.tier_order, f"{where}.tier_order")}
        off = p.threshold_offsets
        thresholds = {
            "standard": base.thresholds.standard + round(p.weight * off.standard),
            "heavy": base.thresholds.heavy + round(p.weight * off.heavy),
        }
        callers[name] = Decider(_validated(base, c_tiers, thresholds, where))
    return Active(Decider(global_cfg), callers, overlay.version)


def load_overlay(path: Path) -> Overlay:
    """Parse the overlay file; OverlayRejected for unreadable or invalid data."""
    try:
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise OverlayRejected("parse", str(e).splitlines()[0]) from e
    try:
        return Overlay.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise OverlayRejected("schema", f"{loc}: {first['msg']}") from e
