"""Glue between the nightly/check-revert CLI and per-caller learning (keeps cli.py small)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pinball.config import Config
from pinball.optimize.apply import State, next_version, overlay_doc, save_state, write_overlay
from pinball.optimize.caller_settings import CallerSettings
from pinball.optimize.caller_state import (
    check_callers,
    load_callers,
    record_changes,
    save_callers,
    stale,
    state_path,
    touch_seen,
)
from pinball.optimize.callers import (
    LearnInput,
    Policy,
    eligible_callers,
    learn_caller,
    plan_profiles,
)
from pinball.optimize.db import ReplayDb
from pinball.optimize.replay import PairResult
from pinball.optimize.sample import Sample
from pinball.optimize.settings import RevertSettings
from pinball.overlay import CALLER_KEY_MAX, CallerProfile, Overlay, merge


@dataclass(frozen=True, slots=True)
class NightInput:
    pool: list[dict[str, Any]]
    results: list[PairResult]
    samples: list[Sample]  # global + per-caller
    global_ids: frozenset[str]
    order: dict[str, tuple[str, ...]]  # tonight's global order


@dataclass(frozen=True, slots=True)
class NightOutcome:
    profiles: dict[str, CallerProfile]
    changed: bool
    audit: dict[str, Any]
    state: dict[str, dict[str, Any]]


def learn_night(
    live: Overlay | None,
    night: NightInput,
    base: Config,
    ctx: tuple[Policy, Path | None, datetime],
) -> NightOutcome:
    policy, overlay_path, now = ctx
    cs = policy.callers
    current = {} if live is None else dict(live.callers)
    state = load_callers(state_path(overlay_path))
    eligible = eligible_callers(night.pool, cs)
    seen = {str(r.get("caller") or "")[:CALLER_KEY_MAX] for r in night.pool}
    touch_seen(state, seen & (set(current) | set(eligible)), set(current), now)
    gone = stale(state, set(current), now, cs)
    data = LearnInput(night.results, night.samples, night.global_ids, night.order)
    learned = [learn_caller(c, data, policy) for c in eligible]
    plan = plan_profiles(current, learned, gone, base, night.order, cs)
    for c in gone:
        state.pop(c, None)
    record_changes(state, {c: p for c, p in plan.changed.items() if c in plan.profiles}, now)
    return NightOutcome(plan.profiles, bool(plan.changed), plan.audit, state)


def with_callers(base: Config, overlay: Overlay, profiles: dict[str, CallerProfile]) -> Overlay:
    """`overlay` with its caller map replaced; validated by merge() (OverlayRejected)."""
    out = overlay.model_copy(update={"callers": profiles})
    merge(base, out)
    return out


def save_night(overlay_path: Path | None, outcome: NightOutcome) -> None:
    save_callers(state_path(overlay_path), outcome.state)


@dataclass(frozen=True, slots=True)
class RevertCtx:
    base: Config
    overlay_path: Path
    state_file: Path | None
    margins: tuple[RevertSettings, CallerSettings]


def revert_callers(
    db: ReplayDb, live: Overlay, state: State, now: datetime, ctx: RevertCtx
) -> dict[str, Any]:
    """Revert regressing caller profiles only; writes a new overlay version when any did."""
    cstate = load_callers(state_path(ctx.overlay_path))
    outcome = check_callers(db, live, cstate, now, ctx.margins)
    entry: dict[str, Any] = {"callers": outcome.audit, "callers_reverted": list(outcome.reverted)}
    if not outcome.reverted:
        return entry
    seq, version = next_version(state, live, now)
    overlay = with_callers(ctx.base, live, outcome.callers).model_copy(update={"version": version})
    write_overlay(ctx.overlay_path, ctx.base, overlay)
    save_state(
        ctx.state_file,
        State(
            seq=seq,
            current=overlay_doc(overlay),
            previous=state.previous,
            changed_at=state.changed_at,
        ),
    )
    save_callers(state_path(ctx.overlay_path), cstate)
    entry["version"] = version
    return entry
