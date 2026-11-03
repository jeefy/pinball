"""Per-caller state, staleness, nightly integration and per-caller revert.

Caller state (JSON, `<overlay>.callers.json` by default) holds per caller:
  last_seen   last night the caller appeared in the sampled pool (staleness)
  changed_at  when its profile was last written by the optimizer (revert window)
  previous    the profile live before that write (None: no profile)
Only caller keys (<= 64 chars), timestamps and profile numbers are stored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pinball.optimize.apply import atomic_write
from pinball.optimize.caller_settings import CallerSettings
from pinball.optimize.db import ReplayDb
from pinball.optimize.revert import judge_windows, windows
from pinball.optimize.settings import RevertSettings
from pinball.overlay import CALLER_KEY_MAX, CallerProfile, Overlay


def state_path(overlay_path: Path | None) -> Path | None:
    return (
        None
        if overlay_path is None
        else overlay_path.with_name(overlay_path.name + ".callers.json")
    )


def load_callers(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k)[:CALLER_KEY_MAX]: dict(v) for k, v in raw.items()}


def save_callers(path: Path | None, state: dict[str, dict[str, Any]]) -> None:
    if path is not None:
        atomic_write(path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def touch_seen(
    state: dict[str, dict[str, Any]], seen: set[str], live: set[str], now: datetime
) -> None:
    """Record tonight's callers; a live profile with no record starts its clock now."""
    for c in seen:
        state.setdefault(c, {})["last_seen"] = now.isoformat()
    for c in live:
        state.setdefault(c, {}).setdefault("last_seen", now.isoformat())


def stale(
    state: dict[str, dict[str, Any]], live: set[str], now: datetime, cs: CallerSettings
) -> set[str]:
    cutoff = now - timedelta(days=cs.stale_days)
    return {c for c in live if datetime.fromisoformat(state[c]["last_seen"]) < cutoff}


def record_changes(
    state: dict[str, dict[str, Any]], changed: dict[str, CallerProfile | None], now: datetime
) -> None:
    for c, prev in changed.items():
        entry = state.setdefault(c, {})
        entry["changed_at"] = now.isoformat()
        entry["previous"] = None if prev is None else prev.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class CallerRevert:
    callers: dict[str, CallerProfile]  # new caller map
    reverted: tuple[str, ...]
    audit: dict[str, Any]


def check_callers(
    db: ReplayDb,
    live: Overlay,
    state: dict[str, dict[str, Any]],
    now: datetime,
    margins: tuple[RevertSettings, CallerSettings],
) -> CallerRevert:
    """Per caller: same error/cost margins as the global check, on that caller's traffic."""
    rs, cs = margins
    per_caller = rs.model_copy(update={"min_requests": cs.revert_min_requests})
    callers = dict(live.callers)
    reverted: list[str] = []
    audit: dict[str, Any] = {}
    for c in sorted(callers):
        changed_at = state.get(c, {}).get("changed_at")
        if changed_at is None:
            continue
        (b0, b1), (a0, a1) = windows(datetime.fromisoformat(changed_at), now, rs.hours)
        decision = judge_windows(db.window(b0, b1, c), db.window(a0, a1, c), per_caller)
        audit[c] = decision.record()
        if decision.revert:
            prev = state[c].get("previous")
            if prev is None:
                del callers[c]
            else:
                callers[c] = CallerProfile.model_validate(prev)
            state[c].pop("changed_at", None)
            state[c].pop("previous", None)
            reverted.append(c)
    return CallerRevert(callers, tuple(reverted), audit)
