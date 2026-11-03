"""Overlay write (validated, atomic), persisted state and the JSON-lines audit log.

State file (JSON, next to the overlay by default):
  seq          last version number issued (versions are monotonic)
  current      overlay document last written by the optimizer
  previous     the overlay that was live before it (None: base only)
  changed_at   when `current` was written (start of the revert window)
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pinball.config import Config
from pinball.overlay import Overlay, OverlayRejected, load_overlay, merge


@dataclass
class State:
    seq: int = 0
    current: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None
    changed_at: str | None = None

    def record(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "current": self.current,
            "previous": self.previous,
            "changed_at": self.changed_at,
        }


def atomic_write(path: Path, text: str) -> None:
    """Temp file in the same directory, fsync, os.replace: readers see old or new, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_state(path: Path | None) -> State:
    if path is None or not path.is_file():
        return State()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return State(
        seq=int(raw.get("seq", 0)),
        current=raw.get("current"),
        previous=raw.get("previous"),
        changed_at=raw.get("changed_at"),
    )


def save_state(path: Path | None, state: State) -> None:
    if path is not None:
        atomic_write(path, json.dumps(state.record(), indent=2, sort_keys=True) + "\n")


def live_overlay(path: Path | None) -> Overlay | None:
    """The overlay pinball currently serves, or None (absent or invalid: base only)."""
    if path is None or not path.is_file():
        return None
    try:
        return load_overlay(path)
    except OverlayRejected:
        return None


def next_version(state: State, live: Overlay | None, now: datetime) -> tuple[int, str]:
    """`opt-<seq:06d>-<UTC stamp>`: zero padded, so it also sorts lexically.

    seq continues from the state file and from a live optimizer version, so it
    never goes backwards even if the state file is lost.
    """
    seq = state.seq
    if live is not None and live.version.startswith("opt-"):
        head = live.version.split("-")[1]
        if head.isdigit():
            seq = max(seq, int(head))
    seq += 1
    return seq, f"opt-{seq:06d}-{now.astimezone(UTC):%Y%m%dT%H%M%SZ}"


def build_overlay(
    base: Config,
    live: Overlay | None,
    tier_order: dict[str, tuple[str, ...]],
    version: str,
    now: datetime,
) -> Overlay:
    """Live overlay with tier_order replaced; validated through merge() or OverlayRejected."""
    doc: dict[str, Any] = {} if live is None else live.model_dump(mode="json")
    doc.update(
        version=version,
        generated_at=now.astimezone(UTC).isoformat(),
        tier_order={t: list(m) for t, m in tier_order.items()},
    )
    overlay = Overlay.model_validate(doc)
    merge(base, overlay)  # raises OverlayRejected: never write what pinball would reject
    return overlay


def overlay_doc(overlay: Overlay) -> dict[str, Any]:
    return overlay.model_dump(mode="json")


def write_overlay(path: Path, base: Config, overlay: Overlay) -> None:
    merge(base, overlay)
    atomic_write(path, yaml.safe_dump(overlay_doc(overlay), sort_keys=False))


def append_audit(path: Path | None, entry: dict[str, Any]) -> None:
    """One JSON object per line; content-free by construction (ids, numbers, model names)."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
