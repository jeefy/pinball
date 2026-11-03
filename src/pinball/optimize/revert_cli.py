"""`pinball-optimize --check-revert`: per-caller revert, then the global one."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from pinball.config import load_config
from pinball.optimize.apply import (
    State,
    append_audit,
    live_overlay,
    load_state,
    save_state,
    write_overlay,
)
from pinball.optimize.callers_run import RevertCtx, revert_callers
from pinball.optimize.deps import Deps
from pinball.optimize.revert import check
from pinball.optimize.settings import Settings
from pinball.overlay import Overlay, OverlayRejected

log = logging.getLogger("pinball.optimize")

EXIT_OK, EXIT_REJECTED = 0, 3


def check_revert(settings: Settings, deps: Deps) -> int:
    state = load_state(settings.state_file)
    live = live_overlay(settings.overlay_path)
    entry: dict[str, Any] = {"kind": "check_revert", "at": deps.now.isoformat()}
    if settings.overlay_path is None or state.current is None or state.changed_at is None:
        entry["result"] = "nothing_to_check"
    elif live is None or live.version != state.current.get("version"):
        # Someone else changed the overlay since we wrote it: not ours to revert.
        entry["result"] = "foreign_overlay"
    else:
        decision = check(
            deps.db, datetime.fromisoformat(state.changed_at), deps.now, settings.revert
        )
        entry.update(decision.record())
        entry["version"] = live.version
        if not decision.revert:
            ctx = RevertCtx(
                load_config(settings.config_path),
                settings.overlay_path,
                settings.state_file,
                (settings.revert, deps.callers),
            )
            try:
                entry.update(revert_callers(deps.db, live, state, deps.now, ctx))
            except OverlayRejected as e:
                entry["result"] = f"callers_rejected {e.reason}"
            else:
                entry["result"] = "callers_reverted" if entry["callers_reverted"] else "kept"
        else:
            base = load_config(settings.config_path)
            if state.previous is None:
                settings.overlay_path.unlink(missing_ok=True)  # base config only
                restored = ""
            else:
                prev = Overlay.model_validate(state.previous)
                try:
                    write_overlay(settings.overlay_path, base, prev)
                except OverlayRejected as e:
                    entry["result"] = f"previous_rejected {e.reason}"
                    append_audit(settings.audit_file, entry)
                    print(json.dumps(entry, indent=2), file=deps.out)
                    return EXIT_REJECTED
                restored = prev.version
            save_state(
                settings.state_file,
                State(seq=state.seq, current=state.previous, previous=None, changed_at=None),
            )
            entry["result"] = "reverted"
            entry["restored_version"] = restored
            log.warning("overlay reverted %s -> %r: %s", live.version, restored, decision.reasons)
    append_audit(settings.audit_file, entry)
    print(json.dumps(entry, indent=2), file=deps.out)
    return EXIT_OK
