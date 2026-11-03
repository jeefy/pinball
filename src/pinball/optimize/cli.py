"""`pinball-optimize`: nightly re-ranking, dry-run and revert check.

  pinball-optimize run [--dry-run] [--seed N] [--per-tier N] [--days N] [--candidates a,b]
  pinball-optimize --check-revert
  python -m pinball.optimize ...

Exit codes: 0 ok (including "nothing to change"), 1 runtime failure,
2 bad configuration, 3 candidate overlay rejected by merge() (not written).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TextIO

from pinball.config import Config, Tier, load_config
from pinball.optimize.apply import (
    State,
    append_audit,
    build_overlay,
    live_overlay,
    load_state,
    next_version,
    overlay_doc,
    save_state,
    write_overlay,
)
from pinball.optimize.db import TIERS, PostgresDb, ReplayDb
from pinball.optimize.gateway import Budget, Gateway, HttpGateway
from pinball.optimize.rank import agentic_share, collect, rerank
from pinball.optimize.replay import make_plan, projected_cost, run_replay
from pinball.optimize.revert import check
from pinball.optimize.sample import draw_samples
from pinball.optimize.settings import Settings, settings_from_env
from pinball.overlay import Overlay, OverlayRejected, merge

log = logging.getLogger("pinball.optimize")

EXIT_OK, EXIT_FAIL, EXIT_CONFIG, EXIT_REJECTED = 0, 1, 2, 3


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pinball-optimize", description="Nightly within-tier re-ranking for pinball."
    )
    p.add_argument("command", nargs="?", default="run", choices=["run"])
    p.add_argument("--dry-run", action="store_true", help="sample and plan; call no model")
    p.add_argument("--check-revert", action="store_true", help="revert a regressing overlay")
    p.add_argument("--seed", type=int, default=None, help="default: today's date as YYYYMMDD")
    p.add_argument("--per-tier", type=int, default=20)
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--candidates", default="", help="comma list; default: every tier model")
    return p.parse_args(argv)


@dataclass
class Deps:
    db: ReplayDb
    gateway: Gateway | None  # None: building one is forbidden (dry-run)
    now: datetime
    out: TextIO


def _base(settings: Settings) -> Config:
    return load_config(settings.config_path)


def _current_order(base: Config, live: Overlay | None) -> dict[str, tuple[str, ...]]:
    """Global tier order pinball serves now: base, with the live overlay's tier_order."""
    tiers: dict[Tier, tuple[str, ...]] = dict(base.tiers)
    if live is not None:
        merge(base, live)  # live_overlay() already validated the schema; this checks rules
        tiers.update(live.tier_order)
    return {str(t): tuple(m) for t, m in tiers.items()}


def nightly(settings: Settings, args: argparse.Namespace, deps: Deps) -> int:
    base = _base(settings)
    live = live_overlay(settings.overlay_path)
    order = _current_order(base, live)
    seed = args.seed if args.seed is not None else int(f"{deps.now:%Y%m%d}")
    samples = draw_samples(deps.db, args.days, args.per_tier, seed)
    only = frozenset(c for c in args.candidates.split(",") if c) or None
    plan = make_plan(samples, order, settings.judges, seed, only)
    budget = Budget(settings.budget_usd, settings.prices)
    projected, fitting = projected_cost(plan, budget)
    summary = {
        "seed": seed,
        "samples": len(samples),
        "per_tier": {t: sum(s.tier == t for s in samples) for t in TIERS},
        "agentic_share": {t: round(agentic_share(samples, t), 4) for t in TIERS},
        "candidate_calls": plan.candidate_calls,
        "judge_calls": plan.judge_calls,
        "projected_worst_case_usd": round(projected, 4),
        "pairs_fitting_cap": fitting,
        "budget_usd": settings.budget_usd,
        "prices_are_estimates": True,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **summary}, indent=2), file=deps.out)
        return EXIT_OK
    if deps.gateway is None:
        raise SystemExit("gateway required for a real run")
    results = run_replay(plan, deps.gateway, budget, settings.workers)
    evidence = collect(results, samples, settings.prices)
    tiers = {
        t: rerank(t, order[t], evidence.get(t, {}), agentic_share(samples, t), settings.rank)
        for t in TIERS
    }
    state = load_state(settings.state_file)
    entry: dict[str, Any] = {
        "kind": "run",
        "at": deps.now.isoformat(),
        **summary,
        "spent_usd": round(budget.spent, 6),
        "calls": budget.calls,
        "budget_exhausted": budget.exhausted,
        "errors": sum(r.error is not None for r in results),
        "previous_version": live.version if live else "",
        "tiers": {t: r.record() for t, r in tiers.items()},
    }
    changed = any(r.after != r.before for r in tiers.values())
    code = EXIT_OK
    if not changed:
        entry["result"] = "unchanged"
    elif settings.overlay_path is None:
        entry["result"] = "no_overlay_path"
    else:
        seq, version = next_version(state, live, deps.now)
        try:
            overlay = build_overlay(
                base, live, {t: r.after for t, r in tiers.items()}, version, deps.now
            )
            write_overlay(settings.overlay_path, base, overlay)
        except OverlayRejected as e:
            entry["result"] = f"rejected {e.reason}"
            code = EXIT_REJECTED
        else:
            save_state(
                settings.state_file,
                State(
                    seq=seq,
                    current=overlay_doc(overlay),
                    previous=overlay_doc(live) if live else None,
                    changed_at=deps.now.isoformat(),
                ),
            )
            entry["result"] = "written"
            entry["version"] = version
    append_audit(settings.audit_file, entry)
    print(json.dumps(entry, indent=2), file=deps.out)
    return code


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
            entry["result"] = "kept"
        else:
            base = _base(settings)
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


def main(
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    db_factory: Callable[[str], ReplayDb] | None = None,
    gateway_factory: Callable[[Settings], Gateway] | None = None,
    now: datetime | None = None,
    out: TextIO | None = None,
) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    settings = settings_from_env(os.environ if env is None else env)
    if not settings.db_dsn:
        print("PINBALL_OPT_DB_DSN is required", file=sys.stderr)
        return EXIT_CONFIG
    real_run = not args.dry_run and not args.check_revert
    if real_run and not settings.gateway_url and gateway_factory is None:
        print("PINBALL_OPT_GATEWAY_URL is required", file=sys.stderr)
        return EXIT_CONFIG
    db = (db_factory or PostgresDb)(settings.db_dsn)
    gateway = None
    if real_run:
        gateway = (gateway_factory or _http_gateway)(settings)
    deps = Deps(db, gateway, now or datetime.now(UTC), out or sys.stdout)
    if args.check_revert:
        return check_revert(settings, deps)
    return nightly(settings, args, deps)


def _http_gateway(settings: Settings) -> Gateway:
    return HttpGateway(settings.gateway_url, settings.gateway_timeout)
