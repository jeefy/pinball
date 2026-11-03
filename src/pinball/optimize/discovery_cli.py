"""Env wiring and failure isolation for the new-model pass (`discover`, and inside `run`)."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from pinball.optimize.apply import append_audit
from pinball.optimize.bot import Bot, BotSettings
from pinball.optimize.discover import DEFAULT_MODELS_URL, fetch_catalog
from pinball.optimize.gitea import DEFAULT_REPO, DEFAULT_URL, Gitea
from pinball.optimize.newmodels import Night, run_pass
from pinball.optimize.settings import Settings

log = logging.getLogger("pinball.optimize")


@dataclass(frozen=True, slots=True)
class DiscoveryEnv:
    token: str
    gitea_url: str
    repo: str
    models_url: str
    candidates_path: Path | None
    budget_usd: float
    bot: BotSettings
    strict: bool  # a failed pass makes the process exit non-zero

    @property
    def enabled(self) -> bool:
        return bool(self.token)


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in ("1", "true", "yes")


def discovery_env(env: Mapping[str, str], dry_run: bool) -> DiscoveryEnv:
    path = env.get("PINBALL_OPT_CANDIDATES")
    return DiscoveryEnv(
        token=env.get("PINBALL_GITEA_TOKEN", ""),
        gitea_url=env.get("PINBALL_GITEA_URL") or DEFAULT_URL,
        repo=env.get("PINBALL_GITEA_REPO") or DEFAULT_REPO,
        models_url=env.get("PINBALL_OPT_MODELS_URL") or DEFAULT_MODELS_URL,
        candidates_path=Path(path) if path else None,
        budget_usd=float(env.get("PINBALL_OPT_CANDIDATE_BUDGET_USD") or 5.0),
        bot=BotSettings(
            max_prs=int(env.get("PINBALL_OPT_MAX_PRS") or 3),
            candidate_nights=int(env.get("PINBALL_OPT_CANDIDATE_NIGHTS") or 7),
            merge_without_checks=_flag(env, "PINBALL_OPT_MERGE_WITHOUT_CHECKS"),
            dry_run=dry_run,
        ),
        strict=_flag(env, "PINBALL_OPT_DISCOVERY_STRICT"),
    )


def make_bot(d: DiscoveryEnv) -> Bot:
    return Bot(Gitea(d.gitea_url, d.repo, d.token, read_only=d.bot.dry_run), d.bot)


def isolated(
    d: DiscoveryEnv,
    night: Callable[[], Night],
    out: TextIO,
    bot_factory: Callable[[DiscoveryEnv], Bot] = make_bot,
) -> tuple[bool, dict[str, Any]]:
    """Run the pass; any exception is logged by type only and reported as `failed`."""
    try:
        report = run_pass(night(), bot_factory(d), d.candidates_path)
        report["result"] = "planned" if d.bot.dry_run else "done"
        ok = all(not str(p["result"]).startswith("failed") for p in report["prs"])
    except Exception as e:  # isolation boundary: ranking must survive any discovery bug
        log.warning("new-model pass failed: %s", type(e).__name__)
        report = {"kind": "new_models", "result": f"failed {type(e).__name__}"}
        ok = False
    print(json.dumps(report, indent=2), file=out)
    return ok, report


def night_pass(
    settings: Settings, d: DiscoveryEnv, now: datetime, catalog: Callable[[], Any] | None
) -> Night:
    """A pass with no evaluation; `run` adds samples, tier order and gateway."""
    loader = catalog or (lambda: fetch_catalog(d.models_url))
    seed = int(f"{now:%Y%m%d}")
    return Night(settings, now.date(), seed, loader, budget_usd=d.budget_usd)


def after_ranking(
    d: DiscoveryEnv, night: Night, out: TextIO, bot_factory: Callable[[DiscoveryEnv], Bot]
) -> int:
    """Inside `run`: skipped without a token; audited; exit 1 only when strict."""
    if not d.enabled:
        return 0
    ok, report = isolated(d, lambda: night, out, bot_factory)
    if not d.bot.dry_run:
        append_audit(night.settings.audit_file, {"at": night.today.isoformat(), **report})
    return 1 if d.strict and not ok else 0


def discover(
    d: DiscoveryEnv, night: Night, out: TextIO, bot_factory: Callable[[DiscoveryEnv], Bot]
) -> int:
    """`pinball-optimize discover`: no DB, no replay; exit 2 without a token, 1 on failure."""
    if not d.enabled:
        print("PINBALL_GITEA_TOKEN is required", file=sys.stderr)
        return 2
    ok, _ = isolated(d, lambda: night, out, bot_factory)
    return 0 if ok else 1
