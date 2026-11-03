"""Nightly new-model pass: evaluate candidates, then promote / drop / propose via bot PRs.

Inputs:
  * PINBALL_OPT_CANDIDATES: the mounted pinball-candidates ConfigMap file (what
    the cluster has synced, so its opt-<id> aliases exist on the gateway);
  * PINBALL_OPT_MODELS_URL: copilot-api /v1/models;
  * carrier main, read through Gitea (the files every PR is cut from).

Order inside the PR cap: promotions, then drops, then new candidates.
Every failure is caught by the caller (cli): this pass never fails ranking.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pinball.optimize.bot import Bot, candidate_change, drop_change, promote_change
from pinball.optimize.carrier_edit import ModelId
from pinball.optimize.discover import CatalogModel, new_models, parse_catalog
from pinball.optimize.evaluate import Verdict, evaluate
from pinball.optimize.gateway import Budget, Gateway
from pinball.optimize.invariants import candidate_lists, pool_ids
from pinball.optimize.sample import Sample
from pinball.optimize.settings import Settings

log = logging.getLogger("pinball.optimize")


@dataclass(frozen=True, slots=True)
class Night:
    """Everything one pass needs besides the bot; built by the cli."""

    settings: Settings
    today: date
    seed: int
    catalog: Callable[[], Any]  # raw /v1/models payload
    samples: Sequence[Sample] = ()
    order: dict[str, tuple[str, ...]] | None = None
    gateway: Gateway | None = None  # None: no evaluation tonight (dry-run)
    budget_usd: float = 0.0


def _evaluable(path: Path | None) -> tuple[ModelId, ...]:
    if path is None or not path.is_file():
        return ()
    return candidate_lists(path.read_text(encoding="utf-8")).candidates


def _rates(settings: Settings, mid: str) -> tuple[str, str]:
    p = settings.prices.models.get(mid, settings.prices.default)
    return f"{p.input:.2f}", f"{p.output:.2f}"


def run_pass(night: Night, bot: Bot, candidates_path: Path | None) -> dict[str, Any]:
    s = night.settings
    report: dict[str, Any] = {"kind": "new_models"}
    main = bot.main_files()
    lists = candidate_lists(main.candidates)
    pool = pool_ids(main.pinball)
    catalog = parse_catalog(night.catalog())
    report["catalog_chat_models"] = len(catalog)

    live = [c for c in _evaluable(candidates_path) if c in lists.candidates]
    verdicts: list[Verdict] = []
    if night.gateway is not None and night.order is not None and live:
        budget = Budget(night.budget_usd, s.prices)
        verdicts = evaluate(
            night.samples, night.order, live, (night.gateway, budget), s, night.seed
        )
        report["evaluation_spent_usd"] = round(budget.spent, 6)
    report["evaluated"] = [v.record() for v in verdicts]

    changes = []
    for v in verdicts:
        model = catalog.get(v.model)
        if v.tier is not None and model is not None:
            changes.append(promote_change(model, v.tier, _rates(s, v.model), v.record()))
    nights = bot.settings.candidate_nights
    for v in verdicts:  # only a candidate that was evaluated tonight, and lost, can age out
        added = lists.added.get(v.model, "")
        age = (night.today - date.fromisoformat(added)).days if added else nights
        if v.tier is None and age >= nights:
            changes.append(drop_change(v.model, nights, night.today))
    fresh: list[CatalogModel] = new_models(catalog, (*pool, *lists.candidates, *lists.dropped))
    report["new"] = [m.id for m in fresh]
    changes += [candidate_change(m, night.today) for m in fresh]
    report["prs"] = [bot.submit(c).record() for c in changes]
    return report
