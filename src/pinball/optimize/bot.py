"""Carrier bot PRs for new models: Step A (evaluation alias), Step B (promote) and drop.

Branches are `pinball-bot/<step>-<id>`; one branch = one PR = one step for one
model, so a branch or PR that already exists (open, merged or closed) means
the step was already taken and is skipped. At most `max_prs` new PRs a night.

Safety gate, in this order, for every PR:
  1. every edited file is produced by a line-based edit of main's current text
     (blob sha pinned: the contents API refuses a commit if main moved);
  2. invariants.check() (the carrier drift test re-implemented) passes on the
     edited texts, else no commit at all;
  3. the bot itself waits for a GREEN combined commit status on the PR head
     (carrier's drift.yaml runs the drift tests on every PR touching
     agentgateway.yaml / pinball.yaml / pinball-candidates.yaml) and only then
     calls merge. carrier main has no branch protection requiring status
     checks, and without one Gitea's merge_when_checks_succeed merges AT ONCE,
     so that flag is not the gate. Failed checks leave the PR open for the
     owner; pending checks are re-polled on the next night. A PR with no
     status at all is merged only when merge_without_checks is set.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from enum import StrEnum
from typing import Any

from pinball.optimize import carrier_edit as edit
from pinball.optimize.carrier_edit import ModelId
from pinball.optimize.discover import CatalogModel
from pinball.optimize.gitea import FileEdit, Gitea, GiteaError
from pinball.optimize.invariants import PATHS, CarrierFiles, InvariantError, check

log = logging.getLogger("pinball.optimize")


class Step(StrEnum):
    CANDIDATE = "candidate"
    PROMOTE = "promote"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class BotSettings:
    max_prs: int = 3
    candidate_nights: int = 7
    checks_wait_s: float = 900.0
    merge_without_checks: bool = False
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class Change:
    step: Step
    model: ModelId
    title: str
    body: str
    edit: Callable[[CarrierFiles], CarrierFiles]

    @property
    def branch(self) -> str:
        return f"pinball-bot/{self.step}-{self.model}"


@dataclass
class Outcome:
    step: str
    model: str
    branch: str
    result: str
    pr: int | None = None

    def record(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model,
            "branch": self.branch,
            "result": self.result,
            "pr": self.pr,
        }


# --- the three changes -----------------------------------------------------------------


def candidate_change(m: CatalogModel, today: date) -> Change:
    def apply(f: CarrierFiles) -> CarrierFiles:
        entry = f'{{id: {m.id}, added: "{today.isoformat()}"}}'
        return replace(
            f,
            gateway=edit.add_opt_alias(f.gateway, m.id),
            candidates=edit.add_list_entry(f.candidates, "candidates", entry),
        )

    body = (
        f"New Copilot chat model `{m.id}` found by `pinball-optimize discover`.\n\n"
        f"Adds ONLY the public evaluation alias `opt-{m.id}` and a `candidates` entry. "
        "It is not in the pinball pool and has no `auto` target: nothing live changes. "
        "The nightly optimizer replays sampled traffic on it from tomorrow.\n"
    )
    return Change(Step.CANDIDATE, m.id, f"pinball: evaluate candidate {m.id}", body, apply)


def promote_change(
    m: CatalogModel, tier: str, rates: tuple[str, str], evidence: dict[str, Any]
) -> Change:
    def apply(f: CarrierFiles) -> CarrierFiles:
        pinball = edit.add_pool_model(f.pinball, m.id, m.prompt_tokens, m.vision)
        gateway = edit.add_pinball_model(f.gateway, m.id)
        gateway = edit.add_auto_target(gateway, m.id)
        gateway = edit.add_rates(gateway, m.id, rates)
        return CarrierFiles(
            pinball=edit.append_to_tier(pinball, tier, m.id),
            gateway=gateway,
            candidates=edit.remove_list_entry(f.candidates, "candidates", m.id),
        )

    flagged = ", ".join(m.defaulted)
    lines = [
        f"`{m.id}` beat the weakest `{tier}` incumbent in overnight replay.",
        "",
        f"* pool: `{{prompt_tokens: {m.prompt_tokens}, vision: {str(m.vision).lower()}}}`",
        f"* tier `{tier}`, LAST position (the overlay may promote it within the tier later)",
        f"* `pinball-{m.id}` internal model and `auto`/`pinball` target",
        f"* PLACEHOLDER openai rates input={rates[0]} output={rates[1]} USD/1M (estimate)",
    ]
    if flagged:
        lines.append(f"* CONSERVATIVE DEFAULT (not in /v1/models capabilities): {flagged}")
    lines += ["", "Evidence (agreed judge fields only):", "```json", _json(evidence), "```"]
    title = f"pinball: add {m.id} to the {tier} tier"
    return Change(Step.PROMOTE, m.id, title, "\n".join(lines) + "\n", apply)


def drop_change(mid: ModelId, nights: int, today: date) -> Change:
    def apply(f: CarrierFiles) -> CarrierFiles:
        cands = edit.remove_list_entry(f.candidates, "candidates", mid)
        entry = f'{{id: {mid}, dropped: "{today.isoformat()}"}}'
        return replace(
            f,
            gateway=edit.remove_opt_alias(f.gateway, mid),
            candidates=edit.add_list_entry(cands, "dropped", entry),
        )

    body = (
        f"`{mid}` won no tier in {nights} nights of evaluation. Removes `opt-{mid}` and "
        "moves it to `dropped` so discovery does not propose it again "
        "(delete the `dropped` entry to re-evaluate).\n"
    )
    return Change(Step.DROP, mid, f"pinball: drop candidate {mid}", body, apply)


def _json(doc: dict[str, Any]) -> str:
    import json

    return json.dumps(doc, indent=2, sort_keys=True)


# --- the bot ---------------------------------------------------------------------------


@dataclass
class Bot:
    gitea: Gitea
    settings: BotSettings
    sleep: Callable[[float], None] = time.sleep
    opened: int = 0
    unmerged: int | None = None  # tonight's PR still waiting: open no further one
    _main: dict[str, tuple[str, str]] = field(default_factory=dict)

    def main_files(self) -> CarrierFiles:
        for key, path in PATHS.items():
            if key not in self._main:
                self._main[key] = self.gitea.file(path)
        return CarrierFiles(**{k: v[0] for k, v in self._main.items()})

    def submit(self, change: Change) -> Outcome:
        out = Outcome(change.step, change.model, change.branch, "")
        try:
            out.result = self._submit(change, out)
        except (GiteaError, InvariantError, edit.EditError) as e:
            out.result = f"failed {type(e).__name__}: {e}"
            log.warning("bot %s %s failed: %s", change.step, change.model, e)
        return out

    def _submit(self, change: Change, out: Outcome) -> str:
        prs = self.gitea.pulls(change.branch)
        if prs or self.gitea.branch_exists(change.branch):
            still_open = [p for p in prs if p.state == "open"]
            if len(still_open) == 1 and not self.settings.dry_run:
                out.pr = still_open[0].number  # a PR whose checks were slow last night
                return self._merge(out.pr, change.branch, wait=False)
            return "exists"
        if self.opened >= self.settings.max_prs:
            return "capped"
        if self.unmerged is not None:
            return f"waiting_for_pr {self.unmerged}"  # PRs are serial: never two edits of one base
        before = self.main_files()
        after = change.edit(before)
        check(after)  # never commit, let alone merge, a pool that breaks the drift test
        edits = [
            FileEdit(PATHS[k], getattr(after, k), self._main[k][1])
            for k in PATHS
            if getattr(after, k) != getattr(before, k)
        ]
        if self.settings.dry_run:
            return "planned " + ",".join(e.path for e in edits)
        self.opened += 1
        self.gitea.commit_files(change.branch, change.title, edits)
        out.pr = self.gitea.open_pull(change.branch, change.title, change.body)
        return self._merge(out.pr, change.branch, wait=True)

    def _merge(self, number: int, branch: str, *, wait: bool) -> str:
        result = self._try_merge(number, branch, wait=wait)
        if result.startswith("merged"):
            self._main.clear()  # main moved: the next change is cut from the new head
        else:
            self.unmerged = number
        return result

    def _try_merge(self, number: int, branch: str, *, wait: bool) -> str:
        """Merge only on a GREEN combined status of the PR head (see module doc)."""
        waited, step = 0.0, 15.0
        while True:
            (pr,) = [p for p in self.gitea.pulls(branch) if p.number == number]
            state, statuses = self.gitea.checks_state(pr.head_sha)
            if state in ("failure", "error"):
                return f"checks_{state}"  # left open for the owner
            if statuses and state == "success":
                break
            if not wait or waited >= self.settings.checks_wait_s:
                if statuses or not self.settings.merge_without_checks:
                    return "checks_pending"  # retried next night
                break
            self.sleep(step)
            waited += step
        self.gitea.merge(number)
        return "merged" if statuses else "merged_without_checks"
