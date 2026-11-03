"""Per-caller profile learning on top of the global within-tier ranking.

Sampling: the global per-tier sample is drawn first and is never altered. For
each caller with at least `min_samples` pool rows, up to `per_tier` extra rows
per tier are drawn from that caller's remaining traffic (seed `<seed>:<caller>:
<tier>`). Global ranking only sees the global sample's results.

Learning, per caller with at least `min_samples` judged non-agentic samples
(global + extra samples of that caller):
  tier_order  rerank() of the caller's evidence starting from tonight's global
              order, with the same margin / max_moves rules; emitted only for
              tiers whose order differs.
  weight      n / (n + shrink_k), n = judged non-agentic samples.
  offsets     standard = -1 only when the caller's light-tier served replies are
              judged unacceptable more often than the global light rate by more
              than `light_fail_gap` (with >= rank.min_samples light verdicts);
              otherwise zero. heavy is never learned.
A caller whose evidence yields neither an order nor an offset gets no profile.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pinball.config import Config
from pinball.optimize.caller_settings import CallerSettings
from pinball.optimize.db import TIERS, ReplayDb
from pinball.optimize.rank import agentic_share, collect, rerank
from pinball.optimize.replay import PairResult
from pinball.optimize.sample import Sample, take
from pinball.optimize.settings import PriceTable, RankSettings
from pinball.overlay import CALLER_KEY_MAX, CallerProfile, Overlay, OverlayRejected, merge


def eligible_callers(pool: list[dict[str, Any]], cs: CallerSettings) -> list[str]:
    """Callers with enough pool rows, busiest first, at most max_profiles."""
    counts = Counter(str(r.get("caller") or "")[:CALLER_KEY_MAX] for r in pool)
    counts.pop("", None)
    ranked = sorted(
        (c for c, n in counts.items() if n >= cs.min_samples), key=lambda c: (-counts[c], c)
    )
    return ranked[: cs.max_profiles]


def draw_caller_samples(
    db: ReplayDb,
    pool: list[dict[str, Any]],
    taken: list[Sample],
    seed: int,
    cs: CallerSettings,
) -> list[Sample]:
    """Extra per-caller samples from rows the global sample did not take."""
    used = {s.log_id for s in taken}
    out: list[Sample] = []
    for caller in eligible_callers(pool, cs):
        for tier in TIERS:
            ids = sorted(
                str(r["id"])
                for r in pool
                if str(r.get("caller") or "")[:CALLER_KEY_MAX] == caller
                and str(r["tier"]) == tier
                and str(r["id"]) not in used
            )
            random.Random(f"{seed}:{caller}:{tier}").shuffle(ids)
            out.extend(take(db, ids, cs.per_tier))
    return out


@dataclass(frozen=True, slots=True)
class Learned:
    caller: str
    n: int  # judged non-agentic samples
    profile: CallerProfile | None
    summary: dict[str, Any]


def judged_non_agentic(results: list[PairResult], samples: list[Sample]) -> int:
    judged = {r.log_id for r in results if r.error is None and r.view() is not None}
    return sum(1 for s in samples if not s.agentic and s.log_id in judged)


def light_fail_rate(results: list[PairResult]) -> tuple[int, float]:
    """(verdicts, share of agreed served_ok == False) over light-tier pairs."""
    oks = [
        v.served_ok
        for r in results
        if r.tier == "light" and r.error is None
        for v in [r.view()]
        if v is not None and v.served_ok is not None
    ]
    return len(oks), (oks.count(False) / len(oks) if oks else 0.0)


@dataclass(frozen=True, slots=True)
class LearnInput:
    results: list[PairResult]
    samples: list[Sample]  # global + per-caller
    global_ids: frozenset[str]
    order: dict[str, tuple[str, ...]]  # tonight's global order (after re-ranking)


@dataclass(frozen=True, slots=True)
class Policy:
    rank: RankSettings
    callers: CallerSettings
    prices: PriceTable


def learn_caller(caller: str, data: LearnInput, policy: Policy) -> Learned:
    mine = [s for s in data.samples if s.caller == caller]
    ids = {s.log_id for s in mine}
    results = [r for r in data.results if r.log_id in ids]
    n = judged_non_agentic(results, mine)
    summary: dict[str, Any] = {"samples": len(mine), "judged_non_agentic": n}
    if n < policy.callers.min_samples:
        summary["result"] = "below_min_samples"
        return Learned(caller, n, None, summary)
    ev = collect(results, mine, policy.prices)
    tier_order: dict[str, tuple[str, ...]] = {}
    tiers: dict[str, Any] = {}
    for t in TIERS:
        r = rerank(t, data.order[t], ev.get(t, {}), agentic_share(mine, t), policy.rank)
        tiers[t] = {
            "moves": [m.record() for m in r.moves],
            "scores": [s.record() for s in r.scores],
        }
        if r.after != r.before:
            tier_order[t] = r.after
    light_n, light_fail = light_fail_rate(results)
    _, global_fail = light_fail_rate([r for r in data.results if r.log_id in data.global_ids])
    standard = 0
    if (
        light_n >= policy.rank.min_samples
        and light_fail - global_fail > policy.callers.light_fail_gap
    ):
        standard = -1
    weight = round(n / (n + policy.callers.shrink_k), 4) if n + policy.callers.shrink_k else 1.0
    summary.update(
        tiers=tiers,
        light_verdicts=light_n,
        light_fail_rate=round(light_fail, 4),
        global_light_fail_rate=round(global_fail, 4),
        weight=weight,
    )
    if not tier_order and not standard:
        summary["result"] = "same_as_global"
        return Learned(caller, n, None, summary)
    profile = CallerProfile.model_validate(
        {"tier_order": tier_order, "threshold_offsets": {"standard": standard}, "weight": weight}
    )
    summary["result"] = "learned"
    return Learned(caller, n, profile, summary)


def valid_profile(
    base: Config, global_order: dict[str, tuple[str, ...]], caller: str, p: CallerProfile
) -> str:
    """'' when merge() accepts this profile on top of the global order, else the reason."""
    probe = Overlay.model_validate(
        {"version": "probe", "generated_at": "2026-01-01T00:00:00Z", "tier_order": global_order}
    ).model_copy(update={"callers": {caller: p}})
    try:
        merge(base, probe)
    except OverlayRejected as e:
        return e.reason
    return ""


@dataclass
class Plan:
    profiles: dict[str, CallerProfile]
    changed: dict[str, CallerProfile | None] = field(default_factory=dict)  # caller -> previous
    audit: dict[str, Any] = field(default_factory=dict)


def plan_profiles(
    live: dict[str, CallerProfile],
    learned: list[Learned],
    stale: set[str],
    base: Config,
    order: dict[str, tuple[str, ...]],
    cs: CallerSettings,
) -> Plan:
    """New caller map: drop stale, apply learned (validated) up to max_changes, cap the total."""
    profiles = {c: p for c, p in live.items() if c not in stale}
    plan = Plan(profiles, audit={c: {"result": "dropped_stale"} for c in sorted(stale & set(live))})
    for c in sorted(stale & set(live)):
        plan.changed[c] = live[c]
    for item in sorted(learned, key=lambda x: (-x.n, x.caller)):
        entry = dict(item.summary)
        plan.audit[item.caller] = entry
        p = item.profile
        if p is None or profiles.get(item.caller) == p:
            continue
        reason = valid_profile(base, order, item.caller, p)
        if reason:
            entry["result"] = f"rejected {reason}"
        elif len(plan.changed) - len(stale & set(live)) >= cs.max_changes:
            entry["result"] = "deferred_max_changes"
        elif item.caller not in profiles and len(profiles) >= cs.max_profiles:
            entry["result"] = "deferred_max_profiles"
        else:
            plan.changed[item.caller] = live.get(item.caller)
            profiles[item.caller] = p
    return plan
