"""Stratified dispatch of replay pairs: breadth first, a soft budget share per tier.

The plan is already interleaved over (tier, candidate). On top of that the cap is
split into per-tier shares proportional to weights (tiers without pairs get none).
A pair is dispatched only when its worst case fits its tier's limit:

    committed(t) = settled cost of finished pairs + worst case of pairs in flight
    limit(t)     = share(t) + rollover * weight(t) / sum(weight of open tiers)

`rollover` is the unused share of CLOSED tiers (no queued and no running pair), so
a tier never spends past its share while another tier still has unissued pairs.
Because the limits sum to the cap, a dispatched pair's judges always fit too.

When nothing runs and no open tier can dispatch, the stuck tier furthest from
fitting is closed (its remaining pairs are `budget` refusals) and its unused
share rolls over to the rest.

Every pair runs on its own daemon thread. A pair that passes `deadline_s` is
recorded as the gateway error `timeout`, its remaining judge calls are cancelled
and its worker slot is freed; the abandoned call keeps its budget reservation
until it returns, so the cap still holds.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pinball.optimize.gateway import Budget, Gateway
from pinball.optimize.replay import PairResult, Plan, empty_result, pair_worst_cost, replay_pair
from pinball.optimize.report import BUDGET
from pinball.optimize.sample import Sample

log = logging.getLogger("pinball.optimize")

TIMEOUT = "timeout"

Pair = tuple[int, Sample, str]  # (plan index, sample, candidate)


@dataclass
class _Lane:
    weight: float
    share: float = 0.0
    queued: list[tuple[Pair, float]] = field(default_factory=list)  # (pair, worst case)
    running: dict[int, float] = field(default_factory=dict)  # plan index -> worst case
    settled: float = 0.0
    closed: bool = False

    @property
    def committed(self) -> float:
        return self.settled + sum(self.running.values())

    @property
    def open(self) -> bool:
        return not self.closed and bool(self.queued or self.running)


@dataclass(frozen=True, slots=True)
class ReplayRun:
    results: list[PairResult]  # in plan order
    shares: dict[str, dict[str, float]]  # tier -> {share_usd, spent_usd}


@dataclass(frozen=True, slots=True)
class Limits:
    workers: int
    deadline_s: float
    weights: Mapping[str, float] = field(default_factory=dict)  # missing tier: 1.0


class _Dispatcher:
    def __init__(self, plan: Plan, budget: Budget, limits: Limits) -> None:
        self.budget = budget
        self.limits = limits
        self.lanes: dict[str, _Lane] = {}
        self.issued: dict[tuple[str, str], int] = {}
        for i, (sample, model) in enumerate(plan.pairs):
            lane = self.lanes.setdefault(
                sample.tier, _Lane(weight=max(limits.weights.get(sample.tier, 1.0), 0.0))
            )
            worst = pair_worst_cost((sample, model), plan.judges, budget.prices)
            lane.queued.append(((i, sample, model), worst))
        total = sum(lane.weight for lane in self.lanes.values())
        for lane in self.lanes.values():
            lane.share = budget.cap * lane.weight / total if total > 0 else 0.0

    def limit(self, tier: str) -> float:
        lane = self.lanes[tier]
        rollover = sum(max(x.share - x.committed, 0.0) for x in self.lanes.values() if not x.open)
        weights = sum(x.weight for x in self.lanes.values() if x.open)
        extra = rollover * lane.weight / weights if weights > 0 else 0.0
        return lane.share + extra

    def next_pair(self) -> tuple[Pair, float] | None:
        """A queued pair that fits its tier, least-covered (tier, candidate) first."""
        best: tuple[tuple[int, int], tuple[Pair, float]] | None = None
        for tier, lane in self.lanes.items():
            if not lane.open:
                continue
            room = self.limit(tier) - lane.committed
            for item in lane.queued:
                (i, _, model), worst = item
                key = (self.issued.get((tier, model), 0), i)
                if worst <= room and (best is None or key < best[0]):
                    best = (key, item)
        if best is None:
            return None
        (_, sample, model), _ = best[1]
        self.issued[sample.tier, model] = best[0][0] + 1
        return best[1]

    def close_stuck(self) -> list[Pair]:
        """Close the stuck lane furthest from fitting; its queue becomes refusals."""
        stuck = [(t, x) for t, x in self.lanes.items() if x.open and x.queued]
        if not stuck:
            return []

        def gap(item: tuple[str, _Lane]) -> float:
            tier, lane = item
            return min(w for _, w in lane.queued) - (self.limit(tier) - lane.committed)

        _, lane = max(stuck, key=gap)
        lane.closed = True
        refused = [p for p, _ in lane.queued]
        lane.queued.clear()
        return refused


def run_replay(
    plan: Plan,
    gateway: Gateway,
    budget: Budget,
    limits: Limits,
    clock: Callable[[], float] = time.monotonic,
) -> ReplayRun:
    d = _Dispatcher(plan, budget, limits)
    done: queue.Queue[tuple[int, PairResult]] = queue.Queue()
    results: dict[int, PairResult] = {}
    started: dict[int, tuple[float, threading.Event]] = {}
    abandoned: dict[int, float] = {}  # plan index -> worst case still held

    def work(pair: Pair, cancel: threading.Event) -> None:
        i, sample, model = pair
        done.put((i, replay_pair(sample, model, plan, gateway, budget, cancel=cancel)))

    while True:
        while len(started) < limits.workers and (picked := d.next_pair()) is not None:
            pair, worst = picked
            i, sample, _ = pair
            lane = d.lanes[sample.tier]
            lane.queued.remove(picked)
            lane.running[i] = worst
            cancel = threading.Event()
            started[i] = (clock(), cancel)
            threading.Thread(target=work, args=(pair, cancel), daemon=True).start()
        if not started:
            refused = d.close_stuck()
            for i, sample, model in refused:
                results[i] = empty_result(sample, model, plan, error=BUDGET)
            if refused:
                continue
            break
        wait = min(t + limits.deadline_s for t, _ in started.values()) - clock()
        try:
            i, result = done.get(timeout=max(wait, 0.0))
        except queue.Empty:
            pass
        else:
            if i in abandoned:  # late return of a timed-out pair: swap hold for spend
                d.lanes[result.tier].settled += result.cost_usd - abandoned.pop(i)
            else:
                started.pop(i)
                results[i] = result
                lane = d.lanes[result.tier]
                lane.running.pop(i)
                lane.settled += result.cost_usd
        now = clock()
        for i, (t0, cancel) in list(started.items()):
            if now - t0 >= limits.deadline_s:
                cancel.set()
                started.pop(i)
                sample, model = plan.pairs[i]
                results[i] = empty_result(sample, model, plan, error=TIMEOUT)
                # Keep the worst case committed until the call returns.
                lane = d.lanes[sample.tier]
                abandoned[i] = lane.running.pop(i)
                lane.settled += abandoned[i]
    ordered = [results[i] for i in sorted(results)]
    shares = {
        t: {"share_usd": round(x.share, 6), "spent_usd": round(x.settled, 6)}
        for t, x in d.lanes.items()
    }
    log.info(
        "replay done pairs=%d spent=%.4f calls=%d exhausted=%s",
        len(ordered),
        budget.spent,
        budget.calls,
        budget.exhausted,
    )
    return ReplayRun(ordered, shares)
