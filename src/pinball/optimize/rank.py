"""Turn agreed verdicts into a within-tier order.

Per (tier, model) evidence comes from both sides of every judged pair: as the
candidate (candidate_ok, candidate outcome) and as the served model
(served_ok, the inverse outcome). With agreed fields only:

    quality  = 0.5 * acceptable_rate + 0.5 * win_rate      (tie = half a win)
    shrunk   = (n * quality + k * prior(position)) / (n + k)
    score    = shrunk - cost_weight * cost_per_request / tier_median_cost

prior(position) = prior_top - prior_step * position encodes the current
order, so thin evidence cannot overturn it. A model with fewer than
min_samples NON-AGENTIC agreed comparisons is pinned to its position. Moves
are adjacent swaps of two unpinned models whose score gap exceeds `margin`,
largest gap first, at most max_moves per tier (agentic_max_moves when the
tier's agentic share is at least agentic_share_high).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from pinball.optimize.replay import PairResult
from pinball.optimize.sample import Sample
from pinball.optimize.settings import PriceTable, RankSettings


@dataclass
class Evidence:
    ok_n: int = 0
    ok_yes: int = 0
    pref_n: int = 0
    pref_points: float = 0.0
    n: int = 0  # agreed comparisons
    non_agentic: int = 0
    costs: list[float] = field(default_factory=list)

    def add(self, ok: bool | None, points: float | None, agentic: bool) -> None:
        if ok is None and points is None:
            return
        self.n += 1
        self.non_agentic += not agentic
        if ok is not None:
            self.ok_n += 1
            self.ok_yes += ok
        if points is not None:
            self.pref_n += 1
            self.pref_points += points

    def quality(self) -> float | None:
        if not self.ok_n and not self.pref_n:
            return None
        parts = []
        if self.ok_n:
            parts.append(self.ok_yes / self.ok_n)
        if self.pref_n:
            parts.append(self.pref_points / self.pref_n)
        return sum(parts) / len(parts)


_POINTS = {"win": 1.0, "tie": 0.5, "loss": 0.0}


def collect(
    results: list[PairResult], samples: list[Sample], prices: PriceTable
) -> dict[str, dict[str, Evidence]]:
    """tier -> model -> Evidence."""
    ev: dict[str, dict[str, Evidence]] = {}
    for r in results:
        if r.error is not None:
            continue
        tier = ev.setdefault(r.tier, {})
        cand = tier.setdefault(r.candidate, Evidence())
        cand.costs.append(prices.cost(r.candidate, r.input_tokens, r.output_tokens))
        view = r.view()
        if view is None:
            continue
        pts = None if view.outcome is None else _POINTS[view.outcome]
        cand.add(view.candidate_ok, pts, r.agentic)
        if r.served_model:
            served = tier.setdefault(r.served_model, Evidence())
            served.add(view.served_ok, None if pts is None else 1.0 - pts, r.agentic)
    for s in samples:
        if s.served_model and s.served_model in ev.get(s.tier, {}):
            ev[s.tier][s.served_model].costs.append(
                prices.cost(s.served_model, s.served_input_tokens, s.served_output_tokens)
            )
    return ev


@dataclass(frozen=True, slots=True)
class ModelScore:
    model: str
    position: int
    n: int
    non_agentic: int
    quality: float | None
    prior: float
    shrunk: float
    cost_per_request: float | None
    score: float
    pinned: bool

    def record(self) -> dict[str, Any]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 6)

        return {
            "model": self.model,
            "position": self.position,
            "n": self.n,
            "non_agentic": self.non_agentic,
            "quality": r(self.quality),
            "prior": r(self.prior),
            "shrunk": r(self.shrunk),
            "cost_per_request": r(self.cost_per_request),
            "score": r(self.score),
            "pinned": self.pinned,
        }


@dataclass(frozen=True, slots=True)
class Move:
    model: str
    over: str
    to: int
    gap: float

    def record(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "over": self.over,
            "to": self.to,
            "gap": round(self.gap, 6),
            "why": f"score {self.model} exceeds {self.over} by {self.gap:.4f}",
        }


@dataclass(frozen=True, slots=True)
class TierResult:
    tier: str
    before: tuple[str, ...]
    after: tuple[str, ...]
    scores: tuple[ModelScore, ...]
    moves: tuple[Move, ...]
    max_moves: int
    agentic_share: float

    def record(self) -> dict[str, Any]:
        return {
            "before": list(self.before),
            "after": list(self.after),
            "max_moves": self.max_moves,
            "agentic_share": round(self.agentic_share, 4),
            "scores": [s.record() for s in self.scores],
            "moves": [m.record() for m in self.moves],
        }


def score_tier(
    order: tuple[str, ...], evidence: dict[str, Evidence], s: RankSettings
) -> list[ModelScore]:
    costs = {m: statistics.fmean(e.costs) for m, e in evidence.items() if m in order and e.costs}
    median = statistics.median(costs.values()) if costs else 0.0
    out: list[ModelScore] = []
    for i, model in enumerate(order):
        e = evidence.get(model, Evidence())
        prior = s.prior_top - s.prior_step * i
        q = e.quality()
        n = e.n
        shrunk = prior if q is None else (n * q + s.shrink_k * prior) / (n + s.shrink_k)
        cost = costs.get(model)
        penalty = s.cost_weight * cost / median if cost is not None and median > 0 else 0.0
        out.append(
            ModelScore(
                model=model,
                position=i,
                n=e.n,
                non_agentic=e.non_agentic,
                quality=q,
                prior=prior,
                shrunk=shrunk,
                cost_per_request=cost,
                score=shrunk - penalty,
                pinned=e.non_agentic < s.min_samples,
            )
        )
    return out


def rerank(
    tier: str,
    order: tuple[str, ...],
    evidence: dict[str, Evidence],
    agentic_share: float,
    s: RankSettings,
) -> TierResult:
    scores = score_tier(order, evidence, s)
    by_model = {x.model: x for x in scores}
    cap = s.max_moves
    if agentic_share >= s.agentic_share_high:
        cap = min(cap, s.agentic_max_moves)
    current = list(order)
    moves: list[Move] = []
    swapped: set[int] = set()  # a slot moves at most once per night
    while len(moves) < cap:
        best: tuple[float, int] | None = None
        for i in range(len(current) - 1):
            hi, lo = by_model[current[i]], by_model[current[i + 1]]
            if hi.pinned or lo.pinned or i in swapped or i + 1 in swapped:
                continue
            gap = lo.score - hi.score
            if gap > s.margin and (best is None or gap > best[0]):
                best = (gap, i)
        if best is None:
            break
        gap, i = best
        moves.append(Move(current[i + 1], current[i], i, gap))
        current[i], current[i + 1] = current[i + 1], current[i]
        swapped.update((i, i + 1))
    return TierResult(tier, order, tuple(current), tuple(scores), tuple(moves), cap, agentic_share)


def agentic_share(samples: list[Sample], tier: str) -> float:
    in_tier = [x for x in samples if x.tier == tier]
    return sum(x.agentic for x in in_tier) / len(in_tier) if in_tier else 0.0
