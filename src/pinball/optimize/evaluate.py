"""Step B: replay candidates against each tier's incumbents and pick a tier per winner.

Every sample of a tier is replayed on the candidate (as `<prefix><id>`, its
public evaluation alias) and judged against the answer the pool served. The
served side of each pair is evidence for the incumbent, so the comparison is
head to head. Per tier the candidate is scored like a model appended at the
LAST position (lowest prior, cost penalty from the price table, which prices
unknown ids like the most expensive tier) and it WINS the tier when

  * it has at least min_samples non-agentic agreed comparisons, and
  * its score beats the weakest incumbent that has agreed evidence by `margin`.

Only fields every judge agreed on count (rank.collect). A candidate that wins
several tiers is assigned the one with the largest gap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pinball.optimize.carrier_edit import ModelId
from pinball.optimize.gateway import Budget, Gateway
from pinball.optimize.rank import ModelScore, collect, score_tier
from pinball.optimize.replay import make_plan
from pinball.optimize.sample import Sample
from pinball.optimize.schedule import Limits, run_replay
from pinball.optimize.settings import Settings


@dataclass(frozen=True, slots=True)
class TierTrial:
    tier: str
    candidate: ModelScore
    weakest: ModelScore | None
    wins: bool

    @property
    def gap(self) -> float:
        return self.candidate.score - self.weakest.score if self.weakest else 0.0

    def record(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "candidate": self.candidate.record(),
            "weakest_incumbent": self.weakest.record() if self.weakest else None,
            "gap": round(self.gap, 6),
            "wins": self.wins,
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    model: ModelId
    trials: tuple[TierTrial, ...]

    @property
    def tier(self) -> str | None:
        winners = sorted((t for t in self.trials if t.wins), key=lambda t: -t.gap)
        return winners[0].tier if winners else None

    def record(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tier": self.tier,
            "trials": [t.record() for t in self.trials],
        }


def judge_tier(
    tier: str, incumbents: tuple[str, ...], candidate: ModelId, evidence: Any, s: Settings
) -> TierTrial:
    scores = score_tier((*incumbents, candidate), evidence, s.rank)
    cand = scores[-1]
    rated = [x for x in scores[:-1] if x.n > 0]
    weakest = min(rated, key=lambda x: x.score) if rated else None
    wins = (
        weakest is not None
        and cand.non_agentic >= s.rank.min_samples
        and cand.score > weakest.score + s.rank.margin
    )
    return TierTrial(tier, cand, weakest, wins)


def evaluate(
    samples: Sequence[Sample],
    order: dict[str, tuple[str, ...]],
    candidates: Sequence[ModelId],
    deps: tuple[Gateway, Budget],
    s: Settings,
    seed: int,
) -> list[Verdict]:
    """One replay of every sample on every candidate; evidence per tier per candidate."""
    if not candidates:
        return []
    gateway, budget = deps
    trial_tiers: dict[str, tuple[str, ...]] = {t: tuple(candidates) for t in order}
    plan = make_plan(samples, trial_tiers, s.judges, seed, model_prefix=s.model_prefix)
    # Its own Budget (PINBALL_OPT_CANDIDATE_BUDGET_USD), never the ranking one.
    limits = Limits(s.workers, s.pair_deadline, s.tier_weights)
    results = run_replay(plan, gateway, budget, limits).results
    verdicts: list[Verdict] = []
    for c in candidates:
        mine = [r for r in results if r.candidate == c]
        evidence = collect(mine, list(samples), s.prices)
        trials = tuple(
            judge_tier(t, order[t], c, evidence.get(t, {}), s) for t in order if order[t]
        )
        verdicts.append(Verdict(c, trials))
    return verdicts
