"""Replay sampled requests on the other models of their tier and judge them."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Final

from pinball.optimize.gateway import (
    Budget,
    BudgetExhausted,
    Gateway,
    GatewayError,
    input_token_bound,
    metered,
)
from pinball.optimize.judge import (
    JUDGE_SYSTEM,
    Agreed,
    Verdict,
    a_is_served,
    agree,
    judge_prompt,
    parse_judgement,
    same_family_judges,
)
from pinball.optimize.sample import Sample
from pinball.optimize.settings import CALLER, JUDGE_CALLER

log = logging.getLogger("pinball.optimize")

CANDIDATE_MAX_TOKENS: Final = 6000
JUDGE_MAX_TOKENS: Final = 1000
JUDGE_ATTEMPTS: Final = 2  # one retry when the judge answer cannot be parsed


@dataclass
class PairResult:
    """Content-free outcome of one (sample, candidate) comparison."""

    log_id: str
    tier: str
    agentic: bool
    served_model: str | None
    candidate: str
    a_is: str  # which answer the judges saw as A: served | candidate
    error: str | None = None
    latency_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    same_family: tuple[str, ...] = ()
    verdicts: dict[str, Verdict | None] = field(default_factory=dict)

    def view(self, independent: bool = False) -> Agreed | None:
        judges = [j for j in self.verdicts if not (independent and j in self.same_family)]
        return agree([self.verdicts[j] for j in judges]) if judges else None

    def record(self) -> dict[str, Any]:
        def agreed(a: Agreed | None) -> dict[str, Any] | None:
            if a is None:
                return None
            return {"candidate_ok": a.candidate_ok, "served_ok": a.served_ok, "outcome": a.outcome}

        return {
            "log_id": self.log_id,
            "tier": self.tier,
            "agentic": self.agentic,
            "served_model": self.served_model,
            "candidate": self.candidate,
            "a_is": self.a_is,
            "error": self.error,
            "latency_s": self.latency_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "same_family": list(self.same_family),
            "verdicts": {
                j: None
                if v is None
                else {
                    "served_ok": v.served_ok,
                    "candidate_ok": v.candidate_ok,
                    "outcome": v.outcome,
                    "reason": v.reason,
                }
                for j, v in self.verdicts.items()
            },
            "agreed": agreed(self.view()),
            "independent": agreed(self.view(True)) if self.same_family else None,
        }


@dataclass(frozen=True, slots=True)
class Plan:
    pairs: tuple[tuple[Sample, str], ...]
    judges: tuple[str, ...]
    seed: int

    @property
    def candidate_calls(self) -> int:
        return len(self.pairs)

    @property
    def judge_calls(self) -> int:
        return len(self.pairs) * len(self.judges)


def make_plan(
    samples: Sequence[Sample],
    tiers: dict[str, tuple[str, ...]],
    judges: tuple[str, ...],
    seed: int,
    only: frozenset[str] | None = None,
) -> Plan:
    """Each sample against every other model of its tier (optionally restricted to `only`)."""
    pairs: list[tuple[Sample, str]] = []
    for s in samples:
        for model in tiers.get(s.tier, ()):
            if model == s.served_model or (only is not None and model not in only):
                continue
            pairs.append((s, model))
    return Plan(tuple(pairs), judges, seed)


def projected_cost(plan: Plan, budget: Budget) -> tuple[float, int]:
    """Worst-case cost of the whole plan and how many pairs fit fully under the cap."""
    total, fitting = 0.0, 0
    for sample, model in plan.pairs:
        messages = list(sample.messages)
        pair = budget.prices.cost(model, input_token_bound(messages), CANDIDATE_MAX_TOKENS)
        # The judge sees the conversation plus two answers; bound the candidate answer by
        # its max_tokens and the served answer by its text.
        judge_in = (
            input_token_bound(messages)
            + CANDIDATE_MAX_TOKENS
            + input_token_bound([{"content": sample.served_text}])
        )
        for j in plan.judges:
            pair += budget.prices.cost(j, judge_in, JUDGE_MAX_TOKENS)
        total += pair
        if total <= budget.cap:
            fitting += 1
    return total, fitting


def replay_pair(
    sample: Sample, model: str, plan: Plan, gateway: Gateway, budget: Budget
) -> PairResult:
    a_served = a_is_served(plan.seed, sample.log_id, model)
    result = PairResult(
        sample.log_id,
        sample.tier,
        sample.agentic,
        sample.served_model,
        model,
        "served" if a_served else "candidate",
        same_family=same_family_judges(model, sample.served_model, plan.judges),
    )
    try:
        gen = metered(
            gateway,
            budget,
            model,
            [dict(m) for m in sample.messages],
            max_tokens=CANDIDATE_MAX_TOKENS,
            caller=CALLER,
        )
    except BudgetExhausted:
        result.error = "budget"
        return result
    except GatewayError as e:
        result.error = e.label
        return result
    result.latency_s = round(gen.latency_s, 3)
    result.input_tokens, result.output_tokens = gen.input_tokens, gen.output_tokens
    a, b = (sample.served_text, gen.text) if a_served else (gen.text, sample.served_text)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_prompt(sample.messages, a, b)},
    ]
    for judge in plan.judges:
        verdict: Verdict | None = None
        for _ in range(JUDGE_ATTEMPTS):
            try:
                answer = metered(
                    gateway,
                    budget,
                    judge,
                    messages,
                    max_tokens=JUDGE_MAX_TOKENS,
                    caller=JUDGE_CALLER,
                )
            except BudgetExhausted:
                break
            except GatewayError:
                continue
            verdict = parse_judgement(answer.text, a_served)
            if verdict is not None:
                break
        result.verdicts[judge] = verdict
    return result


def run_replay(plan: Plan, gateway: Gateway, budget: Budget, workers: int) -> list[PairResult]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(replay_pair, s, m, plan, gateway, budget) for s, m in plan.pairs]
        results = [f.result() for f in futures]
    log.info(
        "replay done pairs=%d errors=%d spent=%.4f calls=%d exhausted=%s",
        len(results),
        sum(r.error is not None for r in results),
        budget.spent,
        budget.calls,
        budget.exhausted,
    )
    return results
