"""Replay sampled requests on the other models of their tier and judge them."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
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
from pinball.optimize.settings import CALLER, JUDGE_CALLER, PriceTable

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
    # Content-free reason a judge gave no verdict: budget | unparseable | gateway label.
    judge_errors: dict[str, str] = field(default_factory=dict)
    cost_usd: float = 0.0  # reported usage at estimated prices, candidate plus judges

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
            "judge_errors": dict(self.judge_errors),
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
    model_prefix: str = ""  # gateway alias prefix for candidates only, never judges

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
    *,
    model_prefix: str = "",
) -> Plan:
    """Each sample against every other model of its tier (optionally restricted to `only`).

    Pairs are interleaved round-robin over (tier, candidate): every (tier, candidate)
    gets its first pair before any gets its second, so a budget cut keeps breadth.
    """
    groups: dict[tuple[str, str], list[tuple[Sample, str]]] = {}
    for tier, models in tiers.items():
        for model in models:
            if only is None or model in only:
                groups[tier, model] = []
    for s in samples:
        for model in tiers.get(s.tier, ()):
            if model != s.served_model and (s.tier, model) in groups:
                groups[s.tier, model].append((s, model))
    lanes = [g for g in groups.values() if g]
    depth = max((len(g) for g in lanes), default=0)
    pairs = [lane[i] for i in range(depth) for lane in lanes if i < len(lane)]
    return Plan(tuple(pairs), judges, seed, model_prefix)


def pair_worst_cost(pair: tuple[Sample, str], judges: tuple[str, ...], prices: PriceTable) -> float:
    """Worst case of one pair: the candidate call plus every judge call at max_tokens."""
    sample, model = pair
    messages = list(sample.messages)
    cost = prices.cost(model, input_token_bound(messages), CANDIDATE_MAX_TOKENS)
    # The judge sees the conversation plus two answers; bound the candidate answer by
    # its max_tokens and the served answer by its text.
    judge_in = (
        input_token_bound(messages)
        + CANDIDATE_MAX_TOKENS
        + input_token_bound([{"content": sample.served_text}])
    )
    return cost + sum(prices.cost(j, judge_in, JUDGE_MAX_TOKENS) for j in judges)


def projected_cost(plan: Plan, budget: Budget) -> tuple[float, int]:
    """Worst-case cost of the whole plan and how many pairs fit fully under the cap."""
    total, fitting = 0.0, 0
    for pair in plan.pairs:
        total += pair_worst_cost(pair, plan.judges, budget.prices)
        if total <= budget.cap:
            fitting += 1
    return total, fitting


def empty_result(sample: Sample, model: str, plan: Plan, error: str | None = None) -> PairResult:
    a_served = a_is_served(plan.seed, sample.log_id, model)
    return PairResult(
        sample.log_id,
        sample.tier,
        sample.agentic,
        sample.served_model,
        model,
        "served" if a_served else "candidate",
        error=error,
        same_family=same_family_judges(model, sample.served_model, plan.judges),
    )


def replay_pair(
    sample: Sample,
    model: str,
    plan: Plan,
    gateway: Gateway,
    budget: Budget,
    *,
    cancel: threading.Event | None = None,
) -> PairResult:
    """One comparison; `cancel` (set after the pair's deadline) stops further judge calls."""
    a_served = a_is_served(plan.seed, sample.log_id, model)
    result = empty_result(sample, model, plan)
    try:
        gen = metered(
            gateway,
            budget,
            model,
            [dict(m) for m in sample.messages],
            max_tokens=CANDIDATE_MAX_TOKENS,
            caller=CALLER,
            wire_model=plan.model_prefix + model,
        )
    except BudgetExhausted:
        result.error = "budget"
        return result
    except GatewayError as e:
        result.error = e.label
        return result
    result.latency_s = round(gen.latency_s, 3)
    result.input_tokens, result.output_tokens = gen.input_tokens, gen.output_tokens
    result.cost_usd += budget.prices.cost(model, gen.input_tokens, gen.output_tokens)
    a, b = (sample.served_text, gen.text) if a_served else (gen.text, sample.served_text)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_prompt(sample.messages, a, b)},
    ]
    for judge in plan.judges:
        if cancel is not None and cancel.is_set():
            result.judge_errors[judge] = "cancelled"
            result.verdicts[judge] = None
            continue
        verdict, failure = _judge(judge, messages, a_served, gateway, budget, result)
        result.verdicts[judge] = verdict
        if failure is not None:
            result.judge_errors[judge] = failure
    return result


def _judge(
    judge: str,
    messages: list[dict[str, str]],
    a_served: bool,
    gateway: Gateway,
    budget: Budget,
    result: PairResult,
) -> tuple[Verdict | None, str | None]:
    """A verdict, or None with the content-free label of the last failed attempt."""
    failure = "budget"
    for _ in range(JUDGE_ATTEMPTS):
        try:
            answer = metered(
                gateway, budget, judge, messages, max_tokens=JUDGE_MAX_TOKENS, caller=JUDGE_CALLER
            )
        except BudgetExhausted:
            return None, "budget"
        except GatewayError as e:
            failure = e.label
            continue
        result.cost_usd += budget.prices.cost(judge, answer.input_tokens, answer.output_tokens)
        verdict = parse_judgement(answer.text, a_served)
        if verdict is not None:
            return verdict, None
        failure = "unparseable"
    return None, failure
