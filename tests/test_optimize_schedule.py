from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any

import pytest

from pinball.optimize.gateway import Budget, Completion, GatewayError
from pinball.optimize.replay import PairResult, make_plan
from pinball.optimize.report import outcome_counts
from pinball.optimize.sample import Sample
from pinball.optimize.schedule import Limits, run_replay
from pinball.optimize.settings import Price, PriceTable, settings_from_env

JUDGES = ("judge-opus", "judge-argon")
TIERS = {"light": ("l1",), "standard": ("s1", "s2", "s3"), "heavy": ("h1", "h2", "h3")}
VERDICT = '{"a_acceptable": true, "b_acceptable": true, "preference": "tie"}'


def sample(i: int, tier: str, served: str) -> Sample:
    return Sample(
        log_id=f"{tier}-{i}",
        tier=tier,
        reason=None,
        pinball_model=served,
        served_model=served,
        agentic=False,
        served_input_tokens=10,
        served_output_tokens=10,
        messages=({"role": "user", "content": f"question {i}"},),
        served_text=f"answer {i}",
    )


def samples(n: int = 10) -> list[Sample]:
    # Tier by tier, heavy first: the old plan order let heavy drain the cap.
    return [sample(i, "heavy", "h1") for i in range(n)] + [
        sample(i, "standard", "s1") for i in range(n)
    ]


def prices(heavy: float = 1.0) -> PriceTable:
    models = {m: Price(input=heavy, output=heavy) for m in ("h1", "h2", "h3")}
    return PriceTable(models=models, default=Price(input=1.0, output=1.0))


class Gw:
    """Reports usage of max_tokens output; `fail` maps a model to a scripted failure."""

    def __init__(self, fail: dict[str, Any] | None = None) -> None:
        self.fail = fail or {}
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def chat(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int, caller: str
    ) -> Completion:
        with self.lock:
            self.calls.append((model, caller))
        action = self.fail.get(model)
        if isinstance(action, threading.Event):
            action.wait()
        elif isinstance(action, str):
            raise GatewayError(action)
        text = "not json" if action is False else (VERDICT if model in JUDGES else "cand")
        return Completion(text, 0, max_tokens, 0.0)

    def candidates(self) -> Counter[str]:
        return Counter(m for m, c in self.calls if c == "eval-replay")


def test_plan_interleaves_every_tier_and_candidate_before_depth():
    plan = make_plan(samples(3), TIERS, JUDGES, 1)
    lanes = [(s.tier, m) for s, m in plan.pairs]
    assert len(set(lanes[:4])) == 4  # h2 h3 s2 s3, then the second round
    assert set(lanes[:4]) == set(lanes)


def test_tight_cap_gives_every_tier_and_model_a_pair_before_any_gets_two():
    gw = Gw()
    budget = Budget(0.09, prices(heavy=2.0))
    run = run_replay(make_plan(samples(), TIERS, JUDGES, 1), gw, budget, Limits(1, 60.0))
    issued = gw.candidates()
    assert set(issued) == {"h2", "h3", "s2", "s3"}
    assert budget.spent <= budget.cap
    order = [m for m, c in gw.calls if c == "eval-replay"]
    assert set(order[:4]) == {"h2", "h3", "s2", "s3"}
    assert outcome_counts(run.results)["budget_refused"] > 0


def test_expensive_tier_cannot_exceed_its_share_while_others_have_pairs():
    gw = Gw()
    # Both tiers want more than their $0.25: heavy pairs cost ~$0.2, standard ~$0.02.
    budget = Budget(0.5, prices(heavy=10.0))
    many = samples(10) + [sample(i, "standard", "s1") for i in range(10, 40)]
    run = run_replay(make_plan(many, TIERS, JUDGES, 1), gw, budget, Limits(2, 60.0))
    heavy = run.shares["heavy"]
    assert heavy["share_usd"] == pytest.approx(0.25)
    assert 0 < heavy["spent_usd"] <= 0.25
    assert run.shares["standard"]["spent_usd"] > 0.25  # heavy's leftover rolled over
    assert outcome_counts(run.results)["budget_refused"] > 0
    assert budget.spent <= budget.cap


def test_unused_share_rolls_over_to_a_tier_with_pairs_left():
    gw = Gw()
    budget = Budget(0.2, prices())
    plan = make_plan(
        [sample(0, "heavy", "h1")] + [sample(i, "standard", "s1") for i in range(20)],
        {"heavy": ("h1", "h2"), "standard": ("s1", "s2")},
        JUDGES,
        1,
    )
    run = run_replay(plan, gw, budget, Limits(1, 60.0))
    assert run.shares["standard"]["share_usd"] == pytest.approx(0.1)
    assert run.shares["standard"]["spent_usd"] > 0.1
    assert budget.spent <= budget.cap


def test_weights_split_the_cap():
    run = run_replay(
        make_plan(samples(1), TIERS, JUDGES, 1),
        Gw(),
        Budget(1.0, prices()),
        Limits(1, 60.0, {"heavy": 3.0}),
    )
    assert run.shares["heavy"]["share_usd"] == pytest.approx(0.75)


def test_budget_refusal_is_not_an_error_and_errors_are_labelled():
    def r(cand: str, error: str | None = None, **judge: str) -> PairResult:
        p = PairResult("id", "heavy", False, "h1", cand, "served", error=error)
        p.judge_errors = dict(judge)
        return p

    out = outcome_counts(
        [
            r("h2", "budget"),
            r("h2", "budget"),
            r("kimi-k3", "HTTP 504"),
            r("kimi-k3", "timeout"),
            r("h3", **{"judge-opus": "unparseable", "judge-argon": "HTTP 500"}),
            r("h3", **{"judge-opus": "budget", "judge-argon": "cancelled"}),
        ]
    )
    assert out["budget_refused"] == 2 and out["judge_budget_refused"] == 1
    assert out["gateway_errors"] == {"kimi-k3": {"HTTP 504": 1, "timeout": 1}}
    assert out["judge_unparseable"] == {"judge-opus": 1}
    assert out["judge_failed"] == {"judge-argon": {"HTTP 500": 1}}
    assert out["errors"] == 4


def test_replay_records_gateway_and_judge_failure_labels():
    gw = Gw({"h2": "HTTP 504", "judge-opus": False, "judge-argon": "transport TimeoutError"})
    run = run_replay(
        make_plan(samples(1), TIERS, JUDGES, 1), gw, Budget(10.0, prices()), Limits(2, 60.0)
    )
    out = outcome_counts(run.results)
    assert out["gateway_errors"] == {"h2": {"HTTP 504": 1}}
    assert out["judge_unparseable"] == {"judge-opus": 3}
    assert out["judge_failed"] == {"judge-argon": {"transport TimeoutError": 3}}
    assert out["budget_refused"] == 0


def test_hung_pair_hits_its_deadline_without_holding_the_run():
    hang = threading.Event()
    gw = Gw({"h2": hang})
    budget = Budget(10.0, prices())
    started = time.monotonic()
    try:
        run = run_replay(make_plan(samples(2), TIERS, JUDGES, 1), gw, budget, Limits(2, 0.3))
    finally:
        hang.set()
    assert time.monotonic() - started < 5
    out = outcome_counts(run.results)
    assert out["gateway_errors"] == {"h2": {"timeout": 2}}
    assert out["pairs_judged"] == len(run.results) - 2
    assert budget.spent <= budget.cap


def test_settings_defaults_and_overrides():
    s = settings_from_env({})
    assert (s.budget_usd, s.pair_deadline, s.tier_weights) == (20.0, 180.0, {})
    s = settings_from_env(
        {"PINBALL_OPT_TIER_WEIGHTS": "heavy=2, light=0.5", "PINBALL_OPT_PAIR_DEADLINE": "90"}
    )
    assert s.tier_weights == {"heavy": 2.0, "light": 0.5} and s.pair_deadline == 90.0
