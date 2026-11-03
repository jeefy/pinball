from __future__ import annotations

import email.message
import io
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from pinball.config import Config, Tier
from pinball.optimize import cli
from pinball.optimize.apply import (
    State,
    build_overlay,
    load_state,
    next_version,
    save_state,
    write_overlay,
)
from pinball.optimize.db import FakeDb, Window
from pinball.optimize.gateway import (
    Budget,
    BudgetExhausted,
    Completion,
    GatewayError,
    HttpGateway,
    metered,
)
from pinball.optimize.judge import Verdict, agree, parse_judgement, same_family_judges
from pinball.optimize.rank import Evidence, rerank
from pinball.optimize.replay import PairResult
from pinball.optimize.revert import judge_windows
from pinball.optimize.sample import draw_samples, normalize_payload
from pinball.optimize.settings import Price, PriceTable, RankSettings, RevertSettings
from pinball.overlay import Overlay, OverlayRejected, load_overlay

BASE_DOC: dict[str, Any] = {
    "models": {
        "l1": {"prompt_tokens": 100000},
        "l2": {"prompt_tokens": 100000},
        "s1": {"prompt_tokens": 100000},
        "s2": {"prompt_tokens": 100000},
        "s3": {"prompt_tokens": 100000},
        "h1": {"prompt_tokens": 100000},
    },
    "tiers": {"light": ["l1", "l2"], "standard": ["s1", "s2", "s3"], "heavy": ["h1"]},
    "thresholds": {"standard": 1, "heavy": 3},
}
BASE = Config.model_validate(BASE_DOC)
NOW = datetime(2026, 11, 3, 3, 0, tzinfo=UTC)
SECRET = "SECRET-PROMPT-TEXT"
FLAT = PriceTable(models={}, default=Price(input=1.0, output=1.0))  # $1 per M tokens


def row(i: int, tier: str = "standard", served: str = "s1", **kw: Any) -> dict[str, Any]:
    return {
        "id": f"{i:08d}-0000",
        "tier": tier,
        "reason": "score",
        "pinball_model": served,
        "served_model": served,
        "input_tokens": 100,
        "output_tokens": 50,
        "request": kw.pop("request", [{"role": "user", "content": f"{SECRET} {i}"}]),
        "response": kw.pop("response", [f"served answer {i}"]),
        **kw,
    }


# --- sampling and normalisation --------------------------------------------------------


def test_sampling_is_deterministic_by_seed():
    db = FakeDb([row(i, tier) for i in range(40) for tier in ("light", "standard")])
    db.rows = [dict(r, id=f"{n:08d}-x") for n, r in enumerate(db.rows)]

    def ids(seed: int) -> list[str]:
        return [s.log_id for s in draw_samples(db, 1, 5, seed)]

    assert ids(7) == ids(7)
    assert ids(7) != ids(8)
    assert len(ids(7)) == 10


def test_sampling_skips_unreplayable_rows_and_fills_quota():
    rows = [row(i, response=[]) for i in range(3)] + [row(i) for i in range(3, 10)]
    samples = draw_samples(FakeDb(rows), 1, 4, 0)
    assert len(samples) == 4
    assert all(s.served_text for s in samples)


def test_normalisation_drops_tool_turns_and_marks_agentic():
    s = normalize_payload(
        row(
            1,
            request=[
                {"role": "system", "content": "sys"},
                {"role": "developer", "content": "dev"},
                {"role": "user", "content": [{"type": "text", "text": "q1"}]},
                {"role": "assistant", "content": ""},
                {"role": "tool", "content": "tool output"},
                {"role": "user", "content": "q2"},
            ],
        )
    )
    assert s is not None and s.agentic
    assert [m["role"] for m in s.messages] == ["system", "user"]
    assert s.messages[0]["content"] == "sys\n\ndev"
    assert "tool output" not in json.dumps(s.messages)
    plain = normalize_payload(row(2))
    assert plain is not None and not plain.agentic
    assert normalize_payload(row(3, request=[{"role": "assistant", "content": "x"}])) is None
    assert "SECRET" not in json.dumps(plain.record())


# --- judging ---------------------------------------------------------------------------


def j(a_ok: bool, b_ok: bool, pref: str) -> str:
    return json.dumps({"a_acceptable": a_ok, "b_acceptable": b_ok, "preference": pref})


def test_parse_judgement_maps_ab_back_to_served_and_candidate():
    v = parse_judgement("```" + j(True, False, "a") + "```", a_served=False)
    assert v == Verdict(served_ok=False, candidate_ok=True, outcome="win", reason="")
    served_a = parse_judgement(j(True, False, "a"), a_served=True)
    assert served_a is not None and served_a.outcome == "loss"
    assert parse_judgement("no json", True) is None
    assert parse_judgement(j(True, True, "maybe"), True) is None


def test_agreement_counts_fields_only_when_judges_agree():
    win = Verdict(True, True, "win", "")
    loss = Verdict(True, False, "loss", "")
    a = agree([win, loss])
    assert a is not None
    assert (a.served_ok, a.candidate_ok, a.outcome) == (True, None, None)
    assert agree([win, None]) is None
    assert agree([win, win]) == agree([win])
    assert same_family_judges("gemini-x", "claude-opus-5", ("judge-opus", "judge-argon")) == (
        "judge-argon",
        "judge-opus",
    )


# --- budget ----------------------------------------------------------------------------


class FakeGateway:
    """Reports token usage; judges prefer the candidate when `better` says so."""

    def __init__(self, better: set[str] | None = None, usage: tuple[int, int] | None = None):
        self.better = better or set()
        self.usage = usage
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def chat(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int, caller: str
    ) -> Completion:
        with self.lock:
            self.calls.append((model, caller))
        tin, tout = self.usage or (sum(len(m["content"]) for m in messages) // 4, max_tokens // 2)
        if caller == "eval-replay":
            return Completion(f"CAND {model}", tin, tout, 0.1)
        prompt = messages[-1]["content"]
        a = prompt.split("ANSWER A:\n")[1].split("\n\nANSWER B:")[0]
        cand_is_a = a.startswith("CAND ")
        cand = (a if cand_is_a else prompt.split("ANSWER B:\n")[1]).split()[1]
        good = cand in self.better
        pref = ("a" if cand_is_a else "b") if good else ("b" if cand_is_a else "a")
        return Completion(j(True, True, pref), tin, tout, 0.1)


def test_budget_never_exceeds_cap_with_reported_usage():
    budget = Budget(0.05, FLAT)
    gw = FakeGateway(usage=(100, 2_000))  # $0.0021 reported per call
    msgs = [{"role": "user", "content": "x" * 400}]
    issued = 0
    for _ in range(100):
        try:
            metered(gw, budget, "m", msgs, max_tokens=4000, caller="eval-replay")
        except BudgetExhausted:
            continue
        issued += 1
        assert budget.spent <= budget.cap
    # Reserve per call (216 + 4000) / 1e6 = $0.004216; call k+1 fits while
    # 0.0021 * k + 0.004216 <= 0.05, i.e. k <= 21: exactly 22 calls.
    assert issued == 22 == len(gw.calls) == budget.calls
    assert budget.spent == pytest.approx(22 * 0.0021)
    assert budget.exhausted


def test_budget_closes_when_reported_usage_overruns_reservation():
    budget = Budget(10.0, FLAT)
    gw = FakeGateway(usage=(5_000_000, 0))  # $5: far above the worst-case reservation
    msgs = [{"role": "user", "content": "hi"}]
    metered(gw, budget, "m", msgs, max_tokens=10, caller="eval-replay")
    assert budget.overrun and budget.exhausted
    assert budget.reserve("m", msgs, 10) is None
    assert len(gw.calls) == 1


def test_failed_call_releases_reservation():
    class Down:
        def chat(self, *a: Any, **k: Any) -> Completion:
            raise GatewayError("HTTP 502")

    budget = Budget(1.0, FLAT)
    with pytest.raises(GatewayError):
        metered(Down(), budget, "m", [{"content": "x"}], max_tokens=10, caller="c")
    assert (budget.spent, budget.reserved, budget.calls) == (0.0, 0.0, 0)


def test_http_gateway_strips_error_bodies_and_reads_usage():
    import urllib.error

    sent: list[Any] = []

    class Resp(io.BytesIO):
        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: object) -> None:
            pass

    def ok(req: Any, timeout: float) -> Resp:
        sent.append(req)
        body = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3}}
        return Resp(json.dumps(body).encode())

    def bad(req: Any, timeout: float) -> Resp:
        raise urllib.error.HTTPError(
            req.full_url, 500, SECRET, email.message.Message(), io.BytesIO(SECRET.encode())
        )

    done = HttpGateway("http://gw/", opener=ok).chat("m", [], max_tokens=5, caller="eval-replay")
    assert (done.text, done.input_tokens, done.output_tokens) == ("hi", 3, 0)
    assert sent[0].full_url == "http://gw/v1/chat/completions"
    assert sent[0].get_header("X-agw-caller") == "eval-replay"
    with pytest.raises(GatewayError) as e:
        HttpGateway("http://gw", opener=bad).chat("m", [], max_tokens=5, caller="c")
    assert e.value.label == "HTTP 500" and SECRET not in str(e.value)


# --- ranking ---------------------------------------------------------------------------


def ev(n: int, quality: float, cost: float = 1.0, agentic: int = 0) -> Evidence:
    e = Evidence()
    for i in range(n):
        good = i < round(n * quality)
        e.add(good, 1.0 if good else 0.0, agentic=i < agentic)
    e.costs = [cost]
    return e


ORDER = ("s1", "s2", "s3")
FAST = RankSettings(min_samples=5, shrink_k=0, prior_step=0.0, cost_weight=0.0, margin=0.0)


def test_rank_promotes_better_model_with_enough_samples():
    r = rerank("standard", ORDER, {"s1": ev(10, 0.2), "s2": ev(10, 0.9)}, 0.0, FAST)
    assert r.after == ("s2", "s1", "s3")
    assert [m.model for m in r.moves] == ["s2"]


def test_rank_min_samples_pins_model():
    r = rerank("standard", ORDER, {"s1": ev(10, 0.2), "s2": ev(4, 1.0)}, 0.0, FAST)
    assert r.after == ORDER
    assert {s.model: s.pinned for s in r.scores}["s2"]


def test_rank_shrinkage_keeps_current_order_on_small_gap():
    s = RankSettings(min_samples=5, shrink_k=20, prior_step=0.05, cost_weight=0.0, margin=0.0)
    evidence = {"s1": ev(10, 0.6), "s2": ev(10, 0.7)}
    assert rerank("standard", ORDER, evidence, 0.0, s).after == ORDER
    big = {"s1": ev(200, 0.3), "s2": ev(200, 0.9)}
    assert rerank("standard", ORDER, big, 0.0, s).after[0] == "s2"
    score = {x.model: x for x in rerank("standard", ORDER, evidence, 0.0, s).scores}["s2"]
    assert score.shrunk == pytest.approx((10 * 0.7 + 20 * 0.70) / 30)


def test_rank_cost_penalty_can_flip_order():
    s = FAST.model_copy(update={"cost_weight": 0.5})
    evidence = {"s1": ev(10, 0.8, cost=4.0), "s2": ev(10, 0.7, cost=1.0), "s3": ev(10, 0.7, 1.0)}
    assert rerank("standard", ORDER, evidence, 0.0, s).after[0] == "s2"


def test_rank_respects_max_moves_and_agentic_cap():
    order = ("a", "b", "c", "d", "e")
    evidence = {m: ev(10, q) for m, q in zip(order, (0.1, 0.9, 0.1, 0.9, 0.5), strict=True)}
    s = FAST.model_copy(update={"max_moves": 2, "agentic_max_moves": 1})
    assert len(rerank("t", order, evidence, 0.0, s).moves) == 2
    assert len(rerank("t", order, evidence, 0.9, s).moves) == 1
    assert len(rerank("t", order, evidence, 0.0, s.model_copy(update={"max_moves": 0})).moves) == 0


def test_rank_agentic_samples_do_not_count_toward_min_samples():
    evidence = {"s1": ev(10, 0.1), "s2": ev(10, 1.0, agentic=8)}
    assert rerank("standard", ORDER, evidence, 0.0, FAST).after == ORDER


# --- apply -----------------------------------------------------------------------------


def test_overlay_built_only_when_merge_validates(tmp_path: Path):
    good = build_overlay(BASE, None, {"standard": ("s2", "s1", "s3")}, "opt-000001-x", NOW)
    with pytest.raises(OverlayRejected):
        build_overlay(BASE, None, {"standard": ("s2", "h1", "s3")}, "v", NOW)
    p = tmp_path / "o.yaml"
    write_overlay(p, BASE, good)
    assert load_overlay(p).tier_order[Tier.STANDARD] == ("s2", "s1", "s3")
    assert [x.name for x in tmp_path.iterdir()] == ["o.yaml"]  # no temp file left


def test_overlay_write_is_atomic_and_keeps_old_file_on_failure(tmp_path: Path, monkeypatch):
    p = tmp_path / "o.yaml"
    p.write_text("old")

    def boom(*a: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("pinball.optimize.apply.os.replace", boom)
    with pytest.raises(OSError):
        write_overlay(p, BASE, build_overlay(BASE, None, {}, "v", NOW))
    assert p.read_text() == "old"
    assert [x.name for x in tmp_path.iterdir()] == ["o.yaml"]


def test_versions_are_monotonic(tmp_path: Path):
    st = State(seq=4)
    seq, v = next_version(st, None, NOW)
    assert (seq, v) == (5, "opt-000005-20261103T030000Z")
    live = Overlay.model_validate({"version": "opt-000009-x", "generated_at": NOW})
    assert next_version(st, live, NOW)[0] == 10
    save_state(tmp_path / "s.json", State(seq=10))
    assert load_state(tmp_path / "s.json").seq == 10


# --- revert ----------------------------------------------------------------------------

RS = RevertSettings(min_requests=10)


@pytest.mark.parametrize(
    ("before", "after", "revert"),
    [
        (Window(100, 2, 1.0, 100), Window(100, 10, 1.0, 100), True),  # errors 2% -> 10%
        (Window(100, 2, 1.0, 100), Window(100, 2, 1.5, 100), True),  # cost +50%
        (Window(100, 2, 1.0, 100), Window(100, 3, 1.1, 100), False),  # within margins
        (Window(100, 0, 1.0, 100), Window(100, 1, 1.0, 100), False),  # below error floor
        (Window(5, 0, 1.0, 5), Window(5, 5, 9.0, 5), False),  # too little traffic
        (Window(100, 2, 0.0, 0), Window(100, 2, 5.0, 100), False),  # before unpriced
    ],
)
def test_revert_triggers(before: Window, after: Window, revert: bool):
    assert judge_windows(before, after, RS).revert is revert


# --- CLI end to end with fakes ---------------------------------------------------------


def env_for(tmp_path: Path, **extra: str) -> dict[str, str]:
    cfg = tmp_path / "pinball.yaml"
    cfg.write_text(yaml.safe_dump(BASE_DOC))
    prices = tmp_path / "prices.yaml"
    prices.write_text(yaml.safe_dump({"default": {"input": 1.0, "output": 1.0}}))
    return {
        "PINBALL_OPT_DB_DSN": "postgresql://fake",
        "PINBALL_OPT_GATEWAY_URL": "http://fake",
        "PINBALL_CONFIG": str(cfg),
        "PINBALL_OVERLAY_PATH": str(tmp_path / "overlay.yaml"),
        "PINBALL_OPT_PRICES": str(prices),
        "PINBALL_OPT_MIN_SAMPLES": "5",
        "PINBALL_OPT_SHRINK_K": "2",
        "PINBALL_OPT_COST_WEIGHT": "0",
        **extra,
    }


def traffic() -> FakeDb:
    return FakeDb([row(i) for i in range(30)])


def run(tmp_path: Path, argv: list[str], db: FakeDb, gw: FakeGateway | None, **env: str):
    out = io.StringIO()

    def gateway_factory(_: Any) -> FakeGateway:
        assert gw is not None, "gateway must not be built"
        return gw

    code = cli.main(argv, env_for(tmp_path, **env), lambda _: db, gateway_factory, NOW, out)
    return code, out.getvalue()


def test_dry_run_makes_zero_gateway_calls(tmp_path: Path):
    code, out = run(tmp_path, ["run", "--dry-run", "--per-tier", "10"], traffic(), None)
    plan = json.loads(out)
    assert code == 0 and plan["dry_run"]
    assert plan["candidate_calls"] == 20 and plan["judge_calls"] == 40
    assert plan["projected_worst_case_usd"] > 0
    assert not (tmp_path / "overlay.yaml").exists()


def test_nightly_run_reorders_writes_overlay_and_audit(tmp_path: Path):
    gw = FakeGateway(better={"s2"})
    code, _ = run(tmp_path, ["run", "--per-tier", "10", "--seed", "1"], traffic(), gw)
    assert code == 0
    overlay = load_overlay(tmp_path / "overlay.yaml")
    assert overlay.tier_order[Tier.STANDARD] == ("s2", "s1", "s3")
    assert overlay.version.startswith("opt-000001-")
    assert {c for _, c in gw.calls} == {"eval-replay", "eval-replay-judge"}
    audit = (tmp_path / "optimize-audit.jsonl").read_text()
    entry = json.loads(audit.splitlines()[-1])
    assert entry["result"] == "written" and entry["budget_exhausted"] is False
    assert entry["tiers"]["standard"]["moves"][0]["model"] == "s2"
    assert SECRET not in audit and "served answer" not in audit and "CAND" not in audit
    state = load_state(tmp_path / "overlay.yaml.state.json")
    assert state.seq == 1 and state.previous is None


def test_budget_exhausted_run_records_flag_and_stays_under_cap(tmp_path: Path):
    gw = FakeGateway(better={"s2"}, usage=(1000, 1000))
    code, out = run(
        tmp_path, ["run", "--per-tier", "10"], traffic(), gw, PINBALL_OPT_BUDGET_USD="0.05"
    )
    entry = json.loads(out)
    assert code == 0
    assert entry["budget_exhausted"] is True
    assert entry["spent_usd"] <= 0.05
    assert len(gw.calls) < 60


def test_rejected_candidate_overlay_is_not_written(tmp_path: Path, monkeypatch):
    def reject(*a: object, **k: object) -> Overlay:
        raise OverlayRejected("guardrail", "nope")

    monkeypatch.setattr(cli, "build_overlay", reject)
    code, _ = run(tmp_path, ["run", "--per-tier", "10"], traffic(), FakeGateway(better={"s2"}))
    assert code == cli.EXIT_REJECTED
    assert not (tmp_path / "overlay.yaml").exists()


def test_check_revert_restores_previous_and_keeps_when_healthy(tmp_path: Path):
    db = traffic()
    run(tmp_path, ["run", "--per-tier", "10", "--seed", "1"], db, FakeGateway(better={"s2"}))
    first = load_overlay(tmp_path / "overlay.yaml").version
    later = NOW + timedelta(days=1)
    db.rows = db.rows  # same traffic, s3 now better
    out = io.StringIO()
    cli.main(
        ["run", "--per-tier", "10", "--seed", "2"],
        env_for(tmp_path),
        lambda _: db,
        lambda _: FakeGateway(better={"s3"}),
        later,
        out,
    )
    second = load_overlay(tmp_path / "overlay.yaml").version
    assert second > first

    hours = timedelta(hours=6)
    healthy = Window(100, 1, 1.0, 100)
    db.windows = [(later - hours, later, healthy), (later, later + hours, healthy)]
    check_at = later + hours
    code = cli.main(
        ["--check-revert"], env_for(tmp_path), lambda _: db, None, check_at, io.StringIO()
    )
    assert code == 0 and load_overlay(tmp_path / "overlay.yaml").version == second

    db.windows[1] = (later, later + hours, Window(100, 20, 1.0, 100))
    out = io.StringIO()
    cli.main(["--check-revert"], env_for(tmp_path), lambda _: db, None, check_at, out)
    assert json.loads(out.getvalue())["result"] == "reverted"
    assert load_overlay(tmp_path / "overlay.yaml").version == first
    lines = (tmp_path / "optimize-audit.jsonl").read_text().splitlines()
    assert [json.loads(x).get("result") for x in lines][-2:] == ["kept", "reverted"]


def test_check_revert_to_base_removes_overlay(tmp_path: Path):
    db = traffic()
    run(tmp_path, ["run", "--per-tier", "10"], db, FakeGateway(better={"s2"}))
    hours = timedelta(hours=6)
    db.windows = [
        (NOW - hours, NOW, Window(100, 0, 1.0, 100)),
        (NOW, NOW + hours, Window(100, 0, 2.0, 100)),
    ]
    cli.main(["--check-revert"], env_for(tmp_path), lambda _: db, None, NOW + hours, io.StringIO())
    assert not (tmp_path / "overlay.yaml").exists()


def test_check_revert_ignores_foreign_overlay_and_missing_state(tmp_path: Path):
    out = io.StringIO()
    cli.main(["--check-revert"], env_for(tmp_path), lambda _: FakeDb(), None, NOW, out)
    assert json.loads(out.getvalue())["result"] == "nothing_to_check"
    db = traffic()
    run(tmp_path, ["run", "--per-tier", "10"], db, FakeGateway(better={"s2"}))
    (tmp_path / "overlay.yaml").write_text(
        yaml.safe_dump({"version": "hand-edit", "generated_at": NOW.isoformat()})
    )
    out = io.StringIO()
    cli.main(["--check-revert"], env_for(tmp_path), lambda _: db, None, NOW, out)
    assert json.loads(out.getvalue())["result"] == "foreign_overlay"


def test_missing_config_exits_2(tmp_path: Path):
    env = env_for(tmp_path)
    del env["PINBALL_OPT_DB_DSN"]
    assert cli.main(["run"], env, lambda _: FakeDb(), None, NOW, io.StringIO()) == 2
    env = env_for(tmp_path)
    del env["PINBALL_OPT_GATEWAY_URL"]
    assert cli.main(["run"], env, lambda _: FakeDb(), None, NOW, io.StringIO()) == 2


def test_pair_result_record_is_content_free():
    r = PairResult("id", "standard", True, "claude-opus-5", "gemini-x", "served")
    r.verdicts = {"judge-opus": Verdict(True, True, "tie", "fine"), "judge-argon": None}
    r.same_family = ("judge-argon", "judge-opus")
    rec = r.record()
    assert rec["agreed"] is None and rec["independent"] is None
    assert set(rec) >= {"log_id", "candidate", "verdicts", "agreed"}
