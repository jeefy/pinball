from __future__ import annotations

import io
import json
from datetime import timedelta
from pathlib import Path

import yaml
from test_optimize import BASE, NOW, SECRET, FakeGateway, env_for, row

from pinball.config import Tier
from pinball.optimize import cli
from pinball.optimize.caller_settings import CallerSettings, caller_settings_from_env
from pinball.optimize.callers import Learned, plan_profiles
from pinball.optimize.db import (
    POOL_SQL,
    WINDOW_CALLER_SQL,
    WINDOW_SQL,
    FakeDb,
    Window,
)
from pinball.optimize.sample import draw_samples
from pinball.overlay import CallerProfile, load_overlay

ENV = {"PINBALL_OPT_CALLER_MIN_SAMPLES": "10", "PINBALL_OPT_CALLER_SHRINK_K": "10"}


def mixed(n_bot: int, n_other: int = 30, bot: str = "bot") -> FakeDb:
    """Global traffic served by s1, plus `n_bot` standard requests from `bot`."""
    rows = [row(i, caller="opencode@bluefin") for i in range(n_other)]
    rows += [row(1000 + i, caller=bot) for i in range(n_bot)]
    return FakeDb(rows)


class CallerGateway(FakeGateway):
    """Candidate wins only on requests whose prompt carries `marker` (the bot's)."""

    def __init__(self, better: set[str], marker: str) -> None:
        super().__init__(better)
        self.marker = marker

    def chat(self, model: str, messages: list[dict[str, str]], *, max_tokens: int, caller: str):
        if caller == "eval-replay-judge" and self.marker not in messages[-1]["content"]:
            self.better, saved = set(), self.better
            try:
                return super().chat(model, messages, max_tokens=max_tokens, caller=caller)
            finally:
                self.better = saved
        return super().chat(model, messages, max_tokens=max_tokens, caller=caller)


def bot_db(n_bot: int) -> FakeDb:
    db = mixed(0)
    db.rows += [
        row(1000 + i, caller="bot", request=[{"role": "user", "content": f"BOTQ {SECRET} {i}"}])
        for i in range(n_bot)
    ]
    return db


def go(tmp_path: Path, db: FakeDb, gw: FakeGateway | None, *argv: str, when=NOW, **env: str):
    out = io.StringIO()
    code = cli.main(
        list(argv) or ["run", "--per-tier", "10", "--seed", "1"],
        env_for(tmp_path, **ENV, **env),
        lambda _: db,
        (lambda _: gw) if gw is not None else None,
        when,
        out,
    )
    return code, json.loads(out.getvalue())


def test_pool_sql_reads_caller_from_agentgateway_user():
    assert "agentgateway_user" in POOL_SQL and "AS caller" in POOL_SQL
    assert "%(caller)s" in WINDOW_CALLER_SQL


def test_every_query_reads_both_virtual_model_names():
    # OpenCode sends model=pinball; other clients send auto. Both are routed by the same
    # virtual model, so a filter on 'auto' alone hid most real traffic (first live run).
    for sql in (POOL_SQL, WINDOW_SQL):
        assert "IN ('auto', 'pinball')" in sql


def test_caller_below_minimum_gets_no_profile(tmp_path: Path):
    code, entry = go(tmp_path, bot_db(5), CallerGateway({"s2"}, "BOTQ"))
    assert code == 0
    assert "bot" not in entry["callers"]
    overlay_file = tmp_path / "overlay.yaml"
    assert not overlay_file.exists() or "bot" not in load_overlay(overlay_file).callers


def test_caller_with_traffic_but_few_judged_samples_gets_no_profile(tmp_path: Path):
    code, entry = go(
        tmp_path, bot_db(12), CallerGateway({"s2"}, "BOTQ"), PINBALL_OPT_CALLER_PER_TIER="2"
    )
    assert code == 0 and entry["callers"]["bot"]["result"] == "below_min_samples"
    assert entry["callers"]["bot"]["judged_non_agentic"] < 10
    assert entry["result"] == "unchanged"


def test_caller_above_minimum_gets_profile_and_global_is_unchanged(tmp_path: Path):
    code, entry = go(tmp_path, bot_db(30), CallerGateway({"s2"}, "BOTQ"))
    assert code == 0 and entry["result"] == "written"
    overlay = load_overlay(tmp_path / "overlay.yaml")
    assert overlay.tier_order[Tier.STANDARD] == ("s1", "s2", "s3")  # global: s2 never wins
    profile = overlay.callers["bot"]
    assert profile.tier_order[Tier.STANDARD] == ("s2", "s1", "s3")
    n = entry["callers"]["bot"]["judged_non_agentic"]
    assert n >= 10 and profile.weight == round(n / (n + 10), 4)
    assert profile.threshold_offsets.standard == 0 and profile.threshold_offsets.heavy == 0
    assert entry["caller_samples"] > 0
    s1 = entry["tiers"]["standard"]["scores"][0]
    assert s1["model"] == "s1" and s1["n"] == 20  # 10 global samples x 2 candidates only


def test_caller_samples_do_not_starve_or_change_global_sample():
    db = bot_db(40)
    alone = draw_samples(db, 1, 10, 7)
    assert [s.log_id for s in draw_samples(db, 1, 10, 7, db.pool(1))] == [s.log_id for s in alone]


def test_seeded_caller_sampling_is_deterministic(tmp_path: Path):
    from pinball.optimize.callers import draw_caller_samples

    db = bot_db(60)
    cs = CallerSettings(min_samples=10, per_tier=5)
    pool = db.pool(1)

    def ids(seed: int) -> list[str]:
        return [
            s.log_id
            for s in draw_caller_samples(db, pool, draw_samples(db, 1, 5, seed, pool), seed, cs)
        ]

    assert ids(3) == ids(3) and ids(3) != ids(4)


def learned(caller: str, n: int, order: tuple[str, ...]) -> Learned:
    p = CallerProfile.model_validate({"tier_order": {"standard": order}, "weight": 0.5})
    return Learned(caller, n, p, {"result": "learned"})


ORDER = {"light": ("l1", "l2"), "standard": ("s1", "s2", "s3"), "heavy": ("h1",)}


def test_out_of_guardrail_profile_is_rejected_before_write():
    bad = Learned(
        "bot",
        50,
        CallerProfile.model_validate({"tier_order": {"standard": ("s1", "h1", "s3")}}),
        {},
    )
    offs = CallerProfile.model_validate({"threshold_offsets": {"heavy": -5}, "weight": 1.0})
    plan = plan_profiles(
        {}, [bad, Learned("b2", 40, offs, {})], set(), BASE, ORDER, CallerSettings()
    )
    assert plan.profiles == {}
    assert plan.audit["bot"]["result"] == "rejected cross_tier"
    assert plan.audit["b2"]["result"] == "rejected guardrail"


def test_profile_cap_and_change_cap():
    many = [learned(f"c{i}", 100 - i, ("s2", "s1", "s3")) for i in range(5)]
    capped = plan_profiles({}, many, set(), BASE, ORDER, CallerSettings(max_profiles=2))
    assert set(capped.profiles) == {"c0", "c1"}
    assert capped.audit["c4"]["result"] == "deferred_max_profiles"
    slow = plan_profiles({}, many, set(), BASE, ORDER, CallerSettings(max_changes=3))
    assert len(slow.profiles) == 3 and slow.audit["c3"]["result"] == "deferred_max_changes"


def test_stale_profile_is_dropped(tmp_path: Path):
    go(tmp_path, bot_db(30), CallerGateway({"s2"}, "BOTQ"))
    assert "bot" in load_overlay(tmp_path / "overlay.yaml").callers
    later = NOW + timedelta(days=15)
    code, entry = go(tmp_path, mixed(0), FakeGateway(), "run", "--seed", "2", when=later)
    assert code == 0 and entry["callers"]["bot"]["result"] == "dropped_stale"
    assert "bot" not in load_overlay(tmp_path / "overlay.yaml").callers


def test_recent_profile_survives_a_quiet_night(tmp_path: Path):
    go(tmp_path, bot_db(30), CallerGateway({"s2"}, "BOTQ"))
    go(tmp_path, mixed(0), FakeGateway(), "run", "--seed", "2", when=NOW + timedelta(days=3))
    assert "bot" in load_overlay(tmp_path / "overlay.yaml").callers


def test_per_caller_revert_only_reverts_that_caller(tmp_path: Path):
    path = tmp_path / "overlay.yaml"
    db = bot_db(30)
    db.rows += [
        row(5000 + i, caller="kept", request=[{"role": "user", "content": f"BOTQ k {i}"}])
        for i in range(30)
    ]
    go(tmp_path, db, CallerGateway({"s2"}, "BOTQ"))
    before = load_overlay(path)
    assert {"bot", "kept"} <= set(before.callers)
    h = timedelta(hours=6)
    db.windows = [
        (NOW - h, NOW, Window(100, 1, 1.0, 100)),
        (NOW, NOW + h, Window(100, 1, 1.0, 100)),
    ]
    db.caller_windows = {
        ("bot", NOW - h, NOW): Window(40, 0, 1.0, 40),
        ("bot", NOW, NOW + h): Window(40, 10, 1.0, 40),
        ("kept", NOW - h, NOW): Window(40, 0, 1.0, 40),
        ("kept", NOW, NOW + h): Window(40, 0, 1.0, 40),
    }
    code, entry = go(tmp_path, db, None, "--check-revert", when=NOW + h)
    assert code == 0 and entry["result"] == "callers_reverted"
    assert entry["callers_reverted"] == ["bot"]
    after = load_overlay(path)
    assert "bot" not in after.callers and after.callers["kept"] == before.callers["kept"]
    assert after.tier_order == before.tier_order and after.version > before.version
    # The next check sees our own write, and the reverted caller is no longer watched.
    code, entry = go(tmp_path, db, None, "--check-revert", when=NOW + h)
    assert entry["result"] == "kept" and entry["callers_reverted"] == []


def test_audit_has_no_prompt_text_or_long_caller(tmp_path: Path):
    long_caller = "x" * 100
    db = bot_db(30)
    db.rows += [row(7000 + i, caller=long_caller) for i in range(30)]
    go(tmp_path, db, CallerGateway({"s2"}, "BOTQ"))
    audit = (tmp_path / "optimize-audit.jsonl").read_text()
    state = (tmp_path / "overlay.yaml.callers.json").read_text()
    for text in (audit, state, (tmp_path / "overlay.yaml").read_text()):
        assert SECRET not in text and "BOTQ" not in text and long_caller not in text
    assert "x" * 64 in audit


def test_caller_settings_from_env():
    cs = caller_settings_from_env({"PINBALL_OPT_CALLER_MIN_SAMPLES": "7"})
    assert cs.min_samples == 7 and caller_settings_from_env({}).min_samples == 30
    assert yaml.safe_dump(cs.model_dump())  # plain data
