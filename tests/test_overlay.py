from __future__ import annotations

import os
import threading
from pathlib import Path

import orjson
import pytest
import yaml
from fastapi.testclient import TestClient

from pinball.config import Config, Tier
from pinball.decide import Decider
from pinball.features import parse
from pinball.overlay import Overlay, OverlayRejected, base_only, load_overlay, merge
from pinball.reload import Reloader
from pinball.server import Metrics, create_app

BASE = Config.model_validate(
    {
        "models": {
            "l1": {"prompt_tokens": 1000},
            "l2": {"prompt_tokens": 1000},
            "s1": {"prompt_tokens": 2000},
            "s2": {"prompt_tokens": 2000},
            "h1": {"prompt_tokens": 4000},
        },
        "tiers": {"light": ["l1", "l2"], "standard": ["s1", "s2"], "heavy": ["h1"]},
        "thresholds": {"standard": 1, "heavy": 3},
        "headroom": 1.0,
    }
)
STAMP = "2026-11-02T03:00:00Z"
TOOLS = orjson.dumps(
    {"messages": [{"role": "user", "content": "x" * 200}], "tools": [{"type": "function"}]}
)  # score 1: standard under base thresholds


def overlay(**extra: object) -> dict[str, object]:
    return {"version": "v1", "generated_at": STAMP, **extra}


def write(path: Path, doc: object) -> None:
    # Replace, as a ConfigMap/PVC writer would: new inode every time.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(doc) if not isinstance(doc, str) else doc)
    os.replace(tmp, path)


def test_valid_overlay_reorders_within_tier():
    active = merge(BASE, Overlay.model_validate(overlay(tier_order={"standard": ["s2", "s1"]})))
    d = active.decider.decide(parse(TOOLS))
    assert (d.tier, d.model, active.version) == (Tier.STANDARD, "s2", "v1")


@pytest.mark.parametrize(
    ("doc", "reason"),
    [
        (overlay(tier_order={"standard": ["s2", "zzz"]}), "unknown_model"),
        (overlay(tier_order={"standard": ["s1", "s2", "h1"]}), "cross_tier"),
        (overlay(tier_order={"heavy": ["s1"]}), "cross_tier"),
        (overlay(tier_order={"standard": ["s1"]}), "not_permutation"),
        (overlay(tier_order={"standard": ["s1", "s1"]}), "not_permutation"),
        (overlay(callers={"bot": {"tier_order": {"light": ["l2", "zzz"]}}}), "unknown_model"),
        (
            overlay(callers={"bot": {"threshold_offsets": {"standard": 5, "heavy": 0}}}),
            "guardrail",
        ),
    ],
)
def test_overlay_breaking_base_rules_is_rejected(doc: dict[str, object], reason: str):
    with pytest.raises(OverlayRejected) as e:
        merge(BASE, Overlay.model_validate(doc))
    assert e.value.reason == reason


def test_schema_rejects_unknown_keys_and_guardrail_fields(tmp_path: Path):
    p = tmp_path / "o.yaml"
    for doc in (overlay(models={"x": {"prompt_tokens": 1}}), overlay(headroom=1.0), {"x": 1}):
        write(p, doc)
        with pytest.raises(OverlayRejected, match="schema"):
            load_overlay(p)


def test_caller_profile_applies_only_to_that_caller():
    doc = overlay(
        callers={
            "bot": {"tier_order": {"standard": ["s2", "s1"]}},
            "cheap": {"threshold_offsets": {"standard": 1, "heavy": 1}, "weight": 1.0},
            "shrunk": {"threshold_offsets": {"standard": 1, "heavy": 1}, "weight": 0.2},
        }
    )
    active = merge(BASE, Overlay.model_validate(doc))
    pick = {c: active.select(c) for c in ("bot", "cheap", "shrunk", "other")}
    f = parse(TOOLS)
    assert pick["bot"][0].decide(f).model == "s2"
    assert pick["cheap"][0].decide(f).tier == Tier.LIGHT
    assert pick["shrunk"][0].decide(f).tier == Tier.STANDARD  # round(0.2) == 0
    assert pick["other"][0].decide(f).model == "s1"
    assert (pick["bot"][1], pick["other"][1]) == ("bot", "")


def test_missing_overlay_file_means_base_only(tmp_path: Path):
    r = Reloader(BASE, tmp_path / "absent.yaml", Metrics.create().reload)
    assert r.active.version == ""
    assert r.active.decider.decide(parse(TOOLS)) == Decider(BASE).decide(parse(TOOLS))


def test_corrupt_overlay_keeps_last_good_and_counts(tmp_path: Path):
    p = tmp_path / "o.yaml"
    write(p, overlay(tier_order={"standard": ["s2", "s1"]}))
    m = Metrics.create()
    r = Reloader(BASE, p, m.reload)
    assert r.active.version == "v1"

    write(p, "version: [unclosed")
    r.poll()
    write(p, {**overlay(tier_order={"heavy": ["s1"]}), "version": "v2"})
    r.poll()

    assert r.active.version == "v1"
    assert r.active.decider.decide(parse(TOOLS)).model == "s2"
    reg = m.registry
    assert reg.get_sample_value("pinball_overlay_reload_total", {"result": "rejected"}) == 2
    assert reg.get_sample_value("pinball_overlay_reload_errors_total", {"reason": "parse"}) == 1
    assert reg.get_sample_value("pinball_overlay_info", {"version": "v1"}) == 1


def test_removed_overlay_falls_back_to_base(tmp_path: Path):
    p = tmp_path / "o.yaml"
    write(p, overlay())
    r = Reloader(BASE, p, Metrics.create().reload)
    p.unlink()
    r.poll()
    assert r.active.version == ""


def test_swap_is_atomic_under_concurrent_decisions(tmp_path: Path):
    # Given two overlays whose (version, model) pairs differ
    p = tmp_path / "o.yaml"
    docs = [
        overlay(tier_order={"standard": ["s1", "s2"]}),
        {**overlay(tier_order={"standard": ["s2", "s1"]}), "version": "v2"},
    ]
    expected = {"v1": "s1", "v2": "s2"}
    write(p, docs[0])
    r = Reloader(BASE, p, Metrics.create().reload)
    seen: list[tuple[str, str]] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            active = r.active
            seen.append((active.version, active.decider.decide(parse(TOOLS)).model))

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    # When the overlay flips many times
    for i in range(40):
        write(p, docs[(i + 1) % 2])
        r.poll()
    stop.set()
    for t in threads:
        t.join()
    # Then every decision saw one consistent state
    assert seen
    assert all(expected[v] == m for v, m in seen)


def test_server_logs_and_applies_caller_profile(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    p = tmp_path / "o.yaml"
    write(p, overlay(callers={"bot": {"tier_order": {"standard": ["s2", "s1"]}}}))
    m = Metrics.create()
    client = TestClient(create_app(Reloader(BASE, p, m.reload), m))
    caplog.set_level("INFO", logger="pinball")
    assert client.post("/c", content=TOOLS, headers={"x-agw-caller": "bot"}).json()["model"] == "s2"
    assert client.post("/c", content=TOOLS).json()["model"] == "s1"
    assert "overlay=v1 profile=bot" in caplog.text
    assert client.get("/healthz").json() == {"ok": True}


def test_background_poll_picks_up_changes(tmp_path: Path):
    p = tmp_path / "o.yaml"
    r = Reloader(BASE, p, Metrics.create().reload)
    with TestClient(create_app(r, Metrics.create(), reload_interval=0.01)):
        write(p, overlay())
        for _ in range(500):
            if r.active.version == "v1":
                break
            threading.Event().wait(0.01)
    assert r.active.version == "v1"


def test_base_only_matches_plain_decider():
    assert base_only(BASE).select("anyone")[1] == ""
