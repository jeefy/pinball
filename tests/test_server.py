from __future__ import annotations

import orjson
import pytest
from fastapi.testclient import TestClient

from pinball.config import Tier, load_config
from pinball.decide import Decider, Decision
from pinball.features import Features
from pinball.server import Metrics, create_app


@pytest.fixture
def metrics() -> Metrics:
    return Metrics.create()


@pytest.fixture
def client(metrics: Metrics) -> TestClient:
    return TestClient(create_app(Decider(load_config()), metrics))


def chat(text: str, **extra: object) -> bytes:
    return orjson.dumps({"model": "auto", "messages": [{"role": "user", "content": text}], **extra})


def test_check_returns_the_decision_on_any_path(client: TestClient):
    r = client.post("/v1/chat/completions", content=chat("Generate a title", max_tokens=20))
    assert r.status_code == 200
    assert r.json()["model"] == "claude-haiku-5.5"
    assert r.json()["tier"] == "light"


def test_tier_hint_header_is_honoured(client: TestClient):
    r = client.post("/check", content=chat("hi"), headers={"x-pinball-tier": "HEAVY"})
    assert r.json()["model"] == "claude-opus-5.5"


def test_unknown_hint_is_ignored(client: TestClient):
    r = client.post("/check", content=chat("hi"), headers={"x-pinball-tier": "ludicrous"})
    assert r.json()["tier"] == "light"


def test_min_tier_header_raises_the_tier_and_is_counted(client: TestClient):
    r = client.post("/check", content=chat("hi"), headers={"x-pinball-min-tier": "Standard"})
    assert (r.json()["tier"], r.json()["reason"]) == ("standard", "short,min_tier:standard")
    assert 'pinball_signals_total{signal="min_tier"} 1.0' in client.get("/metrics").text


def test_tier_header_beats_min_tier_header(client: TestClient):
    headers = {"x-pinball-tier": "light", "x-pinball-min-tier": "heavy"}
    r = client.post("/check", content=chat("hi"), headers=headers)
    assert r.json()["tier"] == "light"
    assert "min_tier" not in r.json()["reason"]


def test_unknown_min_tier_is_ignored(client: TestClient):
    r = client.post("/check", content=chat("hi"), headers={"x-pinball-min-tier": "ludicrous"})
    assert r.json()["tier"] == "light"
    assert "min_tier" not in r.json()["reason"]


def test_unreadable_body_answers_200_with_empty_model(client: TestClient, metrics: Metrics):
    r = client.post("/check", content=b"not json")
    assert r.status_code == 200
    assert r.json()["model"] == ""
    assert r.json()["reason"] == "skip:body is not JSON"


class _Exploding(Decider):
    def decide(
        self, features: Features, hint: Tier | None = None, floor: Tier | None = None
    ) -> Decision:
        raise RuntimeError("boom")


def test_internal_error_still_answers_200(metrics: Metrics):
    client = TestClient(create_app(_Exploding(load_config()), metrics))
    r = client.post("/check", content=chat("hi"))
    assert r.status_code == 200
    assert (r.json()["model"], r.json()["reason"]) == ("", "error")


def test_metrics_count_decisions(client: TestClient):
    client.post("/check", content=chat("Generate a title", max_tokens=20))
    text = client.get("/metrics").text
    assert 'pinball_decisions_total{model="claude-haiku-5.5",tier="light"} 1.0' in text
    assert 'pinball_signals_total{signal="light"}' in text


def test_healthz(client: TestClient):
    assert client.get("/healthz").json() == {"ok": True}


def test_large_body_uses_threadpool_path(client: TestClient):
    padding = "a" * 20000
    r = client.post("/check", content=chat(f"hello {padding}"))
    assert r.status_code == 200
    assert r.json()["model"] != ""


def test_reply_has_no_signals_by_default(client: TestClient):
    assert "signals" not in client.post("/check", content=chat("hi")).json()


def test_reply_signals_opt_in_lists_names_only(metrics: Metrics):
    app = create_app(Decider(load_config()), metrics, reply_signals=True)
    r = TestClient(app).post(
        "/check", content=chat("hi"), headers={"x-pinball-min-tier": "standard"}
    )
    assert r.json()["signals"] == ["short", "min_tier"]
    assert TestClient(app).post("/check", content=b"nope").json().get("signals") is None
