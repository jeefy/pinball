from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import orjson
import pytest

from pinball import discovery
from pinball.config import Config, Tier, load_config
from pinball.decide import Decider
from pinball.features import parse
from pinball.main import initial_config

CATALOG: dict[str, dict[str, Any]] = {
    "big": {"tbench": 65.0, "cost_out": 20.0, "input": 300000, "efforts": ["low", "high"]},
    "mid": {"tbench": 40.0, "cost_out": 10.0},
    "cheap": {"cost_out": 0.5, "vision": True},
    "pricey": {"cost_out": 50.0},
}


def entry(mid: str, **kw: Any) -> dict[str, Any]:
    return {"id": mid, **kw}


def copilot(mid: str, category: str, endpoints: list[str], **supports: Any) -> dict[str, Any]:
    return {
        "id": mid,
        "model_picker_category": category,
        "supported_endpoints": endpoints,
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 128000},
            "supports": {"tool_calls": True, **supports},
        },
    }


def pool(*items: dict[str, Any]) -> tuple[dict[str, Any], dict[Tier, tuple[str, ...]]]:
    models, tiers, _ = discovery.build_pool({"data": list(items)}, CATALOG)
    return models, tiers


def test_tiers_from_tbench_then_cost_then_default():
    _, tiers = pool(entry("big"), entry("mid"), entry("cheap"), entry("pricey"), entry("new"))
    assert tiers[Tier.HEAVY] == ("big", "pricey")  # tbench ranks before price-only
    assert tiers[Tier.STANDARD] == ("mid", "new")
    assert tiers[Tier.LIGHT] == ("cheap",)


def test_vendor_category_wins_over_catalogue():
    _, tiers = pool(copilot("big", "lightweight", ["/chat/completions"]), entry("mid"))
    assert tiers[Tier.LIGHT] == ("big",)


def test_skips_embeddings_snapshots_hidden_and_uncallable():
    models, _ = pool(
        entry("text-embedding-3-small"),
        entry("mid-2025-01-01"),
        {**entry("mid"), "model_picker_enabled": False},
        copilot("weird", "versatile", ["/v1/realtime"]),
        entry("cheap"),
    )
    assert list(models) == ["cheap"]


def test_empty_tier_borrows_nearest():
    _, tiers = pool(entry("mid"))
    assert tiers == {Tier.LIGHT: ("mid",), Tier.STANDARD: ("mid",), Tier.HEAVY: ("mid",)}


def test_live_limits_and_capabilities_win():
    models, _ = pool(
        copilot("big", "powerful", ["/responses"], vision=False, reasoning_effort=["high"])
    )
    spec = models["big"]
    assert spec.prompt_tokens == 128000
    assert spec.api == ("responses",)
    assert spec.efforts == ("high",)


def test_live_supports_without_reasoning_effort_rejects_the_parameter():
    models, _ = pool(copilot("cheap", "lightweight", ["/chat/completions"]))
    assert models["cheap"].efforts == ()


def test_catalogue_fills_unknowns_and_caps_window():
    models, _ = pool(entry("big"), entry("unknown"))
    assert models["big"].prompt_tokens == 250000
    assert models["big"].efforts == ("low", "high")
    assert models["unknown"].prompt_tokens == discovery.DEFAULT_PROMPT_TOKENS
    assert models["unknown"].efforts is None


def test_no_chat_models_is_an_error():
    with pytest.raises(discovery.DiscoveryError):
        discovery.build_pool({"data": [entry("text-embedding-3")]}, CATALOG)
    with pytest.raises(discovery.DiscoveryError):
        discovery.build_pool({"nope": 1}, CATALOG)


def test_pin_moves_listed_models_to_the_front():
    base = load_config().model_copy(update={"pin": {Tier.HEAVY: ("pricey", "absent")}})
    models, tiers = pool(entry("big"), entry("pricey"))
    cfg = discovery.apply_pool(base, models, tiers)
    assert cfg.tiers[Tier.HEAVY] == ("pricey", "big")


def _cfg(*items: dict[str, Any]) -> Config:
    models, tiers = pool(*items)
    return discovery.apply_pool(load_config(), models, tiers)


def test_pick_skips_models_that_cannot_serve_the_api_shape():
    cfg = _cfg(
        copilot("nano", "lightweight", ["/responses"]),
        copilot("cheap", "lightweight", ["/chat/completions"]),
    )
    body = orjson.dumps({"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    d = Decider(cfg).decide(parse(body))
    assert d.model in ("cheap",)
    body = orjson.dumps({"model": "auto", "input": "hi"})
    assert Decider(cfg).decide(parse(body)).model == "nano"


def test_pick_skips_models_that_reject_the_effort_word():
    cfg = _cfg(
        copilot("haiku", "lightweight", ["/chat/completions"]),
        copilot("mini", "versatile", ["/chat/completions"], reasoning_effort=["low"]),
    )
    body = orjson.dumps(
        {
            "model": "auto",
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    d = Decider(cfg).decide(parse(body), hint=Tier.LIGHT)
    assert d.model == "mini"


def test_features_detect_api_shape():
    assert parse(b'{"messages":[]}').api == "chat"
    assert parse(b'{"system":"x","messages":[]}').api == "messages"
    assert parse(b'{"input":"x"}').api == "responses"
    assert parse(b'{"prompt":"x"}').api == ""


class _Resp(io.BytesIO):
    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_discover_sends_token_and_returns_payload():
    seen: dict[str, Any] = {}
    payload = {"data": [entry("big"), entry("cheap")]}

    def opener(req: Any, timeout: float) -> _Resp:
        seen["auth"] = req.get_header("Authorization")
        return _Resp(json.dumps(payload).encode())

    cfg, raw = discovery.discover(load_config(), "http://x/v1/models", "tok", CATALOG, opener)
    assert seen["auth"] == "Bearer tok"
    assert raw == payload
    assert cfg.tiers[Tier.HEAVY] == ("big",)


def test_initial_config_falls_back_to_cache_then_static(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    static = load_config()
    cache = tmp_path / "pool.json"
    assert initial_config(static, "", "", cache) is static

    def down(*_: Any, **__: Any) -> Any:
        raise discovery.DiscoveryError("down")

    monkeypatch.setattr(discovery, "fetch", down)
    assert initial_config(static, "http://x", "", cache) is static
    discovery.save_cache(cache, {"data": [{"id": "claude-opus-5.5"}]})
    cfg = initial_config(static, "http://x", "", cache)
    assert list(cfg.models) == ["claude-opus-5.5"]

    monkeypatch.setattr(discovery, "fetch", lambda *_a, **_k: {"data": [{"id": "gpt-5.4-nano"}]})
    cfg = initial_config(static, "http://x", "", cache)
    assert list(cfg.models) == ["gpt-5.4-nano"]
    assert json.loads(cache.read_text()) == {"data": [{"id": "gpt-5.4-nano"}]}


def test_shipped_catalogue_tiers_known_models_sensibly():
    cat = discovery.load_catalog()
    ids = ["claude-haiku-5.5", "claude-sonnet-5.5", "claude-opus-5.5"]
    _, tiers, _ = discovery.build_pool({"data": [{"id": i} for i in ids]}, cat)
    assert tiers[Tier.HEAVY][0] == "claude-opus-5.5"
    assert "claude-haiku-5.5" in tiers[Tier.STANDARD] + tiers[Tier.LIGHT]
