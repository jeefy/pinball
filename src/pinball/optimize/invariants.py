"""In-process copy of carrier's packages/carrier_core/tests/test_pinball_pool.py.

The bot runs it on the EDITED files before it commits or asks Gitea to merge,
so a broken edit never reaches carrier main even when no PR check runs. Keep
it in step with the carrier test (the README of both repos says so).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from pinball.optimize.carrier_edit import ModelId, model_id, provider_for

FALLBACK = "pinball-claude-sonnet-5.5"
WHEN = 'default(extauthz.model, "") == "{}"'


class InvariantError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CarrierFiles:
    """Text of the three carrier files the bot edits (k8s/ paths below)."""

    pinball: str
    gateway: str
    candidates: str


PATHS = {
    "pinball": "k8s/pinball.yaml",
    "gateway": "k8s/agentgateway.yaml",
    "candidates": "k8s/pinball-candidates.yaml",
}


def configmap_data(text: str, key: str) -> dict[str, Any]:
    for doc in yaml.safe_load_all(text):
        if isinstance(doc, dict) and doc.get("kind") == "ConfigMap" and key in doc.get("data", {}):
            data = yaml.safe_load(doc["data"][key])
            if not isinstance(data, dict):
                raise InvariantError(f"{key} is not a mapping")
            return data
    raise InvariantError(f"no ConfigMap with {key}")


@dataclass(frozen=True, slots=True)
class Lists:
    candidates: tuple[ModelId, ...]
    added: dict[ModelId, str]
    dropped: frozenset[ModelId]


def candidate_lists(text: str) -> Lists:
    data = configmap_data(text, "candidates.yaml")
    entries = data.get("candidates") or []
    ids = tuple(model_id(str(e["id"])) for e in entries)
    added = {model_id(str(e["id"])): str(e.get("added", "")) for e in entries}
    dropped = frozenset(model_id(str(e["id"])) for e in data.get("dropped") or [])
    return Lists(ids, added, dropped)


def pool_ids(text: str) -> tuple[ModelId, ...]:
    return tuple(model_id(str(m)) for m in configmap_data(text, "pinball.yaml")["models"])


def _require(ok: bool, what: str) -> None:
    if not ok:
        raise InvariantError(what)


def check(files: CarrierFiles) -> None:
    """Raise InvariantError on the first broken pool/candidate invariant."""
    pool_doc = configmap_data(files.pinball, "pinball.yaml")
    pool = set(pool_doc["models"])
    gw = configmap_data(files.gateway, "config.yaml")
    lists = candidate_lists(files.candidates)
    cands = set(lists.candidates)
    for tier, members in pool_doc["tiers"].items():
        _require(set(members) <= pool, f"tier {tier} has non-pool members")
    virtual = {v["name"]: v for v in gw["llm"]["virtualModels"]}
    targets = virtual["auto"]["routing"]["conditional"]["targets"]
    _require(virtual["pinball"]["routing"] == virtual["auto"]["routing"], "pinball alias != auto")
    routed = {t["when"]: t["model"] for t in targets if "when" in t}
    _require(routed == {WHEN.format(m): f"pinball-{m}" for m in pool}, "auto targets != pool")
    _require(targets[-1] == {"model": FALLBACK}, "auto fallback is not last")
    _require(all("when" in t for t in targets[:-1]), "unconditional target before fallback")
    models = {m["name"]: m for m in gw["llm"]["models"]}
    internal = {n: m for n, m in models.items() if n.startswith("pinball-")}
    _require(set(internal) == {f"pinball-{m}" for m in pool}, "pinball-* models != pool")
    for name, m in internal.items():
        mid = name.removeprefix("pinball-")
        _require(m.get("visibility") == "internal", f"{name} is not internal")
        _require(m["params"]["model"] == mid, f"{name} upstream id mismatch")
        _require(m["provider"]["reference"] == provider_for(model_id(mid)), f"{name} provider")
    opt = {n.removeprefix("opt-"): m for n, m in models.items() if n.startswith("opt-")}
    _require(not (pool & cands), "a candidate is already in the pool")
    _require(set(opt) == pool | cands, "opt-* aliases != pool + candidates")
    for mid, m in opt.items():
        _require(m.get("visibility", "public") == "public", f"opt-{mid} is not public")
        _require(m["params"]["model"] == mid, f"opt-{mid} upstream id mismatch")
        _require(m["provider"]["reference"] == provider_for(model_id(mid)), f"opt-{mid} provider")
    (catalog,) = gw["config"]["modelCatalog"]
    rates = catalog["inline"]["providers"]["openai"]["models"]
    _require(pool <= set(rates), "pool model without openai rates")
