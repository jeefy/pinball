#!/usr/bin/env python3
"""Regenerate src/pinball/catalog.json from models.dev (MIT) and Terminal-Bench.

Maintainer / CI tool only: a deployed pinball never runs it or fetches either
source. Usage: scripts/build_catalog.py [--models-dev FILE] [--tbench FILE]
(files instead of the network, for reproducible builds).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODELS_DEV = "https://models.dev/api.json"
TBENCH = "https://ofhuhcpkvzjlejydnvyd.supabase.co/functions/v1/leaderboard-read"
TBENCH_BODY = {"package": "terminal-bench/terminal-bench", "name": "4-1-0"}
# First provider wins for an id listed by several.
PROVIDERS = ("github-copilot", "anthropic", "openai", "google", "ollama-cloud")
OUT = Path(__file__).resolve().parents[1] / "src" / "pinball" / "catalog.json"


def _fetch(url: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", s.lower())


def _entry(m: dict[str, Any]) -> dict[str, Any]:
    limit = m.get("limit") or {}
    cost = m.get("cost") or {}
    efforts: list[str] = []
    for opt in m.get("reasoning_options") or []:
        if opt.get("type") == "effort":
            efforts = [str(v) for v in opt.get("values") or []]
    e: dict[str, Any] = {
        "name": m.get("name") or m["id"],
        "family": m.get("family") or "",
        "context": int(limit.get("context") or 0),
        "input": int(limit.get("input") or 0),
        "output": int(limit.get("output") or 0),
        "vision": "image" in ((m.get("modalities") or {}).get("input") or []),
        "tools": bool(m.get("tool_call")),
        "reasoning": bool(m.get("reasoning")),
        "efforts": efforts,
    }
    if "input" in cost and "output" in cost:
        e["cost_in"] = float(cost["input"])
        e["cost_out"] = float(cost["output"])
    return e


def build(models_dev: dict[str, Any], tbench: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for p in PROVIDERS:
        for mid, m in (models_dev.get(p) or {}).get("models", {}).items():
            if mid not in models:
                models[mid] = _entry({**m, "id": mid})
    scores: dict[str, float] = {}
    for row in tbench.get("rows", []):
        label = _norm(row["metadata"]["model_display"]["label"])
        acc = float(row["metrics"]["accuracy"])
        if label:
            scores[label] = max(acc, scores.get(label, 0.0))
    for e in models.values():
        # "Claude Opus 5.5" matches the label "Opus 5.5"; the longest matching label wins
        name = _norm(e["name"])
        hits = [label for label in scores if len(label) >= 4 and name.endswith(label)]
        if hits:
            e["tbench"] = scores[max(hits, key=len)]
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "sources": [MODELS_DEV, "terminal-bench 4.1.0 leaderboard"],
        "models": dict(sorted(models.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dev")
    ap.add_argument("--tbench")
    a = ap.parse_args()
    md = json.loads(Path(a.models_dev).read_text()) if a.models_dev else _fetch(MODELS_DEV)
    tb = json.loads(Path(a.tbench).read_text()) if a.tbench else _fetch(TBENCH, TBENCH_BODY)
    cat = build(md, tb)
    OUT.write_text(json.dumps(cat, indent=1, sort_keys=False) + "\n")
    scored = sum("tbench" in e for e in cat["models"].values())
    print(f"wrote {OUT} models={len(cat['models'])} tbench={scored}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
