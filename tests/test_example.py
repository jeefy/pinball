from __future__ import annotations

import re
from pathlib import Path

import yaml

from pinball.config import load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "agentgateway.yaml"
WHEN = re.compile(r'^default\(extauthz\.model, ""\) == "([^"]+)"$')


def test_example_auto_targets_cover_the_pool_exactly():
    # Given the shipped pool and the example gateway config
    pool = set(load_config().models)
    llm = yaml.safe_load(EXAMPLE.read_text())["llm"]
    models = {m["name"]: m for m in llm["models"]}
    (auto,) = [v for v in llm["virtualModels"] if v["name"] == "auto"]
    targets = auto["routing"]["conditional"]["targets"]

    # When each conditional target is resolved to the upstream model it serves
    routed = {}
    for t in targets[:-1]:
        match = WHEN.match(t["when"])
        assert match, t["when"]
        model = models[t["model"]]
        assert model["visibility"] == "internal"
        assert model["params"]["model"] == match.group(1)
        routed[match.group(1)] = t["model"]

    # Then every pool model has exactly one target and the last is a plain fallback
    assert set(routed) == pool
    assert len(routed) == len(targets) - 1
    assert "when" not in targets[-1]
    assert targets[-1]["model"] in models
