from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pinball.config import TIER_ORDER, Config, default_config_path, keyword_pattern, load_config

BASE = {
    "models": {"a": {"prompt_tokens": 10}, "b": {"prompt_tokens": 20}},
    "tiers": {"light": ["a"], "standard": ["a"], "heavy": ["b"]},
    "thresholds": {"standard": 1, "heavy": 3},
}


def test_shipped_config_loads_with_every_tier_populated():
    cfg = load_config()
    assert default_config_path().name == "pinball.yaml"
    assert all(cfg.tier_models(t) for t in TIER_ORDER)


def test_tier_naming_a_model_outside_the_pool_is_rejected():
    with pytest.raises(ValidationError, match="not in `models`"):
        Config.model_validate({**BASE, "tiers": {**BASE["tiers"], "heavy": ["zzz"]}})


def test_missing_tier_is_rejected():
    with pytest.raises(ValidationError, match="tiers.heavy"):
        Config.model_validate({**BASE, "tiers": {"light": ["a"], "standard": ["a"]}})


def test_heavy_threshold_must_exceed_standard():
    with pytest.raises(ValidationError, match="greater than"):
        Config.model_validate({**BASE, "thresholds": {"standard": 3, "heavy": 3}})


def test_load_config_reads_an_explicit_path(tmp_path: Path):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(BASE))
    assert set(load_config(p).models) == {"a", "b"}


def test_keyword_pattern_matches_whole_words_only():
    pat = keyword_pattern(("plan", "root cause"))
    assert pat is not None
    assert pat.search("ROOT CAUSE please")
    assert not pat.search("airplane")
    assert keyword_pattern(()) is None
