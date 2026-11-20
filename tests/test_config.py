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


def test_keyword_pattern_wildcard_suffix_matches_word_endings():
    pat = keyword_pattern(("migrier*", "ursache"))
    assert pat is not None
    assert pat.search("Bitte migriere die DB")
    assert pat.search("Wir migrieren die DB")
    assert pat.search("die Ursache ist klar")
    assert not pat.search("Ursachenforschung")
    assert not pat.search("immigrierte")


def test_shipped_german_keywords_score_a_german_request():
    import orjson

    from pinball.decide import Decider
    from pinball.features import parse

    text = "Analysiere die Ursache und optimiere die Architektur"
    body = orjson.dumps({"messages": [{"role": "user", "content": text}]})
    d = Decider(load_config()).decide(parse(body))
    assert d.score >= 2 and "kw:architektur" in d.signals


def test_scoring_defaults_apply_when_the_block_is_absent():
    s = Config.model_validate(BASE).scoring
    assert (s.code, s.very_long, s.long_chars, s.short_chars, s.max_heavy_keywords) == (
        1,
        2,
        4000,
        160,
        2,
    )


def test_scoring_rejects_unknown_keys_and_inverted_cutoffs():
    with pytest.raises(ValidationError, match="extra"):
        Config.model_validate({**BASE, "scoring": {"cod": 1}})
    with pytest.raises(ValidationError, match="very_long_chars"):
        Config.model_validate({**BASE, "scoring": {"long_chars": 5000, "very_long_chars": 5000}})
