from __future__ import annotations

import orjson
import pytest

from pinball.config import Config, Tier, load_config
from pinball.decide import Decider
from pinball.features import parse

CONFIG = Config.model_validate(
    {
        "models": {
            "l1": {"prompt_tokens": 1000, "vision": True},
            "s1": {"prompt_tokens": 2000, "vision": False},
            "s2": {"prompt_tokens": 2000, "vision": True},
            "h1": {"prompt_tokens": 4000, "vision": True},
            "big": {"prompt_tokens": 100000, "vision": True},
        },
        "tiers": {"light": ["l1"], "standard": ["s1", "s2"], "heavy": ["h1"]},
        "thresholds": {"standard": 1, "heavy": 3},
        "headroom": 1.0,
        "heavy_keywords": ["refactor", "root cause", "security"],
        "light_keywords": ["title"],
    }
)


def chat(*users: str, **extra: object) -> bytes:
    return orjson.dumps(
        {"model": "auto", "messages": [{"role": "user", "content": u} for u in users], **extra}
    )


def decide(raw: bytes, hint: Tier | None = None):
    return Decider(CONFIG).decide(parse(raw), hint)


def test_short_title_request_with_small_output_goes_light():
    d = decide(chat("Generate a title for this chat", max_tokens=50))
    assert (d.tier, d.model) == (Tier.LIGHT, "l1")
    assert {"light:title", "short", "small_output"} <= set(d.signals)


def test_plain_question_without_signals_is_standard_when_tools_present():
    d = decide(chat("x" * 200, tools=[{"type": "function"}]))
    assert (d.tier, d.model, d.signals) == (Tier.STANDARD, "s1", ("tools",))


def test_agentic_debugging_with_code_goes_heavy():
    msg = "Find the root cause:\n```\nTraceback (most recent call last)\n```\n" + "y" * 200
    d = decide(chat(msg, tools=[{"type": "function"}]))
    assert (d.tier, d.model) == (Tier.HEAVY, "h1")
    assert d.score == 3


def test_tier_never_drops_when_a_later_message_is_trivial():
    # Given a conversation whose first user turn was heavy and the latest is "thanks"
    heavy = "Refactor the auth module for security\n```py\npass\n```" + "z" * 200
    # When
    d = decide(chat(heavy, "thanks"))
    # Then
    assert d.tier is Tier.HEAVY


def test_tier_steps_up_when_a_later_message_is_harder():
    d = decide(chat("hi", "Refactor this for security\n```\ncode\n```" + "z" * 200))
    assert d.tier is Tier.HEAVY


def test_keywords_in_tool_output_do_not_score():
    raw = orjson.dumps(
        {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "x" * 200},
                {"role": "tool", "content": "refactor security root cause"},
            ],
        }
    )
    d = decide(raw)
    assert not [s for s in d.signals if s.startswith("kw:")]


def test_keyword_points_are_capped_at_two():
    d = decide(chat("refactor security root cause " + "x" * 200))
    assert d.score == 2
    assert len([s for s in d.signals if s.startswith("kw:")]) == 2


def test_long_and_very_long_messages_add_points():
    assert decide(chat("a " * 2500)).signals[0] == "long"
    assert decide(chat("a " * 7000)).signals[0] == "very_long"


def test_high_effort_adds_two_and_low_effort_subtracts_one():
    assert decide(chat("x" * 200, reasoning_effort="high")).score == 2
    assert decide(chat("x" * 200, reasoning_effort="low")).score == -1


def test_structured_output_subtracts_one():
    d = decide(chat("x" * 200, response_format={"type": "json_object"}))
    assert (d.score, d.signals) == (-1, ("structured",))


def test_prompt_too_big_for_tier_moves_to_next_model_that_fits():
    # Given a light request (~1430 tokens) larger than l1's 1000-token window
    d = decide(chat("Generate a title " + "x" * 5000, max_tokens=50))
    # Then the tier stays light but the model is the first that fits
    assert (d.tier, d.model) == (Tier.LIGHT, "s1")
    assert "fit:s1" in d.signals


def test_images_skip_models_without_vision():
    raw = orjson.dumps(
        {
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "x" * 200},
                        {"type": "image_url", "image_url": {"url": "data:"}},
                    ],
                }
            ],
            "tools": [{"type": "function"}],
        }
    )
    d = decide(raw)
    assert (d.tier, d.model) == (Tier.STANDARD, "s2")


def test_prompt_bigger_than_every_tier_falls_back_to_largest_window():
    d = decide(chat("x" * 50000))
    assert d.model == "big"
    assert "fit:big" in d.signals


@pytest.mark.parametrize("hint", [Tier.LIGHT, Tier.HEAVY])
def test_hint_overrides_the_score(hint: Tier):
    d = decide(chat("Refactor for security " + "x" * 200, tools=[{}]), hint)
    assert d.tier is hint
    assert f"hint:{hint}" in d.signals


def test_shipped_config_routes_title_to_haiku_and_debugging_to_opus():
    decider = Decider(load_config())
    light = decider.decide(parse(chat("Generate a short title", max_tokens=30)))
    heavy = decider.decide(
        parse(
            chat(
                "Find the root cause of this deadlock\n```go\nmu.Lock()\n```",
                tools=[{"type": "function"}],
            )
        )
    )
    assert light.model == "claude-haiku-5.5"
    assert heavy.model == "claude-opus-5.5"


def test_scoring_block_changes_weights_and_cutoffs():
    tuned = CONFIG.model_copy(
        update={"scoring": CONFIG.scoring.model_copy(update={"heavy_keyword": 3, "short_chars": 0})}
    )
    d = Decider(tuned).decide(parse(chat("fix the security bug")))
    assert (d.tier, d.score, d.signals) == (Tier.HEAVY, 3, ("kw:security",))


def test_max_heavy_keywords_caps_signals_and_points():
    tuned = CONFIG.model_copy(
        update={"scoring": CONFIG.scoring.model_copy(update={"max_heavy_keywords": 1})}
    )
    d = Decider(tuned).decide(parse(chat("refactor for security and find the root cause")))
    assert d.score == 1 and [s for s in d.signals if s.startswith("kw:")] == ["kw:refactor"]
