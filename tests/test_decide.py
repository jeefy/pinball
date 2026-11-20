from __future__ import annotations

import orjson
import pytest

from pinball.config import Config, Tier, WrappedUserMessage, load_config
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


def decide(raw: bytes, hint: Tier | None = None, floor: Tier | None = None):
    return Decider(CONFIG).decide(parse(raw), hint, floor)


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


def test_floor_raises_light_to_standard_and_says_so():
    d = decide(chat("Generate a title", max_tokens=20), floor=Tier.STANDARD)
    assert (d.tier, d.model) == (Tier.STANDARD, "s1")
    assert "min_tier:standard" in d.signals


def test_floor_does_not_lower_a_higher_score():
    d = decide(chat("Refactor for security " + "x" * 200, tools=[{}]), floor=Tier.STANDARD)
    assert d.tier is Tier.HEAVY
    assert not any(s.startswith("min_tier") for s in d.signals)


def test_floor_at_the_scored_tier_adds_no_signal():
    d = decide(chat("x" * 200, tools=[{"type": "function"}]), floor=Tier.STANDARD)
    assert (d.tier, d.signals) == (Tier.STANDARD, ("tools",))


def test_hint_beats_floor():
    d = decide(chat("hi"), Tier.LIGHT, Tier.HEAVY)
    assert (d.tier, d.model) == (Tier.LIGHT, "l1")
    assert not any(s.startswith("min_tier") for s in d.signals)


def test_floored_tier_without_a_fitting_model_falls_up():
    raw = chat("hi", max_tokens=20, system="x" * 10000)
    assert decide(raw).tier is Tier.LIGHT
    d = decide(raw, floor=Tier.STANDARD)
    assert (d.tier, d.model) == (Tier.STANDARD, "h1")
    assert {"min_tier:standard", "fit:h1"} <= set(d.signals)


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


SKILL = (
    '[IMPORTANT: The user has invoked the "ops" skill, indicating they want you to follow '
    "its instructions. The full skill content is loaded below.]"
)
INSTR = "The user has provided the following instruction alongside the skill invocation: "
WRAPPED = CONFIG.model_copy(
    update={
        "wrapped_user_messages": (
            WrappedUserMessage(
                prefix="[IMPORTANT: The user has invoked the ",
                instruction_markers=(INSTR,),
                instruction_end="\n\n[Runtime note:",
            ),
        )
    }
)


def _scored(d) -> tuple[str, ...]:
    return tuple(s for s in d.signals if not s.startswith("fit:"))


SKILL_BODY = "Debug the deadlock and refactor for security.\n```sh\nkubectl get pods\n```\n" * 300


def test_wrapped_skill_body_is_not_scored_but_its_instruction_is():
    bare = Decider(CONFIG).decide(parse(chat(f"{SKILL}\n\n{SKILL_BODY}\n\n{INSTR}what time is it")))
    assert bare.tier is Tier.HEAVY
    d = Decider(WRAPPED).decide(parse(chat(f"{SKILL}\n\n{SKILL_BODY}\n\n{INSTR}what time is it")))
    assert d.tier is Tier.LIGHT
    assert _scored(d) == ("short", "wrapped")
    assert d.est_tokens == bare.est_tokens


def test_wrapped_instruction_keeps_its_own_signals_and_stops_at_the_end_marker():
    text = (
        f"{SKILL}\n\n{SKILL_BODY}\n\n{INSTR}find the root cause"
        "\n\n[Runtime note: deadlock refactor]"
    )
    d = Decider(WRAPPED).decide(parse(chat(text)))
    assert [s for s in d.signals if s.startswith("kw:")] == ["kw:root cause"]


def test_wrapped_message_without_instruction_is_skipped_and_ordinary_messages_still_count():
    d = Decider(WRAPPED).decide(parse(chat(f"{SKILL}\n\n{SKILL_BODY}", "please refactor this")))
    assert _scored(d) == ("kw:refactor", "wrapped")


def _ctx_decider(**scoring: int) -> Decider:
    return Decider(CONFIG.model_copy(update={"scoring": CONFIG.scoring.model_copy(update=scoring)}))


def test_context_signal_is_off_by_default():
    d = decide(chat("a " * 7000))
    assert not any(s.startswith("context:") for s in d.signals)


def test_context_size_adds_points_and_never_falls_as_conversation_grows():
    dec = _ctx_decider(large_context_tokens=100, very_large_context_tokens=1000)
    msg = "short " * 20
    scores = [dec.decide(parse(chat(*[msg] * n))) for n in (1, 5, 50, 200)]
    assert [d.score for d in scores] == sorted(d.score for d in scores)
    assert "context:large" in scores[1].signals and "context:very_large" not in scores[1].signals
    assert "context:very_large" in scores[-1].signals
    assert scores[-1].score - scores[0].score == 2


def test_very_large_context_needs_large_context():
    with pytest.raises(ValueError, match="very_large_context_tokens"):
        CONFIG.scoring.model_validate({"very_large_context_tokens": 10})


def _decide(cfg, effort=None, tier=Tier.LIGHT):
    body = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    if effort is not None:
        body["reasoning_effort"] = effort
    return Decider(cfg).decide(parse(orjson.dumps(body)), tier)


def _with_efforts(efforts, model="claude-haiku-5.5"):
    cfg = load_config()
    spec = cfg.models[model].model_copy(update={"efforts": efforts})
    return cfg.model_copy(update={"models": {**cfg.models, model: spec}})


def _all_efforts(efforts):
    cfg = load_config()
    models = {n: s.model_copy(update={"efforts": efforts}) for n, s in cfg.models.items()}
    return cfg.model_copy(update={"models": models})


def test_none_and_minimal_become_lowest_accepted_but_still_score_low():
    cfg = _with_efforts(("low", "medium", "high"))
    for word in ("none", "minimal"):
        dec = _decide(cfg, word)
        assert dec.model == "claude-haiku-5.5"
        assert dec.reasoning_effort == "low"
        assert "effort:low" in dec.signals


def test_none_does_not_fall_through_to_the_largest_window_model():
    # Every model takes low..max (as discovery reports for Copilot): `none` used to
    # match none of them and land on kimi-k3, still sending `none` upstream.
    dec = _decide(_all_efforts(("low", "medium", "high", "xhigh", "max")), "none")
    assert (dec.model, dec.reasoning_effort) == ("claude-haiku-5.5", "low")
    assert not [s for s in dec.signals if s.startswith("fit:")]


def test_accepted_effort_passes_through_and_absent_stays_absent():
    cfg = _with_efforts(("low", "medium", "high", "xhigh", "max"))
    assert _decide(cfg, "high").reasoning_effort == "high"
    assert _decide(cfg, "xhigh").reasoning_effort == "xhigh"
    assert _decide(cfg).reasoning_effort is None


def test_unknown_efforts_leave_the_word_untouched():
    assert _decide(_all_efforts(None), "none").reasoning_effort == "none"
    assert _decide(_all_efforts(None), "high").reasoning_effort == "high"


def test_model_without_effort_support_gets_the_field_dropped():
    cfg = _with_efforts((), model="claude-haiku-4.5")
    dec = Decider(cfg).decide(parse(chat("hi", reasoning_effort="low")), Tier.LIGHT, None)
    assert dec.model == "claude-haiku-5.5"  # the light model that accepts it is preferred
    alone = cfg.model_copy(update={"tiers": {**cfg.tiers, Tier.LIGHT: ("claude-haiku-4.5",)}})
    dec = Decider(alone).decide(parse(chat("hi", reasoning_effort="none")), Tier.LIGHT, None)
    assert (dec.model, dec.reasoning_effort) == ("claude-haiku-4.5", None)


def test_effort_no_model_accepts_is_clamped_on_the_tier_model():
    cfg = _all_efforts(("medium", "high"))
    got = [_decide(cfg, w) for w in ("none", "low", "max")]
    assert [d.reasoning_effort for d in got] == ["medium", "medium", "high"]
    assert {d.model for d in got} == {"claude-haiku-5.5"}  # not the largest-window fallback
