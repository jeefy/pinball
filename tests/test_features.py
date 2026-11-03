from __future__ import annotations

import orjson
import pytest

from pinball.features import BodyError, Effort, parse


def body(doc: dict) -> bytes:
    return orjson.dumps(doc)


def test_chat_completions_scores_only_user_text_when_tools_and_assistant_turns_present():
    # Given a chat-completions agent turn with system, tool calls and tool output
    doc = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the flaky test"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "bash", "arguments": '{"cmd":"ls"}'}}],
            },
            {"role": "tool", "content": "refactor debug security"},
        ],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
    }
    # When
    f = parse(body(doc))
    # Then
    assert f.user_texts == ("Fix the flaky test",)
    assert f.tools == 1
    assert f.model == "auto"
    tool_calls = orjson.dumps(doc["messages"][2]["tool_calls"])
    assert f.text_chars == len("You are an agent.") + len("Fix the flaky test") + len(
        tool_calls
    ) + len("refactor debug security")


def test_anthropic_messages_excludes_tool_result_and_counts_images():
    # Given an Anthropic Messages body with a tool_result block and an image
    doc = {
        "model": "auto",
        "system": [{"type": "text", "text": "sys"}],
        "max_tokens": 4096,
        "thinking": {"type": "enabled", "budget_tokens": 32000},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": "secret refactor"},
                    {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                    {"type": "text", "text": "what is in this picture"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "tool_use", "id": "t", "name": "x", "input": {"a": 1}},
                ],
            },
        ],
    }
    # When
    f = parse(body(doc))
    # Then
    assert f.user_texts == ("what is in this picture",)
    assert f.images == 1
    assert f.effort is Effort.HIGH
    assert f.max_output_tokens == 4096


def test_responses_api_string_input_is_the_user_message():
    # Given a Responses body with shorthand string input
    doc = {
        "model": "auto",
        "instructions": "be brief",
        "input": "hello there",
        "reasoning": {"effort": "minimal"},
        "max_output_tokens": 100,
        "text": {"format": {"type": "json_schema"}},
    }
    # When
    f = parse(body(doc))
    # Then
    assert f.user_texts == ("hello there",)
    assert f.effort is Effort.LOW
    assert f.max_output_tokens == 100
    assert f.structured is True


def test_responses_api_items_skip_function_call_output():
    # Given Responses input items including tool traffic
    doc = {
        "model": "auto",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "plan it"}]},
            {"type": "function_call", "name": "x", "arguments": '{"q":1}'},
            {"type": "function_call_output", "call_id": "c", "output": "debug output"},
        ],
    }
    # When
    f = parse(body(doc))
    # Then
    assert f.user_texts == ("plan it",)
    assert f.text_chars == len("plan it") + len('{"q":1}') + len("debug output")


@pytest.mark.parametrize(
    ("doc", "effort"),
    [
        ({"reasoning_effort": "high"}, Effort.HIGH),
        ({"reasoning_effort": "xhigh"}, Effort.HIGH),
        ({"output_config": {"effort": "medium"}}, Effort.MEDIUM),
        ({"thinking": {"type": "enabled", "budget_tokens": 2000}}, Effort.MEDIUM),
        ({"reasoning_effort": "bogus"}, None),
        ({}, None),
    ],
)
def test_effort_is_read_from_each_format(doc: dict, effort: Effort | None):
    assert parse(body({"model": "auto", **doc})).effort is effort


def test_max_completion_tokens_wins_over_max_tokens():
    f = parse(body({"model": "auto", "max_tokens": 9000, "max_completion_tokens": 50}))
    assert f.max_output_tokens == 50


def test_estimated_tokens_include_tools_and_images():
    # Given 350 text chars (100 tokens) and one image
    doc = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x" * 350},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ],
    }
    # When / Then
    assert parse(body(doc)).est_prompt_tokens == 100 + 1600


@pytest.mark.parametrize("raw", [b"not json", b"[1,2]", b""])
def test_non_object_bodies_raise_body_error(raw: bytes):
    with pytest.raises(BodyError):
        parse(raw)
