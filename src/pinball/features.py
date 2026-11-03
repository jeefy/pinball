"""Request body -> Features, for OpenAI chat completions, Anthropic Messages and
OpenAI Responses bodies alike.

Only counts and the text of user-authored messages leave this module. Tool
output, tool arguments, assistant turns and system prompts are measured
(prompt-size estimate) but never scored: an agent's `cat` of a file full of
the word "refactor" must not change the tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import orjson

type Json = dict[str, Json] | list[Json] | str | int | float | bool | None

# Roughly what one image costs on Copilot's Claude models (a ~1 MP image).
IMAGE_TOKENS = 1600
CHARS_PER_TOKEN = 3.5
THINKING_HIGH_BUDGET = 16000


class Effort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_EFFORT_WORDS: dict[str, Effort] = {
    "none": Effort.LOW,
    "minimal": Effort.LOW,
    "low": Effort.LOW,
    "medium": Effort.MEDIUM,
    "high": Effort.HIGH,
    "xhigh": Effort.HIGH,
    "max": Effort.HIGH,
}


@dataclass(frozen=True, slots=True)
class BodyError(Exception):
    """The body is not a JSON object; the caller falls back to the gateway default."""

    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class Features:
    model: str
    user_texts: tuple[str, ...]
    text_chars: int
    images: int
    tools: int
    tools_chars: int
    effort: Effort | None
    max_output_tokens: int | None
    structured: bool

    @property
    def est_prompt_tokens(self) -> int:
        chars = self.text_chars + self.tools_chars
        return int(chars / CHARS_PER_TOKEN) + self.images * IMAGE_TOKENS


@dataclass(slots=True)
class _Acc:
    """Mutable accumulator for one walk over the body."""

    user_texts: list[str] = field(default_factory=list)
    text_chars: int = 0
    images: int = 0


def parse(body: bytes) -> Features:
    try:
        doc: Json = orjson.loads(body)
    except orjson.JSONDecodeError as e:
        raise BodyError("body is not JSON") from e
    if not isinstance(doc, dict):
        raise BodyError("body is not a JSON object")

    acc = _Acc()
    _walk_system(doc.get("system"), acc)
    _walk_system(doc.get("instructions"), acc)
    _walk_messages(doc.get("messages"), acc)
    _walk_messages(doc.get("input"), acc)

    tools = doc.get("tools") or doc.get("functions")
    tools_list = tools if isinstance(tools, list) else []
    model = doc.get("model")
    return Features(
        model=model if isinstance(model, str) else "",
        user_texts=tuple(acc.user_texts),
        text_chars=acc.text_chars,
        images=acc.images,
        tools=len(tools_list),
        tools_chars=len(orjson.dumps(tools_list)) if tools_list else 0,
        effort=_effort(doc),
        max_output_tokens=_max_output_tokens(doc),
        structured=_structured(doc),
    )


def _walk_system(value: Json, acc: _Acc) -> None:
    match value:
        case str(s):
            acc.text_chars += len(s)
        case list(blocks):
            for b in blocks:
                _block(b, acc, None)
        case _:
            pass


def _walk_messages(value: Json, acc: _Acc) -> None:
    match value:
        case str(s):
            # Responses API shorthand: `input` is the user's message.
            acc.text_chars += len(s)
            if s.strip():
                acc.user_texts.append(s)
        case list(items):
            for item in items:
                _message(item, acc)
        case _:
            pass


def _message(item: Json, acc: _Acc) -> None:
    match item:
        case {"role": "user", **rest}:
            texts: list[str] = []
            _content(rest.get("content"), acc, texts)
            joined = "\n".join(t for t in texts if t.strip())
            if joined:
                acc.user_texts.append(joined)
        case {"role": str(), **rest}:
            _content(rest.get("content"), acc, None)
            _measure(rest.get("tool_calls"), acc)
        case dict():
            # Responses API items without a role (function_call, *_output, reasoning).
            _block(item, acc, None)
        case _:
            pass


def _content(content: Json, acc: _Acc, user: list[str] | None) -> None:
    match content:
        case str(s):
            acc.text_chars += len(s)
            if user is not None:
                user.append(s)
        case list(blocks):
            for b in blocks:
                _block(b, acc, user)
        case _:
            pass


def _block(block: Json, acc: _Acc, user: list[str] | None) -> None:
    """One content block. `user` collects user-authored text; None elsewhere."""
    match block:
        case str(s):
            acc.text_chars += len(s)
            if user is not None:
                user.append(s)
        case {"type": "text" | "input_text" | "output_text", "text": str(t)}:
            acc.text_chars += len(t)
            if user is not None:
                user.append(t)
        case {"type": "image_url" | "image" | "input_image"}:
            acc.images += 1
        case {"type": "tool_result", **rest}:
            _content(rest.get("content"), acc, None)
        case {"type": "function_call_output", **rest}:
            _content(rest.get("output"), acc, None)
        case {"type": "thinking", "thinking": str(t)}:
            acc.text_chars += len(t)
        case {"type": "tool_use", **rest}:
            _measure(rest.get("input"), acc)
        case {"type": "function_call", **rest}:
            _measure(rest.get("arguments"), acc)
        case _:
            pass


def _measure(value: Json, acc: _Acc) -> None:
    match value:
        case None:
            pass
        case str(s):
            acc.text_chars += len(s)
        case _:
            acc.text_chars += len(orjson.dumps(value))


def _effort(doc: dict[str, Json]) -> Effort | None:
    match doc:
        case {"reasoning_effort": str(word)}:
            return _EFFORT_WORDS.get(word.lower())
        case {"reasoning": {"effort": str(word)}}:
            return _EFFORT_WORDS.get(word.lower())
        case {"output_config": {"effort": str(word)}}:
            return _EFFORT_WORDS.get(word.lower())
        case {"thinking": {"type": "enabled", "budget_tokens": int(budget)}}:
            return Effort.HIGH if budget >= THINKING_HIGH_BUDGET else Effort.MEDIUM
        case _:
            return None


def _max_output_tokens(doc: dict[str, Json]) -> int | None:
    for key in ("max_completion_tokens", "max_output_tokens", "max_tokens"):
        value = doc.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _structured(doc: dict[str, Json]) -> bool:
    match doc:
        case {"response_format": {"type": "json_object" | "json_schema"}}:
            return True
        case {"text": {"format": {"type": "json_object" | "json_schema"}}}:
            return True
        case _:
            return False
