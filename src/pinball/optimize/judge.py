"""Two-judge A/B comparison (ported from replay_auto.py).

The served/candidate order is randomised per (seed, sample, candidate) and the
same order is shown to every judge. A verdict field counts only when all
judges agree on it. When a judge shares a family with the candidate (gemini)
or the served model (claude-opus), the report also carries the verdict
without that judge (the "independent" view).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Final

SYSTEM_HEAD_CHARS: Final = 4000
TAIL_CHARS: Final = 24000  # ~6k tokens
REASON_MAX_CHARS: Final = 200

JUDGE_SYSTEM: Final = (
    "You compare two assistant replies, A and B, to the same conversation. The conversation "
    "was rebuilt as plain text from a longer agent session: tool calls and tool results are "
    "NOT shown, and long conversations are trimmed (system prompt head plus the most recent "
    "turns; a marker says so). Do not penalise a reply for not using tools or for lacking "
    "context that the trimming removed. For each reply, decide whether it is an acceptable "
    "answer to the final user message: it addresses what was asked, is accurate and free of "
    "fabricated claims, and respects the system prompt's instructions. Then pick the better "
    "reply, or tie. Respond with only a JSON object, no prose and no code fence:\n"
    '{"a_acceptable": true|false, "b_acceptable": true|false, "preference": "a"|"b"|"tie", '
    '"reason": "<one short sentence>"}'
)


def render_conversation(messages: tuple[dict[str, str], ...]) -> tuple[str, bool]:
    """System prompt head plus the last ~6k tokens of the rest; True when anything was cut."""
    turns = [f"[{m['role']}]\n{m['content']}" for m in messages]
    head = ""
    if messages and messages[0]["role"] == "system":
        head, turns = turns[0], turns[1:]
    body = "\n\n".join(turns)
    trimmed = len(head) > SYSTEM_HEAD_CHARS or len(body) > TAIL_CHARS
    head = head[:SYSTEM_HEAD_CHARS]
    if len(body) > TAIL_CHARS:
        body = body[-TAIL_CHARS:]
    marker = "[... conversation trimmed ...]" if trimmed else ""
    return "\n\n".join(p for p in (head, marker, body) if p), trimmed


def judge_prompt(messages: tuple[dict[str, str], ...], answer_a: str, answer_b: str) -> str:
    conversation, trimmed = render_conversation(messages)
    note = " (TRIMMED: system prompt head plus most recent turns only)" if trimmed else ""
    return f"CONVERSATION{note}:\n{conversation}\n\nANSWER A:\n{answer_a}\n\nANSWER B:\n{answer_b}"


def a_is_served(seed: int, log_id: str, candidate: str) -> bool:
    """Recorded, seeded A/B order for one (sample, candidate)."""
    return random.Random(f"{seed}:{log_id}:{candidate}").random() < 0.5


@dataclass(frozen=True, slots=True)
class Verdict:
    served_ok: bool
    candidate_ok: bool
    outcome: str  # candidate perspective: win | tie | loss
    reason: str


def parse_json_object(text: str) -> dict[str, Any] | None:
    """First-'{'-to-last-'}' JSON object in `text` (tolerates fences and prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def parse_judgement(text: str, a_served: bool) -> Verdict | None:
    """Parse the judge JSON and map A/B back to served/candidate; None when unusable."""
    raw = parse_json_object(text)
    if raw is None:
        return None
    a_ok, b_ok = _as_bool(raw.get("a_acceptable")), _as_bool(raw.get("b_acceptable"))
    preference = str(raw.get("preference") or "").strip().lower()
    if a_ok is None or b_ok is None or preference not in ("a", "b", "tie"):
        return None
    candidate_preferred = (preference == "b") == a_served
    outcome = "tie" if preference == "tie" else ("win" if candidate_preferred else "loss")
    served_ok, candidate_ok = (a_ok, b_ok) if a_served else (b_ok, a_ok)
    reason = str(raw.get("reason") or "").strip()[:REASON_MAX_CHARS]
    return Verdict(served_ok, candidate_ok, outcome, reason)


@dataclass(frozen=True, slots=True)
class Agreed:
    """Per-field consensus; a field is None when the judges disagree."""

    candidate_ok: bool | None
    served_ok: bool | None
    outcome: str | None


def agree(verdicts: list[Verdict | None]) -> Agreed | None:
    """A field counts only when every judge gave the same value; None if any judge failed."""
    if not verdicts or any(v is None for v in verdicts):
        return None
    usable = [v for v in verdicts if v is not None]

    def common(values: list[Any]) -> Any:
        return values[0] if len(set(values)) == 1 else None

    return Agreed(
        candidate_ok=common([v.candidate_ok for v in usable]),
        served_ok=common([v.served_ok for v in usable]),
        outcome=common([v.outcome for v in usable]),
    )


def same_family_judges(
    candidate: str, served: str | None, judges: tuple[str, ...]
) -> tuple[str, ...]:
    """Judges sharing a family with the candidate (gemini) or the served model (claude-opus)."""
    out: list[str] = []
    if candidate.startswith("gemini-") and "judge-argon" in judges:
        out.append("judge-argon")
    if (served or "").startswith("claude-opus") and "judge-opus" in judges:
        out.append("judge-opus")
    return tuple(out)
