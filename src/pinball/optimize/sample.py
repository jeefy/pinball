"""Payload normalisation and seeded per-tier sampling (ported from replay_auto.py).

The gateway log strips tool_calls, tool_call_id and tool definitions. Only
requests whose final message is a non-empty user turn and whose served reply
has text are kept; tool turns and empty assistant turns are dropped and the
sample is marked `agentic` (partially visible to the judges).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Final

from pinball.optimize.db import TIERS, ReplayDb

KEEP_ROLES: Final = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "assistant": "assistant",
}


@dataclass(frozen=True)
class Sample:
    log_id: str
    tier: str
    reason: str | None
    pinball_model: str | None
    served_model: str | None
    agentic: bool
    served_input_tokens: int
    served_output_tokens: int
    # In memory only; excluded from repr and from record().
    messages: tuple[dict[str, str], ...] = field(repr=False)
    served_text: str = field(repr=False)

    def record(self) -> dict[str, Any]:
        """Content-free description for reports."""
        return {
            "log_id": self.log_id,
            "tier": self.tier,
            "reason": self.reason,
            "pinball_model": self.pinball_model,
            "served_model": self.served_model,
            "agentic": self.agentic,
            "n_turns": len(self.messages),
        }


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(str(p) for p in parts).strip()
    return ""


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_payload(row: dict[str, Any]) -> Sample | None:
    """Rebuild a logged request as plain text turns, or None when it cannot be replayed."""
    request, response = row.get("request"), row.get("response")
    if not isinstance(request, list) or not isinstance(response, list):
        return None
    agentic = False
    turns: list[dict[str, str]] = []
    for message in request:
        if not isinstance(message, dict):
            continue
        role, text = str(message.get("role")), text_of(message.get("content"))
        if role not in KEEP_ROLES:
            agentic = agentic or role in ("tool", "function")
            continue
        if not text:
            agentic = agentic or role == "assistant"
            continue
        mapped = KEEP_ROLES[role]
        if turns and turns[-1]["role"] == mapped:
            turns[-1] = {"role": mapped, "content": f"{turns[-1]['content']}\n\n{text}"}
        else:
            turns.append({"role": mapped, "content": text})
    served = "\n".join(text_of(part) for part in response).strip()
    last = request[-1] if request else None
    last_is_user_text = (
        isinstance(last, dict) and last.get("role") == "user" and bool(text_of(last.get("content")))
    )
    if not turns or turns[-1]["role"] != "user" or not last_is_user_text or not served:
        return None
    return Sample(
        log_id=str(row["id"]),
        tier=str(row["tier"]),
        reason=row.get("reason"),
        pinball_model=row.get("pinball_model"),
        served_model=row.get("served_model"),
        agentic=agentic,
        served_input_tokens=_int(row.get("input_tokens")),
        served_output_tokens=_int(row.get("output_tokens")),
        messages=tuple(turns),
        served_text=served,
    )


def draw_samples(db: ReplayDb, days: int, per_tier: int, seed: int) -> list[Sample]:
    """Per tier: seeded shuffle of the pool, keep the first `per_tier` replayable rows."""
    by_tier: dict[str, list[str]] = defaultdict(list)
    for row in db.pool(days):
        by_tier[str(row["tier"])].append(str(row["id"]))
    samples: list[Sample] = []
    for tier in TIERS:
        ids = sorted(by_tier.get(tier, []))
        random.Random(f"{seed}:{tier}").shuffle(ids)
        kept: list[Sample] = []
        for start in range(0, len(ids), max(per_tier, 1)):
            if len(kept) >= per_tier:
                break
            chunk = ids[start : start + per_tier]
            fetched = {str(r["id"]): r for r in db.payloads(chunk)}
            for log_id in chunk:
                sample = normalize_payload(fetched[log_id]) if log_id in fetched else None
                if sample is not None and len(kept) < per_tier:
                    kept.append(sample)
        samples.extend(kept)
    return samples
