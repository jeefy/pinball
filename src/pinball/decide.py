"""Features -> Decision: a tier from heuristic signals, then a model that fits.

Stickiness without state: the per-message part of the score is the maximum
over every user-authored message in the conversation. Conversations only grow,
so that maximum never falls and a session never steps down a tier (no prompt
cache thrash). It can step up when the user asks for something harder. The
context-window fit is monotonic for the same reason. Both replicas compute
the same answer for the same body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, assert_never

from pinball.config import TIER_ORDER, Config, Tier, keyword_pattern
from pinball.features import Effort, Features

LONG_MESSAGE_CHARS: Final = 4000
VERY_LONG_MESSAGE_CHARS: Final = 12000
SHORT_MESSAGE_CHARS: Final = 160
MAX_KEYWORD_POINTS: Final = 2
SMALL_OUTPUT_TOKENS: Final = 512

_CODE: Final = re.compile(
    r"```|Traceback \(most recent call last\)|\bpanic: |^\s+at [\w.$<>]+\(",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class Decision:
    model: str
    tier: Tier
    score: int
    signals: tuple[str, ...]
    est_tokens: int

    @property
    def reason(self) -> str:
        return ",".join(self.signals) or "none"


@dataclass(frozen=True, slots=True)
class _MessageScore:
    score: int
    signals: tuple[str, ...]


class Decider:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._heavy = keyword_pattern(config.heavy_keywords)
        self._light = keyword_pattern(config.light_keywords)
        self._largest = max(config.models, key=lambda n: config.models[n].prompt_tokens)

    def decide(self, features: Features, hint: Tier | None = None) -> Decision:
        message = max(
            (self._score_message(t) for t in features.user_texts),
            key=lambda m: m.score,
            default=_MessageScore(0, ()),
        )
        request_score, request_signals = _score_request(features)
        score = message.score + request_score
        signals = [*message.signals, *request_signals]

        if hint is None:
            tier = self._tier_for(score)
        else:
            tier = hint
            signals.append(f"hint:{hint}")

        est = features.est_prompt_tokens
        model, fit = self._pick(tier, est, features.images > 0)
        if fit:
            signals.append(fit)
        return Decision(model, tier, score, tuple(signals), est)

    def _tier_for(self, score: int) -> Tier:
        t = self._config.thresholds
        if score >= t.heavy:
            return Tier.HEAVY
        if score >= t.standard:
            return Tier.STANDARD
        return Tier.LIGHT

    def _pick(self, tier: Tier, est_tokens: int, images: bool) -> tuple[str, str]:
        """First model in `tier` (then higher tiers) whose window and inputs fit."""
        cfg = self._config
        primary = cfg.tier_models(tier)[0]
        for t in TIER_ORDER[TIER_ORDER.index(tier) :]:
            for name in cfg.tier_models(t):
                spec = cfg.models[name]
                if est_tokens <= spec.prompt_tokens * cfg.headroom and (spec.vision or not images):
                    return name, "" if name == primary else f"fit:{name}"
        return self._largest, f"fit:{self._largest}"

    def _score_message(self, text: str) -> _MessageScore:
        score = 0
        signals: list[str] = []
        code = _CODE.search(text) is not None
        if code:
            score += 1
            signals.append("code")

        heavy = sorted(
            {m.group(0).lower() for m in self._heavy.finditer(text)} if self._heavy else ()
        )
        if heavy:
            score += min(len(heavy), MAX_KEYWORD_POINTS)
            signals.extend(f"kw:{w}" for w in heavy[:MAX_KEYWORD_POINTS])

        n = len(text)
        if n >= VERY_LONG_MESSAGE_CHARS:
            score += 2
            signals.append("very_long")
        elif n >= LONG_MESSAGE_CHARS:
            score += 1
            signals.append("long")

        light = self._light.search(text) if self._light else None
        if light:
            score -= 1
            signals.append(f"light:{light.group(0).lower()}")
        if n < SHORT_MESSAGE_CHARS and not code and not heavy:
            score -= 1
            signals.append("short")
        return _MessageScore(score, tuple(signals))


def _score_request(features: Features) -> tuple[int, list[str]]:
    score = 0
    signals: list[str] = []
    if features.tools:
        score += 1
        signals.append("tools")
    match features.effort:
        case Effort.HIGH:
            score += 2
            signals.append("effort:high")
        case Effort.LOW:
            score -= 1
            signals.append("effort:low")
        case Effort.MEDIUM | None:
            pass
        case unreachable:
            assert_never(unreachable)
    if features.max_output_tokens is not None and features.max_output_tokens <= SMALL_OUTPUT_TOKENS:
        score -= 1
        signals.append("small_output")
    if features.structured:
        score -= 1
        signals.append("structured")
    return score, signals
