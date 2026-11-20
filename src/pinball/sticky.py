"""In-memory conversation memory, kept outside the reloadable routing state.

Key: a hash of the caller, the first user message and the opening exchange (the
first reply and the message after it: written text and tool-call ids). Stable for
a conversation, distinct for conversations that open with the same scaffolding,
and nothing readable is stored. Value: the tier and model last served. It lives in
the app, not in `Active`, so an overlay reload cannot drop it.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict

from pinball.config import TIER_ORDER, Sticky, Tier


class Sessions:
    def __init__(self, cfg: Sticky, clock=time.monotonic) -> None:
        self._cfg = cfg
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: OrderedDict[str, tuple[Tier, str, float]] = OrderedDict()

    @staticmethod
    def key(caller: str, first_user_text: str, opening: str) -> str:
        raw = f"{caller}\0{first_user_text}\0{opening}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> tuple[Tier, str] | None:
        with self._lock:
            hit = self._seen.get(key)
            if hit is None or self._clock() - hit[2] > self._cfg.ttl_seconds:
                self._seen.pop(key, None)
                return None
            return hit[0], hit[1]

    def put(self, key: str, tier: Tier, model: str) -> None:
        with self._lock:
            old = self._seen.get(key)
            if old and TIER_ORDER.index(old[0]) > TIER_ORDER.index(tier):
                tier = old[0]  # never remember a step down
            self._seen[key] = (tier, model, self._clock())
            self._seen.move_to_end(key)
            while len(self._seen) > self._cfg.sessions:
                self._seen.popitem(last=False)
