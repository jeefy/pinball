"""Gateway calls (candidate replay and judging) and the hard spend cap.

Spend control: before every call the budget RESERVES the worst case for that
call (input tokens estimated conservatively from characters, output tokens at
the call's max_tokens) at the model's price. A call that does not fit under
the cap is never issued. After the call the reservation is replaced by the
cost of the usage the gateway reported. Because output is bounded by
max_tokens and input is over-estimated, actual spend stays under the cap; if
the gateway ever reports more than was reserved, the actual figure is
recorded truthfully, `overrun` is set and the budget closes: no further call
is issued.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol

from pinball.optimize.settings import PriceTable

CHARS_PER_TOKEN_BOUND: Final = 2  # real tokenizers average ~4 chars/token
MESSAGE_OVERHEAD_TOKENS: Final = 16


class GatewayError(RuntimeError):
    """Content-free: `label` is safe to persist (HTTP status, transport, ...)."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class Gateway(Protocol):
    def chat(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int, caller: str
    ) -> Completion: ...


def input_token_bound(messages: list[dict[str, str]]) -> int:
    chars = sum(len(m.get("content", "")) for m in messages)
    return chars // CHARS_PER_TOKEN_BOUND + MESSAGE_OVERHEAD_TOKENS * (len(messages) + 1)


class Budget:
    """Thread-safe dollar cap with reserve/settle accounting."""

    def __init__(self, cap_usd: float, prices: PriceTable) -> None:
        self.cap = cap_usd
        self.prices = prices
        self.spent = 0.0
        self.reserved = 0.0
        self.calls = 0
        self.refused = 0
        self.overrun = False
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        return self.refused > 0 or self.overrun

    def reserve(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> float | None:
        """Worst-case cost if it fits under the cap (and is now held), else None."""
        worst = self.prices.cost(model, input_token_bound(messages), max_tokens)
        with self._lock:
            if self.overrun or self.spent + self.reserved + worst > self.cap:
                self.refused += 1
                return None
            self.reserved += worst
            return worst

    def settle(self, held: float, model: str, input_tokens: int, output_tokens: int) -> float:
        actual = self.prices.cost(model, input_tokens, output_tokens)
        with self._lock:
            self.reserved -= held
            self.spent += actual
            self.calls += 1
            if actual > held:
                self.overrun = True
        return actual

    def release(self, held: float) -> None:
        """A failed call: the gateway reported no usage, so nothing is charged."""
        with self._lock:
            self.reserved -= held


class BudgetExhausted(Exception):
    pass


def metered(
    gateway: Gateway,
    budget: Budget,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    caller: str,
    wire_model: str | None = None,
) -> Completion:
    """Price on `model`; send `wire_model` (a gateway alias) when given."""
    held = budget.reserve(model, messages, max_tokens)
    if held is None:
        raise BudgetExhausted(model)
    try:
        done = gateway.chat(wire_model or model, messages, max_tokens=max_tokens, caller=caller)
    except BaseException:
        budget.release(held)
        raise
    budget.settle(held, model, done.input_tokens, done.output_tokens)
    return done


def _merged_text(payload: dict[str, Any]) -> str:
    """Copilot can split one Claude turn over several choices; join their text."""
    choices = payload.get("choices") or []
    parts = [str((c.get("message") or {}).get("content") or "") for c in choices]
    return "\n".join(p for p in parts if p).strip()


class HttpGateway:
    """agentgateway /v1/chat/completions over stdlib HTTP (no extra dependency)."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        clock: Callable[[], float] | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        import time

        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._timeout = timeout
        self._clock = clock or time.monotonic
        self._open = opener or urllib.request.urlopen

    def chat(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int, caller: str
    ) -> Completion:
        body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens})
        headers = {"content-type": "application/json", "x-agw-caller": caller}
        request = urllib.request.Request(self._url, body.encode(), headers, method="POST")
        started = self._clock()
        try:
            with self._open(request, timeout=self._timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as e:
            # The error body can echo request text: keep only the status.
            raise GatewayError(f"HTTP {e.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise GatewayError(f"transport {type(e).__name__}") from None
        if not isinstance(payload, dict) or not payload.get("choices"):
            raise GatewayError("empty choices")
        text = _merged_text(payload)
        if not text:
            raise GatewayError("empty answer")
        usage = payload.get("usage") or {}
        return Completion(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=self._clock() - started,
        )
