"""HTTP ext_authz endpoint for agentgateway.

agentgateway POSTs the client's request body here before resolving the LLM
model. The reply is always 200: a non-2xx answer from an ext_authz service
denies the client request, and pinball must never block traffic. The gateway
copies the JSON reply into `extauthz.*` (protocol.http.metadata) and the `auto`
virtual model routes on `extauthz.model`; an empty model selects its fallback
target.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

import anyio.to_thread
import orjson
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from pinball.config import Tier
from pinball.decide import Decider, Decision
from pinball.features import BodyError, parse
from pinball.overlay import Active
from pinball.reload import Reloader, ReloadMetrics

log = logging.getLogger("pinball")

HINT_HEADER: Final = "x-pinball-tier"
MIN_TIER_HEADER: Final = "x-pinball-min-tier"
CALLER_HEADER: Final = "x-agw-caller"
LISTENER_HEADER: Final = "x-pinball-listener"
_JSON: Final = "application/json"


@dataclass(frozen=True, slots=True)
class Metrics:
    registry: CollectorRegistry
    decisions: Counter
    signals: Counter
    skipped: Counter
    errors: Counter
    seconds: Histogram
    body_bytes: Histogram
    reload: ReloadMetrics

    @classmethod
    def create(cls) -> Metrics:
        r = CollectorRegistry()
        return cls(
            registry=r,
            decisions=Counter(
                "pinball_decisions_total", "Routing decisions", ["tier", "model"], registry=r
            ),
            signals=Counter(
                "pinball_signals_total", "Signals behind decisions", ["signal"], registry=r
            ),
            skipped=Counter(
                "pinball_skipped_total", "Bodies pinball could not read", ["reason"], registry=r
            ),
            errors=Counter("pinball_errors_total", "Unexpected errors (fail open)", registry=r),
            seconds=Histogram(
                "pinball_decide_seconds",
                "Parse + decide time",
                buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
                registry=r,
            ),
            body_bytes=Histogram(
                "pinball_body_bytes",
                "Request body size",
                buckets=(1e3, 1e4, 1e5, 1e6, 4e6, 16e6),
                registry=r,
            ),
            reload=ReloadMetrics.create(r),
        )


def _reply(model: str, tier: str, reason: str, score: int = 0, est_tokens: int = 0) -> Response:
    payload = {
        "model": model,
        "tier": tier,
        "reason": reason,
        "score": score,
        "est_tokens": est_tokens,
    }
    return Response(orjson.dumps(payload), media_type=_JSON)


def _header(request: Request, name: str) -> str:
    return request.headers.get(name, "")[:64]


def _hint(raw: str) -> Tier | None:
    try:
        return Tier(raw.strip().lower())
    except ValueError:
        return None


def create_app(
    source: Decider | Reloader, metrics: Metrics, reload_interval: float | None = None
) -> FastAPI:
    """`source` is a fixed Decider, or a Reloader whose active state is read per request.

    With a Reloader and `reload_interval`, the app polls the overlay in the
    background for its lifetime.
    """
    if isinstance(source, Decider):
        fixed = Active(source, {}, "")

        def current() -> Active:
            return fixed
    else:
        reloader = source

        def current() -> Active:
            return reloader.active

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if isinstance(source, Reloader) and reload_interval is not None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(source.run, reload_interval)
                yield
                tg.cancel_scope.cancel()
        else:
            yield

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    def decide(
        body: bytes, hint: Tier | None, floor: Tier | None, caller: str
    ) -> tuple[Decision, str, str]:
        # Read the reference once: the whole request uses one consistent state.
        active = current()
        decider, profile = active.select(caller)
        return decider.decide(parse(body), hint, floor), active.version, profile

    @app.get("/healthz")
    async def healthz() -> Response:
        return Response(b'{"ok":true}', media_type=_JSON)

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # agentgateway sends the check to whatever path protocol.http.path names
    # (the client's path when unset), so accept POST anywhere.
    @app.post("/{path:path}")
    async def check(request: Request) -> Response:
        start = time.perf_counter()
        body = await request.body()
        metrics.body_bytes.observe(len(body))
        caller = _header(request, CALLER_HEADER)
        listener = _header(request, LISTENER_HEADER)
        try:
            # Parsing a multi-MB body is CPU work; keep /healthz responsive.
            decision, overlay, profile = await anyio.to_thread.run_sync(
                decide,
                body,
                _hint(_header(request, HINT_HEADER)),
                _hint(_header(request, MIN_TIER_HEADER)),
                caller,
            )
        except BodyError as e:
            metrics.skipped.labels(reason=e.reason).inc()
            log.info("skip reason=%r body_bytes=%d listener=%s", e.reason, len(body), listener)
            return _reply("", "", f"skip:{e.reason}")
        except Exception:  # boundary: any failure must still answer 200 (fail open)
            metrics.errors.inc()
            log.exception("decide failed body_bytes=%d listener=%s", len(body), listener)
            return _reply("", "", "error")

        elapsed = time.perf_counter() - start
        metrics.seconds.observe(elapsed)
        metrics.decisions.labels(tier=decision.tier.value, model=decision.model).inc()
        for s in decision.signals:
            metrics.signals.labels(signal=s.split(":", 1)[0]).inc()
        log.info(
            "decide tier=%s model=%s score=%d est_tokens=%d body_bytes=%d ms=%.1f "
            "listener=%s caller=%s overlay=%s profile=%s reason=%s",
            decision.tier,
            decision.model,
            decision.score,
            decision.est_tokens,
            len(body),
            elapsed * 1000,
            listener,
            caller,
            overlay or "-",
            profile or "-",
            decision.reason,
        )
        return _reply(
            decision.model,
            decision.tier.value,
            decision.reason,
            decision.score,
            decision.est_tokens,
        )

    return app
