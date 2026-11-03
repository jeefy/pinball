"""Hot reload of the learned overlay without a restart.

A background task stats the overlay file every interval. When (inode, mtime,
size) changes the file is parsed and merged; success swaps `active`, a single
reference assignment, so each request sees one complete state. A rejected or
unreadable overlay leaves the last good state in place. A missing file means
base config only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anyio
from prometheus_client import CollectorRegistry, Counter, Gauge, Info

from pinball.config import Config
from pinball.overlay import Active, OverlayRejected, base_only, load_overlay, merge

log = logging.getLogger("pinball")

_Stamp = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ReloadMetrics:
    version: Info
    generated: Gauge
    reloads: Counter
    errors: Counter

    @classmethod
    def create(cls, r: CollectorRegistry) -> ReloadMetrics:
        return cls(
            version=Info(
                "pinball_overlay", "Active overlay version (empty: base only)", registry=r
            ),
            generated=Gauge(
                "pinball_overlay_generated_timestamp_seconds",
                "generated_at of the active overlay (0: base only)",
                registry=r,
            ),
            reloads=Counter(
                "pinball_overlay_reload_total", "Overlay reload attempts", ["result"], registry=r
            ),
            errors=Counter(
                "pinball_overlay_reload_errors_total", "Rejected overlays", ["reason"], registry=r
            ),
        )


class Reloader:
    def __init__(self, base: Config, path: Path | None, metrics: ReloadMetrics) -> None:
        self._base = base
        self._path = path
        self._metrics = metrics
        self._stamp: _Stamp | None = None
        self.active: Active = base_only(base)
        metrics.version.info({"version": ""})
        self.poll()

    def poll(self) -> None:
        """Reload if the file changed since the last poll."""
        if self._path is None:
            return
        try:
            st = self._path.stat()
        except FileNotFoundError:
            if self._stamp is not None:
                self._stamp = None
                self._swap(base_only(self._base), 0.0, "removed")
            return
        stamp = (st.st_ino, st.st_mtime_ns, st.st_size)
        if stamp == self._stamp:
            return
        self._stamp = stamp
        try:
            overlay = load_overlay(self._path)
            active = merge(self._base, overlay)
        except OverlayRejected as e:
            self._metrics.reloads.labels(result="rejected").inc()
            self._metrics.errors.labels(reason=e.reason).inc()
            log.error(
                "overlay rejected path=%s reason=%s detail=%s keeping=%r",
                self._path,
                e.reason,
                e.detail,
                self.active.version,
            )
            return
        except OSError as e:
            # Stamp is cleared so the next poll retries the read.
            self._stamp = None
            self._metrics.reloads.labels(result="rejected").inc()
            self._metrics.errors.labels(reason="io").inc()
            log.error("overlay unreadable path=%s error=%s", self._path, e)
            return
        self._swap(active, overlay.generated_at.timestamp(), "applied")

    def _swap(self, active: Active, generated: float, result: str) -> None:
        previous = self.active.version
        self.active = active
        self._metrics.version.info({"version": active.version})
        self._metrics.generated.set(generated)
        self._metrics.reloads.labels(result=result).inc()
        log.info(
            "overlay %s version=%r previous=%r callers=%d",
            result,
            active.version,
            previous,
            len(active.callers),
        )

    async def run(self, interval: float) -> None:
        while True:
            await anyio.sleep(interval)
            self.poll()
