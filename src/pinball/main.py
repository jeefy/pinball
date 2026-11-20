"""pinball entrypoint: HTTP ext_authz, /healthz and /metrics on one port.

Environment:
  PINBALL_HOST     0.0.0.0
  PINBALL_PORT     9201
  PINBALL_CONFIG   pinball.yaml path (default: the packaged copy)
  PINBALL_OVERLAY_PATH             learned overlay file (default: none, base only)
  PINBALL_OVERLAY_RELOAD_SECONDS   overlay poll interval (default: 30)
  PINBALL_DISCOVERY_URL            upstream /v1/models; set = build the pool from it
  PINBALL_DISCOVERY_TOKEN          bearer token for that URL (default: none)
  PINBALL_DISCOVERY_CACHE          last good /models answer (default: none)
  PINBALL_DISCOVERY_INTERVAL_SECONDS  re-discovery interval (default: 21600)
  LOG_LEVEL        INFO
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import anyio
import uvicorn

from pinball import discovery
from pinball.config import Config, load_config
from pinball.reload import Reloader
from pinball.server import Metrics, create_app


def main() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config_path = os.environ.get("PINBALL_CONFIG")
    static = load_config(Path(config_path) if config_path else None)
    url = os.environ.get("PINBALL_DISCOVERY_URL", "")
    token = os.environ.get("PINBALL_DISCOVERY_TOKEN", "")
    cache = os.environ.get("PINBALL_DISCOVERY_CACHE")
    cache_path = Path(cache) if cache else None
    every = float(os.environ.get("PINBALL_DISCOVERY_INTERVAL_SECONDS", "21600"))
    if every <= 0:
        raise SystemExit("PINBALL_DISCOVERY_INTERVAL_SECONDS must be > 0")
    config = initial_config(static, url, token, cache_path)
    overlay_path = os.environ.get("PINBALL_OVERLAY_PATH")
    interval = float(os.environ.get("PINBALL_OVERLAY_RELOAD_SECONDS", "30"))
    if interval <= 0:
        raise SystemExit("PINBALL_OVERLAY_RELOAD_SECONDS must be > 0")
    metrics = Metrics.create()
    reloader = Reloader(config, Path(overlay_path) if overlay_path else None, metrics.reload)
    background: list[Callable[[], Coroutine[Any, Any, None]]] = []
    if url:

        async def rediscover() -> None:
            while True:
                await anyio.sleep(every)
                fresh = await anyio.to_thread.run_sync(refresh, static, url, token, cache_path)
                if fresh is not None:
                    reloader.set_base(fresh)

        background.append(rediscover)
    app = create_app(
        reloader,
        metrics,
        interval if overlay_path else None,
        config.reply_signals,
        background,
        config.sticky,
    )
    logging.getLogger("pinball").info(
        "pinball starting models=%d tiers=%s overlay=%s version=%r",
        len(config.models),
        {t.value: list(m) for t, m in config.tiers.items()},
        overlay_path or "-",
        reloader.active.version,
    )
    uvicorn.run(
        app,
        host=os.environ.get("PINBALL_HOST", "0.0.0.0"),
        port=int(os.environ.get("PINBALL_PORT", "9201")),
        log_level=level.lower(),
        access_log=False,
        timeout_graceful_shutdown=10,
    )


def refresh(static: Config, url: str, token: str, cache: Path | None) -> Config | None:
    """A freshly discovered config (cached on success), or None on failure."""
    log = logging.getLogger("pinball")
    try:
        config, payload = discovery.discover(static, url, token)
    except (discovery.DiscoveryError, ValueError) as e:
        log.warning("discovery failed url=%s error=%s", url, e)
        return None
    if cache is not None:
        try:
            discovery.save_cache(cache, payload)
        except OSError as e:
            log.warning("discovery cache not written path=%s error=%s", cache, e)
    log.info(
        "discovered models=%d tiers=%s",
        len(config.models),
        {t.value: list(m) for t, m in config.tiers.items()},
    )
    return config


def initial_config(static: Config, url: str, token: str, cache: Path | None) -> Config:
    """Live pool, else the cached one, else the static pinball.yaml pool."""
    if not url:
        return static
    live = refresh(static, url, token, cache)
    if live is not None:
        return live
    if cache is not None and cache.is_file():
        try:
            config = discovery.from_cache(static, cache)
            logging.getLogger("pinball").info("serving cached pool path=%s", cache)
            return config
        except (discovery.DiscoveryError, OSError, ValueError) as e:
            logging.getLogger("pinball").warning("discovery cache unusable error=%s", e)
    return static
