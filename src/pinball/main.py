"""pinball entrypoint: HTTP ext_authz, /healthz and /metrics on one port.

Environment:
  PINBALL_HOST     0.0.0.0
  PINBALL_PORT     9201
  PINBALL_CONFIG   pinball.yaml path (default: the packaged copy)
  PINBALL_OVERLAY_PATH             learned overlay file (default: none, base only)
  PINBALL_OVERLAY_RELOAD_SECONDS   overlay poll interval (default: 30)
  LOG_LEVEL        INFO
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn

from pinball.config import load_config
from pinball.reload import Reloader
from pinball.server import Metrics, create_app


def main() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config_path = os.environ.get("PINBALL_CONFIG")
    config = load_config(Path(config_path) if config_path else None)
    overlay_path = os.environ.get("PINBALL_OVERLAY_PATH")
    interval = float(os.environ.get("PINBALL_OVERLAY_RELOAD_SECONDS", "30"))
    if interval <= 0:
        raise SystemExit("PINBALL_OVERLAY_RELOAD_SECONDS must be > 0")
    metrics = Metrics.create()
    reloader = Reloader(config, Path(overlay_path) if overlay_path else None, metrics.reload)
    app = create_app(reloader, metrics, interval if overlay_path else None, config.reply_signals)
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
