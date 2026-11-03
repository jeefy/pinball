"""pinball entrypoint: HTTP ext_authz, /healthz and /metrics on one port.

Environment:
  PINBALL_HOST     0.0.0.0
  PINBALL_PORT     9201
  PINBALL_CONFIG   pinball.yaml path (default: the packaged copy)
  LOG_LEVEL        INFO
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn

from pinball.config import load_config
from pinball.decide import Decider
from pinball.server import Metrics, create_app


def main() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config_path = os.environ.get("PINBALL_CONFIG")
    config = load_config(Path(config_path) if config_path else None)
    app = create_app(Decider(config), Metrics.create())
    logging.getLogger("pinball").info(
        "pinball starting models=%d tiers=%s",
        len(config.models),
        {t.value: list(m) for t, m in config.tiers.items()},
    )
    uvicorn.run(
        app,
        host=os.environ.get("PINBALL_HOST", "0.0.0.0"),
        port=int(os.environ.get("PINBALL_PORT", "9201")),
        log_level=level.lower(),
        access_log=False,
        timeout_graceful_shutdown=10,
    )
