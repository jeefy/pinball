"""Injected collaborators of a `pinball-optimize` run (tests swap every one)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TextIO

from pinball.optimize.bot import Bot
from pinball.optimize.caller_settings import CallerSettings
from pinball.optimize.db import ReplayDb
from pinball.optimize.discovery_cli import DiscoveryEnv, make_bot
from pinball.optimize.gateway import Gateway


@dataclass
class Deps:
    db: ReplayDb
    gateway: Gateway | None  # None: building one is forbidden (dry-run)
    now: datetime
    out: TextIO
    callers: CallerSettings = field(default_factory=CallerSettings)
    discovery: DiscoveryEnv | None = None
    bot_factory: Callable[[DiscoveryEnv], Bot] = make_bot
    catalog: Callable[[], Any] | None = None
