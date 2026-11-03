"""Optimizer settings from the environment, and the per-model price table."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

CALLER: Final = "eval-replay"
JUDGE_CALLER: Final = "eval-replay-judge"
DEFAULT_JUDGES: Final = ("judge-opus", "judge-argon")


class Price(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)


# ESTIMATES, not invoices: Copilot bills by subscription, so these are notional
# list-price-like figures rounded UP, used only to bound the nightly spend and
# to compare models by cost. Override with PINBALL_OPT_PRICES.
DEFAULT_PRICES: Final[dict[str, Price]] = {
    "claude-haiku-5.5": Price(input=1.0, output=5.0),
    "claude-haiku-4.5": Price(input=1.0, output=5.0),
    "claude-sonnet-5.5": Price(input=3.0, output=15.0),
    "claude-sonnet-5": Price(input=3.0, output=15.0),
    "gemini-3.8-flash": Price(input=1.0, output=4.0),
    "claude-opus-5.5": Price(input=15.0, output=75.0),
    "claude-opus-5": Price(input=15.0, output=75.0),
    "claude-opus-4.8": Price(input=15.0, output=75.0),
    "claude-fable-5.1": Price(input=15.0, output=75.0),
    "claude-fable-5": Price(input=15.0, output=75.0),
    "gemini-4-argon": Price(input=4.0, output=20.0),
    "kimi-k3": Price(input=1.0, output=4.0),
    "judge-opus": Price(input=15.0, output=75.0),
    "judge-argon": Price(input=4.0, output=20.0),
}
# Unknown models are priced like the most expensive tier, so they never look cheap.
UNKNOWN_PRICE: Final = Price(input=15.0, output=75.0)


class PriceTable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    models: dict[str, Price] = DEFAULT_PRICES
    default: Price = UNKNOWN_PRICE

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self.models.get(model, self.default)
        return (input_tokens * p.input + output_tokens * p.output) / 1_000_000


def load_prices(path: Path | None) -> PriceTable:
    """Defaults, with a YAML file `{models: {name: {input, output}}, default: {...}}` on top."""
    if path is None:
        return PriceTable()
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    override = PriceTable.model_validate(raw)
    return PriceTable(
        models={**DEFAULT_PRICES, **override.models},
        default=override.default if "default" in raw else UNKNOWN_PRICE,
    )


class RankSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_samples: int = Field(default=10, ge=1)  # non-agentic agreed comparisons
    shrink_k: float = Field(default=20.0, ge=0)  # pseudo-samples of the current-order prior
    prior_top: float = Field(default=0.75, ge=0, le=1)  # prior quality of position 0
    prior_step: float = Field(default=0.05, ge=0)  # prior drop per position
    cost_weight: float = Field(default=0.1, ge=0)  # penalty per (cost / tier median cost)
    margin: float = Field(default=0.02, ge=0)  # score gap needed to swap neighbours
    max_moves: int = Field(default=2, ge=0)  # adjacent swaps per tier per night
    agentic_share_high: float = Field(default=0.5, ge=0, le=1)
    agentic_max_moves: int = Field(default=1, ge=0)


class RevertSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hours: float = Field(default=6.0, gt=0)
    error_margin: float = Field(default=0.5, ge=0)  # relative rise in non-2xx share
    error_floor: float = Field(default=0.01, ge=0)  # absolute rise that is always noise
    cost_margin: float = Field(default=0.25, ge=0)  # relative rise in cost per request
    min_requests: int = Field(default=50, ge=1)  # per window


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    db_dsn: str = ""
    gateway_url: str = ""
    gateway_timeout: float = Field(default=300.0, gt=0)
    budget_usd: float = Field(default=10.0, ge=0)
    overlay_path: Path | None = None
    config_path: Path | None = None
    audit_path: Path | None = None
    state_path: Path | None = None
    judges: tuple[str, ...] = DEFAULT_JUDGES
    model_prefix: str = ""  # candidates reach the gateway as <prefix><pool-id>
    workers: int = Field(default=4, ge=1)
    prices: PriceTable = PriceTable()
    rank: RankSettings = RankSettings()
    revert: RevertSettings = RevertSettings()

    @property
    def audit_file(self) -> Path | None:
        if self.audit_path is not None or self.overlay_path is None:
            return self.audit_path
        return self.overlay_path.with_name("optimize-audit.jsonl")

    @property
    def state_file(self) -> Path | None:
        if self.state_path is not None or self.overlay_path is None:
            return self.state_path
        return self.overlay_path.with_name(self.overlay_path.name + ".state.json")


_RANK_ENV: Final = {
    "PINBALL_OPT_MIN_SAMPLES": "min_samples",
    "PINBALL_OPT_SHRINK_K": "shrink_k",
    "PINBALL_OPT_COST_WEIGHT": "cost_weight",
    "PINBALL_OPT_SWAP_MARGIN": "margin",
    "PINBALL_OPT_MAX_MOVES": "max_moves",
    "PINBALL_OPT_AGENTIC_SHARE_HIGH": "agentic_share_high",
    "PINBALL_OPT_AGENTIC_MAX_MOVES": "agentic_max_moves",
}
_REVERT_ENV: Final = {
    "PINBALL_OPT_REVERT_HOURS": "hours",
    "PINBALL_OPT_REVERT_ERROR_MARGIN": "error_margin",
    "PINBALL_OPT_REVERT_ERROR_FLOOR": "error_floor",
    "PINBALL_OPT_REVERT_COST_MARGIN": "cost_margin",
    "PINBALL_OPT_REVERT_MIN_REQUESTS": "min_requests",
}
_TOP_ENV: Final = {
    "PINBALL_OPT_DB_DSN": "db_dsn",
    "PINBALL_OPT_GATEWAY_URL": "gateway_url",
    "PINBALL_OPT_GATEWAY_TIMEOUT": "gateway_timeout",
    "PINBALL_OPT_BUDGET_USD": "budget_usd",
    "PINBALL_OVERLAY_PATH": "overlay_path",
    "PINBALL_CONFIG": "config_path",
    "PINBALL_OPT_AUDIT_PATH": "audit_path",
    "PINBALL_OPT_STATE_PATH": "state_path",
    "PINBALL_OPT_WORKERS": "workers",
    "PINBALL_OPT_MODEL_PREFIX": "model_prefix",
}


def _pick(env: Mapping[str, str], names: Mapping[str, str]) -> dict[str, str]:
    return {field: env[var] for var, field in names.items() if env.get(var)}


def settings_from_env(env: Mapping[str, str]) -> Settings:
    data: dict[str, object] = dict(_pick(env, _TOP_ENV))
    data["rank"] = RankSettings.model_validate(_pick(env, _RANK_ENV))
    data["revert"] = RevertSettings.model_validate(_pick(env, _REVERT_ENV))
    if env.get("PINBALL_OPT_JUDGES"):
        data["judges"] = tuple(j for j in env["PINBALL_OPT_JUDGES"].split(",") if j)
    prices = env.get("PINBALL_OPT_PRICES")
    data["prices"] = load_prices(Path(prices) if prices else None)
    return Settings.model_validate(data)
