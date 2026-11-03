"""New-model discovery: list what Copilot serves and diff it against pool + candidates.

copilot-api's /v1/models answers `{"data": [{"id": ..., ...}]}`. Upstream Copilot
entries carry `capabilities` (`type`, `limits.max_prompt_tokens`,
`supports.vision`); a slimmed answer may drop them. Chat capability is
`capabilities.type == "chat"` when present, else "not an embeddings id".
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final

from pinball.optimize.carrier_edit import EditError, ModelId, model_id

DEFAULT_MODELS_URL: Final = "http://copilot-api.carrier.svc.cluster.local:4141/v1/models"
# Embeddings, dated snapshots (aliases of a newer id) and legacy GPT-3.5/4/4o.
DEFAULT_IGNORE: Final = r"(embed|-\d{4}-\d{2}-\d{2}$|-\d{4}$|^gpt-3\.5|^gpt-4(o)?(-|$))"
# prompt_tokens is a cost cap in carrier (k8s/pinball.yaml), never a window.
PROMPT_TOKENS_CAP: Final = 250_000
PROMPT_TOKENS_DEFAULT: Final = 100_000


class CatalogError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CatalogModel:
    id: ModelId
    prompt_tokens: int
    vision: bool
    defaulted: tuple[str, ...]  # fields filled with the conservative default

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_tokens": self.prompt_tokens,
            "vision": self.vision,
            "defaulted": list(self.defaulted),
        }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _chat_model(raw: dict[str, Any], ignore: re.Pattern[str]) -> CatalogModel | None:
    try:
        mid = model_id(str(raw.get("id", "")))
    except EditError:
        return None
    caps = _mapping(raw.get("capabilities"))
    kind = caps.get("type")
    if (kind is not None and kind != "chat") or ignore.search(mid):
        return None
    if raw.get("model_picker_enabled") is False:
        return None
    limits = _mapping(caps.get("limits"))
    supports = _mapping(caps.get("supports"))
    defaulted: list[str] = []
    window = limits.get("max_prompt_tokens")
    if isinstance(window, int) and window > 0:
        prompt_tokens = min(window, PROMPT_TOKENS_CAP)
    else:
        prompt_tokens = PROMPT_TOKENS_DEFAULT
        defaulted.append("prompt_tokens")
    vision = supports.get("vision")
    if not isinstance(vision, bool):
        vision = False
        defaulted.append("vision")
    return CatalogModel(mid, prompt_tokens, vision, tuple(defaulted))


def parse_catalog(payload: Any, ignore: str = DEFAULT_IGNORE) -> dict[ModelId, CatalogModel]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise CatalogError("models answer has no data list")
    rx = re.compile(ignore)
    out: dict[ModelId, CatalogModel] = {}
    for raw in payload["data"]:
        if isinstance(raw, dict):
            m = _chat_model(raw, rx)
            if m is not None:
                out.setdefault(m.id, m)
    return out


def fetch_catalog(url: str, opener: Callable[..., Any] | None = None, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise CatalogError(f"models HTTP {e.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise CatalogError(f"models transport {type(e).__name__}") from None


def new_models(
    catalog: dict[ModelId, CatalogModel], known: Iterable[ModelId]
) -> list[CatalogModel]:
    """Chat models in the catalog that are not pool, candidate or dropped; sorted by id."""
    seen = set(known)
    return [catalog[m] for m in sorted(catalog) if m not in seen]
