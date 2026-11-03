"""Line-based edits of the carrier manifests that keep every comment and byte around them.

The carrier files (k8s/pinball.yaml, k8s/agentgateway.yaml,
k8s/pinball-candidates.yaml) are heavily commented, so they are never
re-serialised: each edit inserts, rewrites or deletes exactly one line next to
an anchor and raises EditError when the anchor is not where it is expected.
Model ids come from upstream; `model_id()` is the only way in, so nothing but
`[a-z0-9._-]` ever reaches a YAML line.
"""

from __future__ import annotations

import re
from typing import NewType

ModelId = NewType("ModelId", str)

_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}")
KIMI_PROVIDER = "copilot-completions-kimi"
DEFAULT_PROVIDER = "copilot-completions"


class EditError(ValueError):
    pass


def model_id(raw: str) -> ModelId:
    if not _ID.fullmatch(raw):
        raise EditError(f"unsafe model id {raw!r}")
    return ModelId(raw)


def provider_for(mid: ModelId) -> str:
    return KIMI_PROVIDER if mid.startswith("kimi-") else DEFAULT_PROVIDER


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_entry(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith("#")


def _find(lines: list[str], pattern: str, start: int = 0) -> int:
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.match(lines[i]):
            return i
    raise EditError(f"anchor {pattern!r} not found")


def _block_entries(lines: list[str], header: int) -> list[int]:
    """Indices of the non-comment lines nested under `lines[header]`."""
    base = _indent(lines[header])
    out: list[int] = []
    for i in range(header + 1, len(lines)):
        if not _is_entry(lines[i]):
            continue
        if _indent(lines[i]) <= base:
            break
        out.append(i)
    return out


def _insert_after(lines: list[str], i: int, new: str) -> str:
    return "".join([*lines[: i + 1], new + "\n", *lines[i + 1 :]])


def _last(lines: list[str], pattern: str) -> int:
    rx = re.compile(pattern)
    hits = [i for i, line in enumerate(lines) if rx.match(line)]
    if not hits:
        raise EditError(f"anchor {pattern!r} not found")
    return hits[-1]


def _gateway_line(prefix: str, mid: ModelId, internal: bool) -> str:
    vis = " visibility: internal," if internal else ""
    ref = provider_for(mid)
    return (
        f"- {{name: {prefix}{mid},{vis} provider: {{reference: {ref}}}, params: {{model: {mid}}}}}"
    )


# --- k8s/agentgateway.yaml -------------------------------------------------------------


def add_opt_alias(text: str, mid: ModelId) -> str:
    lines = text.splitlines(keepends=True)
    i = _last(lines, r"\s*- \{name: opt-")
    line = " " * _indent(lines[i]) + _gateway_line("opt-", mid, internal=False)
    return _insert_after(lines, i, line)


def remove_opt_alias(text: str, mid: ModelId) -> str:
    lines = text.splitlines(keepends=True)
    i = _find(lines, rf"\s*- \{{name: opt-{re.escape(mid)},")
    return "".join(lines[:i] + lines[i + 1 :])


def add_pinball_model(text: str, mid: ModelId) -> str:
    lines = text.splitlines(keepends=True)
    i = _last(lines, r"\s*- \{name: pinball-")
    line = " " * _indent(lines[i]) + _gateway_line("pinball-", mid, internal=True)
    return _insert_after(lines, i, line)


def add_auto_target(text: str, mid: ModelId) -> str:
    """Before the unconditional fallback of `auto` (the `pinball` alias shares it by anchor)."""
    lines = text.splitlines(keepends=True)
    auto = _find(lines, r"\s*- name: auto\s*$")
    fallback = _find(lines, r"\s*- \{model: [^,{}]+\}\s*$", auto)
    when = f'\'default(extauthz.model, "") == "{mid}"\''
    line = " " * _indent(lines[fallback]) + f"- {{when: {when}, model: pinball-{mid}}}"
    return _insert_after(lines, fallback - 1, line)


def add_rates(text: str, mid: ModelId, rates: tuple[str, str]) -> str:
    """Placeholder cost-catalog rates (USD/1M) under modelCatalog providers.openai.models."""
    lines = text.splitlines(keepends=True)
    catalog = _find(lines, r"\s*modelCatalog:\s*$")
    openai = _find(lines, r"\s*openai:\s*$", catalog)
    models = _find(lines, r"\s*models:\s*$", openai)
    entries = _block_entries(lines, models)
    if not entries:
        raise EditError("openai rates block is empty")
    rate = f'{{rates: {{input: "{rates[0]}", output: "{rates[1]}"}}}}'
    line = " " * _indent(lines[entries[0]]) + f"{mid}: {rate}"
    return _insert_after(lines, entries[-1], line)


# --- k8s/pinball.yaml ------------------------------------------------------------------


def add_pool_model(text: str, mid: ModelId, prompt_tokens: int, vision: bool) -> str:
    lines = text.splitlines(keepends=True)
    header = _find(lines, r"\s*models:\s*$", _find(lines, r"\s*pinball\.yaml: \|"))
    entries = _block_entries(lines, header)
    if not entries:
        raise EditError("pinball models block is empty")
    spec = f"{{prompt_tokens: {prompt_tokens}, vision: {'true' if vision else 'false'}}}"
    line = " " * _indent(lines[entries[0]]) + f"{mid}: {spec}"
    return _insert_after(lines, entries[-1], line)


_TIER = r"(\s*{tier}:\s*\[)([^\]]*)(\].*)"


def append_to_tier(text: str, tier: str, mid: ModelId) -> str:
    """Last position: the nightly overlay may then promote it within the tier."""
    lines = text.splitlines(keepends=True)
    header = _find(lines, r"\s*tiers:\s*$", _find(lines, r"\s*pinball\.yaml: \|"))
    rx = re.compile(_TIER.format(tier=re.escape(tier)))
    for i in _block_entries(lines, header):
        m = rx.fullmatch(lines[i].rstrip("\n"))
        if m:
            members = [x.strip() for x in m.group(2).split(",") if x.strip()]
            if mid in members:
                raise EditError(f"{mid} already in tier {tier}")
            body = ", ".join([*members, mid])
            nl = "\n" if lines[i].endswith("\n") else ""
            lines[i] = f"{m.group(1)}{body}{m.group(3)}{nl}"
            return "".join(lines)
    raise EditError(f"tier {tier} is not a one-line flow list")


# --- k8s/pinball-candidates.yaml (sections `candidates:` and `dropped:`) --------------


def add_list_entry(text: str, section: str, entry: str) -> str:
    """Append the flow mapping `entry` to the block list `section` (`[]` when empty)."""
    lines = text.splitlines(keepends=True)
    i = _find(lines, rf"\s*{re.escape(section)}:(\s*\[\])?\s*$")
    if lines[i].rstrip().endswith("[]"):
        nl = "\n" if lines[i].endswith("\n") else ""
        lines[i] = lines[i].rstrip()[:-2].rstrip() + nl
    entries = _block_entries(lines, i)
    line = " " * (_indent(lines[i]) + 2) + f"- {entry}"
    return _insert_after(lines, entries[-1] if entries else i, line)


def remove_list_entry(text: str, section: str, mid: ModelId) -> str:
    lines = text.splitlines(keepends=True)
    header = _find(lines, rf"\s*{re.escape(section)}:\s*$")
    hits = [
        i
        for i in _block_entries(lines, header)
        if re.match(rf"\s*- \{{id: {re.escape(mid)},", lines[i])
    ]
    if len(hits) != 1:
        raise EditError(f"{mid} not listed once under {section}")
    del lines[hits[0]]
    if not _block_entries(lines, header):
        nl = "\n" if lines[header].endswith("\n") else ""
        lines[header] = lines[header].rstrip() + " []" + nl
    return "".join(lines)
