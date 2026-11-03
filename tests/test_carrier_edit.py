from __future__ import annotations

import difflib
import os
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from pinball.optimize import carrier_edit as edit
from pinball.optimize.bot import candidate_change, drop_change, promote_change
from pinball.optimize.carrier_edit import EditError, model_id
from pinball.optimize.discover import CatalogModel
from pinball.optimize.invariants import (
    PATHS,
    CarrierFiles,
    InvariantError,
    candidate_lists,
    check,
    configmap_data,
)

# CARRIER_ROOT=<carrier checkout> runs these against the real files instead.
ROOT = Path(os.environ.get("CARRIER_ROOT") or Path(__file__).parent / "fixtures" / "carrier")
TODAY = date(2026, 11, 3)
NEW = CatalogModel(model_id("gpt-7-nova"), 200000, True, ())
KIMI = CatalogModel(model_id("kimi-k4"), 100000, False, ("prompt_tokens", "vision"))


def carrier() -> CarrierFiles:
    return CarrierFiles(**{k: (ROOT / p).read_text() for k, p in PATHS.items()})


def added_lines(before: str, after: str) -> list[str]:
    """Lines only in `after`; fails if any line of `before` was removed or changed."""
    diff = list(difflib.ndiff(before.splitlines(keepends=True), after.splitlines(keepends=True)))
    return [d[2:] for d in diff if d.startswith("+ ")]


def changed(before: str, after: str) -> tuple[list[str], list[str]]:
    diff = list(difflib.ndiff(before.splitlines(keepends=True), after.splitlines(keepends=True)))
    return [d[2:] for d in diff if d.startswith("- ")], [d[2:] for d in diff if d.startswith("+ ")]


def test_real_carrier_files_hold_the_invariants():
    check(carrier())


def test_step_a_adds_only_the_alias_and_the_candidate_entry():
    before = carrier()
    after = candidate_change(NEW, TODAY).edit(before)
    assert after.pinball == before.pinball
    assert changed(before.gateway, after.gateway)[0] == []
    assert added_lines(before.gateway, after.gateway) == [
        "        - {name: opt-gpt-7-nova, provider: {reference: copilot-completions},"
        " params: {model: gpt-7-nova}}\n"
    ]
    assert candidate_lists(after.candidates).candidates == ("gpt-7-nova",)
    check(after)


def test_step_a_routes_kimi_through_the_kimi_provider():
    after = candidate_change(KIMI, TODAY).edit(carrier())
    models = configmap_data(after.gateway, "config.yaml")["llm"]["models"]
    (alias,) = [m for m in models if m["name"] == "opt-kimi-k4"]
    assert alias["provider"]["reference"] == "copilot-completions-kimi"
    check(after)


def test_step_b_wires_the_winner_into_the_pool_at_the_last_tier_slot():
    staged = candidate_change(NEW, TODAY).edit(carrier())
    after = promote_change(NEW, "standard", ("3.00", "15.00"), {}).edit(staged)
    check(after)
    pool = configmap_data(after.pinball, "pinball.yaml")
    assert pool["models"]["gpt-7-nova"] == {"prompt_tokens": 200000, "vision": True}
    assert pool["tiers"]["standard"][-1] == "gpt-7-nova"
    before_pool = configmap_data(staged.pinball, "pinball.yaml")
    assert pool["tiers"]["standard"][:-1] == before_pool["tiers"]["standard"]
    assert candidate_lists(after.candidates).candidates == ()
    gw = configmap_data(after.gateway, "config.yaml")
    targets = {v["name"]: v for v in gw["llm"]["virtualModels"]}["pinball"]["routing"]
    assert {
        "when": 'default(extauthz.model, "") == "gpt-7-nova"',
        "model": "pinball-gpt-7-nova",
    } in targets["conditional"]["targets"]
    removed, added = changed(staged.gateway, after.gateway)
    assert (removed, len(added)) == ([], 3)  # pinball-<id>, auto target, rates
    removed, added = changed(staged.pinball, after.pinball)
    assert removed == ["      standard: [claude-sonnet-5.5, claude-sonnet-5, gemini-3.8-flash]\n"]
    assert added == [
        "      gpt-7-nova: {prompt_tokens: 200000, vision: true}\n",
        "      standard: [claude-sonnet-5.5, claude-sonnet-5, gemini-3.8-flash, gpt-7-nova]\n",
    ]


def test_step_b_preserves_every_comment_of_pinball_yaml():
    staged = candidate_change(NEW, TODAY).edit(carrier())
    after = promote_change(NEW, "heavy", ("1.00", "4.00"), {}).edit(staged)
    comments = [x for x in staged.pinball.splitlines() if x.lstrip().startswith("#")]
    assert comments == [x for x in after.pinball.splitlines() if x.lstrip().startswith("#")]


def test_drop_removes_the_alias_and_remembers_the_model():
    staged = candidate_change(NEW, TODAY).edit(carrier())
    after = drop_change(NEW.id, 7, TODAY).edit(staged)
    assert after.gateway == carrier().gateway
    lists = candidate_lists(after.candidates)
    assert lists.candidates == () and lists.dropped == {"gpt-7-nova"}
    check(after)


def test_invariants_reject_a_pool_model_without_an_auto_target():
    staged = candidate_change(NEW, TODAY).edit(carrier())
    broken = replace(staged, pinball=edit.add_pool_model(staged.pinball, NEW.id, 1000, False))
    with pytest.raises(InvariantError):
        check(broken)


def test_invariants_reject_an_alias_that_is_no_candidate():
    f = carrier()
    with pytest.raises(InvariantError, match="opt-"):
        check(replace(f, gateway=edit.add_opt_alias(f.gateway, NEW.id)))


def test_unsafe_model_ids_never_reach_yaml():
    for raw in ("x: y", "a}b", "", "UPPER", "a\nb", "-lead"):
        with pytest.raises(EditError):
            model_id(raw)


def test_edits_refuse_a_missing_anchor():
    with pytest.raises(EditError):
        edit.add_opt_alias("kind: ConfigMap\n", NEW.id)
    with pytest.raises(EditError):
        edit.append_to_tier(carrier().pinball, "ultra", NEW.id)
    with pytest.raises(EditError):
        edit.remove_list_entry(carrier().candidates, "candidates", NEW.id)


def test_append_to_tier_refuses_a_duplicate():
    with pytest.raises(EditError):
        edit.append_to_tier(carrier().pinball, "light", model_id("claude-haiku-5.5"))


def test_candidate_list_round_trips_to_empty_flow_list():
    text = carrier().candidates
    once = edit.add_list_entry(text, "candidates", '{id: a1, added: "2026-11-01"}')
    twice = edit.add_list_entry(once, "candidates", '{id: b2, added: "2026-11-02"}')
    assert candidate_lists(twice).candidates == ("a1", "b2")
    back = edit.remove_list_entry(
        edit.remove_list_entry(twice, "candidates", model_id("a1")), "candidates", model_id("b2")
    )
    assert back == text
