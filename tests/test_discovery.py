from __future__ import annotations

import io
import json
import logging
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fake_gitea import FakeGitea, Repo
from test_carrier_edit import ROOT, carrier
from test_optimize import NOW, FakeGateway, env_for, traffic

from pinball.optimize import cli
from pinball.optimize.bot import Bot, BotSettings, candidate_change, promote_change
from pinball.optimize.carrier_edit import model_id
from pinball.optimize.discover import (
    CatalogError,
    CatalogModel,
    fetch_catalog,
    new_models,
    parse_catalog,
)
from pinball.optimize.discovery_cli import discovery_env
from pinball.optimize.gitea import Gitea, GiteaError
from pinball.optimize.invariants import PATHS, CarrierFiles, candidate_lists, check, pool_ids

TOKEN = "tok-SECRET-0123456789"
TODAY = date(2026, 11, 3)


def entry(mid: str, kind: str = "chat", **caps: Any) -> dict[str, Any]:
    return {"id": mid, "capabilities": {"type": kind, **caps}}


CATALOG = {
    "data": [
        entry("claude-sonnet-5.5"),  # pool
        entry("gpt-7-nova", limits={"max_prompt_tokens": 400000}, supports={"vision": True}),
        entry("grok-5", limits={"max_prompt_tokens": 128000}, supports={"vision": False}),
        {"id": "mai-2"},  # no capabilities: chat by default, conservative limits
        entry("text-embedding-3-small", "embeddings"),
        {"id": "text-embedding-ada-002"},  # no capabilities, embeddings by name
        entry("gpt-4o-2024-11-20"),  # dated snapshot
        {"id": "Bad Id"},
    ]
}


def repo_files(f: CarrierFiles) -> dict[str, str]:
    return {PATHS[k]: getattr(f, k) for k in PATHS}


# --- catalog ---------------------------------------------------------------------------


def test_catalog_keeps_chat_models_and_caps_prompt_tokens():
    cat = parse_catalog(CATALOG)
    assert sorted(cat) == ["claude-sonnet-5.5", "gpt-7-nova", "grok-5", "mai-2"]
    assert cat[model_id("gpt-7-nova")] == CatalogModel(model_id("gpt-7-nova"), 250000, True, ())
    assert cat[model_id("mai-2")].defaulted == ("prompt_tokens", "vision")
    assert cat[model_id("mai-2")].vision is False


def test_catalog_rejects_a_non_list_answer():
    with pytest.raises(CatalogError):
        parse_catalog({"object": "list"})


def test_new_models_diffs_against_pool_candidates_and_dropped():
    cat = parse_catalog(CATALOG)
    known = [model_id("claude-sonnet-5.5"), model_id("grok-5"), model_id("mai-2")]
    assert [m.id for m in new_models(cat, known)] == ["gpt-7-nova"]


def test_fetch_catalog_reports_status_only():
    def boom(*_: object, **__: object) -> Any:
        raise OSError("connection refused to secret-host")

    with pytest.raises(CatalogError, match="transport OSError"):
        fetch_catalog("http://x", opener=boom)


# --- bot against the fake Gitea --------------------------------------------------------


def bot_for(fake: FakeGitea, **kw: Any) -> Bot:
    s = BotSettings(**{"checks_wait_s": 0.0, **kw})
    return Bot(Gitea(fake.url, "jeefy/carrier", TOKEN, read_only=s.dry_run), s, lambda _: None)


def gpt7() -> CatalogModel:
    return parse_catalog(CATALOG)[model_id("gpt-7-nova")]


def test_step_a_pr_is_opened_merged_on_green_checks_and_main_stays_valid():
    repo = Repo(repo_files(carrier()))
    with FakeGitea(repo) as fake:
        out = bot_for(fake).submit(candidate_change(gpt7(), TODAY))
    assert (out.result, out.pr) == ("merged", 1)
    main = CarrierFiles(**{k: repo.files[p] for k, p in PATHS.items()})
    check(main)
    assert candidate_lists(main.candidates).candidates == ("gpt-7-nova",)
    assert {a for _, _, a in repo.requests} == {f"token {TOKEN}"}


def test_existing_branch_or_pr_is_not_duplicated():
    repo = Repo(repo_files(carrier()))
    with FakeGitea(repo) as fake:
        bot_for(fake).submit(candidate_change(gpt7(), TODAY))
        writes = len(repo.writes)
        again = bot_for(fake).submit(candidate_change(gpt7(), TODAY))
    assert again.result == "exists"
    assert len(repo.writes) == writes


def test_pr_cap_per_night():
    repo = Repo(repo_files(carrier()))
    models = [CatalogModel(model_id(f"m-{i}"), 1000, False, ()) for i in range(5)]
    with FakeGitea(repo) as fake:
        bot = bot_for(fake, max_prs=3)
        results = [bot.submit(candidate_change(m, TODAY)).result for m in models]
    assert results == ["merged"] * 3 + ["capped"] * 2
    assert len(repo.pulls) == 3 and repo.merged == [1, 2, 3]
    check(CarrierFiles(**{k: repo.files[p] for k, p in PATHS.items()}))


def test_an_unmerged_pr_blocks_the_next_one():
    # Every PR is cut from main; a second PR on the same base would conflict.
    repo = Repo(repo_files(carrier()), status="pending")
    models = [CatalogModel(model_id(f"m-{i}"), 1000, False, ()) for i in range(2)]
    with FakeGitea(repo) as fake:
        bot = bot_for(fake)
        results = [bot.submit(candidate_change(m, TODAY)).result for m in models]
    assert results == ["checks_pending", "waiting_for_pr 1"]
    assert len(repo.pulls) == 1


def test_failed_checks_leave_the_pr_open():
    repo = Repo(repo_files(carrier()), status="failure")
    with FakeGitea(repo) as fake:
        out = bot_for(fake).submit(candidate_change(gpt7(), TODAY))
    assert out.result == "checks_failure" and repo.merged == []


def test_no_pr_checks_means_no_merge_unless_configured():
    repo = Repo(repo_files(carrier()), status="", statuses=0)
    with FakeGitea(repo) as fake:
        held = bot_for(fake).submit(candidate_change(gpt7(), TODAY))
        assert held.result == "checks_pending" and repo.merged == []
        forced = bot_for(fake, merge_without_checks=True).submit(candidate_change(gpt7(), TODAY))
    assert forced.result == "merged_without_checks" and repo.merged == [1]


def test_a_pending_pr_is_merged_on_a_later_night():
    repo = Repo(repo_files(carrier()), status="pending")
    with FakeGitea(repo) as fake:
        bot_for(fake).submit(candidate_change(gpt7(), TODAY))
        repo.status = "success"
        later = bot_for(fake).submit(candidate_change(gpt7(), TODAY))
    assert (later.result, repo.merged) == ("merged", [1])


def test_auto_merge_refused_when_invariants_fail():
    # A promotion for a model that never became a candidate has no opt- twin.
    repo = Repo(repo_files(carrier()))
    with FakeGitea(repo) as fake:
        out = bot_for(fake).submit(promote_change(gpt7(), "standard", ("1.00", "4.00"), {}))
    assert out.result.startswith("failed EditError")
    assert repo.writes == []

    def break_twins(f: CarrierFiles) -> CarrierFiles:
        staged = candidate_change(gpt7(), TODAY).edit(f)
        return replace(staged, candidates=f.candidates)  # alias without candidate entry

    bad = replace(candidate_change(gpt7(), TODAY), edit=break_twins)
    with FakeGitea(repo) as fake:
        out = bot_for(fake).submit(bad)
    assert out.result.startswith("failed InvariantError") and repo.writes == []


def test_dry_run_client_refuses_writes_before_the_socket():
    repo = Repo(repo_files(carrier()))
    with FakeGitea(repo) as fake:
        g = Gitea(fake.url, "jeefy/carrier", TOKEN, read_only=True)
        with pytest.raises(GiteaError, match="read-only"):
            g.open_pull("b", "t", "x")
    assert repo.writes == []
    assert TOKEN not in repr(g)


# --- nightly wiring --------------------------------------------------------------------


def staged_carrier() -> CarrierFiles:
    return candidate_change(gpt7(), date(2026, 11, 1)).edit(carrier())


def nightly(tmp_path: Path, repo: Repo, argv: list[str], gw: FakeGateway | None, **env: str):
    cands = tmp_path / "candidates.yaml"
    cands.write_text(repo.files.get(PATHS["candidates"], ""))
    out = io.StringIO()
    with FakeGitea(repo) as fake:
        e = env_for(
            tmp_path,
            PINBALL_GITEA_TOKEN=TOKEN,
            PINBALL_GITEA_URL=fake.url,
            PINBALL_OPT_CANDIDATES=str(cands),
            PINBALL_OPT_AUDIT_PATH=str(tmp_path / "audit.jsonl"),
            **env,
        )
        code = cli.main(
            argv,
            e,
            lambda _: traffic(),
            lambda _: gw if gw is not None else pytest.fail("no gateway in dry-run"),
            NOW,
            out,
            catalog=lambda: CATALOG,
        )
    return code, out.getvalue()


def test_nightly_promotes_a_winning_candidate_and_proposes_new_ones(tmp_path: Path, caplog):
    caplog.set_level(logging.DEBUG)
    repo = Repo(repo_files(staged_carrier()))
    code, out = nightly(tmp_path, repo, ["run", "--per-tier", "10"], FakeGateway({"gpt-7-nova"}))
    assert code == 0
    report = json.loads(out[out.index('{\n  "kind": "new_models"') :])
    (verdict,) = report["evaluated"]
    assert verdict["tier"] == "standard"
    assert [(p["step"], p["model"], p["result"]) for p in report["prs"]] == [
        ("promote", "gpt-7-nova", "merged"),
        ("candidate", "grok-5", "merged"),
        ("candidate", "mai-2", "merged"),
    ]
    main = CarrierFiles(**{k: repo.files[p] for k, p in PATHS.items()})
    check(main)
    assert "gpt-7-nova" in pool_ids(main.pinball)
    assert candidate_lists(main.candidates).candidates == ("grok-5", "mai-2")
    leaked = out + (tmp_path / "audit.jsonl").read_text() + caplog.text
    assert TOKEN not in leaked
    assert all(TOKEN not in p["body"] for p in repo.pulls if "body" in p)


def test_nightly_drops_a_losing_candidate_after_n_nights(tmp_path: Path):
    repo = Repo(repo_files(staged_carrier()))
    code, out = nightly(
        tmp_path,
        repo,
        ["run", "--per-tier", "10"],
        FakeGateway(),
        PINBALL_OPT_CANDIDATE_NIGHTS="2",
        PINBALL_OPT_MAX_PRS="1",
    )
    report = json.loads(out[out.index('{\n  "kind": "new_models"') :])
    assert report["evaluated"][0]["tier"] is None
    assert [(p["step"], p["result"]) for p in report["prs"]] == [
        ("drop", "merged"),
        ("candidate", "capped"),
        ("candidate", "capped"),
    ]
    assert candidate_lists(repo.files[PATHS["candidates"]]).dropped == {"gpt-7-nova"}
    assert code == 0


def test_dry_run_plans_prs_with_zero_http_writes(tmp_path: Path):
    repo = Repo(repo_files(staged_carrier()))
    code, out = nightly(tmp_path, repo, ["run", "--dry-run", "--per-tier", "10"], None)
    report = json.loads(out[out.index('{\n  "kind": "new_models"') :])
    assert code == 0 and report["result"] == "planned"
    assert {p["result"].split()[0] for p in report["prs"]} == {"planned"}
    assert repo.writes == [] and repo.pulls == []


def test_discovery_failure_never_fails_ranking(tmp_path: Path):
    repo = Repo({})  # every file read 404s
    code, out = nightly(tmp_path, repo, ["run", "--per-tier", "10"], FakeGateway({"s3"}))
    assert code == 0
    assert '"result": "written"' in out
    assert '"result": "failed GiteaError"' in out
    strict, _ = nightly(
        tmp_path, repo, ["run", "--per-tier", "10"], FakeGateway(), PINBALL_OPT_DISCOVERY_STRICT="1"
    )
    assert strict == 1


def test_discover_subcommand_needs_a_token_and_reports(tmp_path: Path):
    assert cli.main(["discover"], env_for(tmp_path), out=io.StringIO()) == 2
    repo = Repo(repo_files(carrier()))
    code, out = nightly(tmp_path, repo, ["discover", "--dry-run"], None)
    assert code == 0
    assert [p["model"] for p in json.loads(out)["prs"]] == ["gpt-7-nova", "grok-5", "mai-2"]


def test_discovery_env_defaults():
    d = discovery_env({}, dry_run=False)
    assert (d.enabled, d.gitea_url, d.repo, d.bot.max_prs, d.bot.candidate_nights) == (
        False,
        "https://gitea.grizzlygeek.net",
        "jeefy/carrier",
        3,
        7,
    )


def test_fixture_root_is_used():
    assert (ROOT / PATHS["pinball"]).is_file()
