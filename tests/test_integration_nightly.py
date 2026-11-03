"""One `run` night with ranking, per-caller learning and new-model discovery together."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

from fake_gitea import FakeGitea, Repo, mounted_key
from test_callers import ENV, CallerGateway, bot_db
from test_discovery import CATALOG, TOKEN, repo_files, staged_carrier
from test_optimize import NOW, SECRET, env_for

from pinball.config import load_config
from pinball.optimize import cli
from pinball.optimize.invariants import PATHS
from pinball.overlay import load_overlay, merge

BUDGET = 5.0


def night(tmp_path: Path, repo: Repo, gw: CallerGateway) -> tuple[int, str, list[dict[str, Any]]]:
    cands = tmp_path / "candidates.yaml"
    cands.write_text(mounted_key(repo.files.get(PATHS["candidates"], "")))
    audit = tmp_path / "audit.jsonl"
    out = io.StringIO()
    with FakeGitea(repo) as fake:
        env = env_for(
            tmp_path,
            **ENV,
            PINBALL_OPT_BUDGET_USD=str(BUDGET),
            PINBALL_GITEA_TOKEN=TOKEN,
            PINBALL_GITEA_URL=fake.url,
            PINBALL_OPT_CANDIDATES=str(cands),
            PINBALL_OPT_AUDIT_PATH=str(audit),
        )
        code = cli.main(
            ["run", "--per-tier", "10", "--seed", "1"],
            env,
            lambda _: bot_db(30),
            lambda _: gw,
            NOW,
            out,
            catalog=lambda: CATALOG,
        )
    return code, out.getvalue(), [json.loads(x) for x in audit.read_text().splitlines()]


def test_ranking_callers_and_discovery_share_one_night(tmp_path: Path, caplog):
    caplog.set_level(logging.DEBUG)
    gw = CallerGateway({"s2", "gpt-7-nova"}, "BOTQ")
    code, out, audit = night(tmp_path, Repo(repo_files(staged_carrier())), gw)

    assert code == 0
    (run,) = [e for e in audit if e.get("kind") == "run"]
    (found,) = [e for e in audit if e.get("kind") == "new_models"]
    assert run["result"] == "written"
    assert {"errors", "tier_budget", "budget_refused"} <= run.keys()  # coverage
    assert run["callers"]["bot"]["result"]  # per-caller
    assert found["evaluated"] and found["prs"]  # discovery
    assert run["spent_usd"] <= BUDGET
    assert sum(t["spent_usd"] for t in run["tier_budget"].values()) <= BUDGET + 1e-9

    overlay = load_overlay(tmp_path / "overlay.yaml")
    assert overlay is not None and "bot" in overlay.callers
    merge(load_config(tmp_path / "pinball.yaml"), overlay)
    # Candidate-evaluation replays never become caller or ranking evidence.
    assert "gpt-7-nova" not in json.dumps(run["callers"]) + json.dumps(run["tiers"])

    leaked = out + (tmp_path / "audit.jsonl").read_text() + caplog.text
    assert TOKEN not in leaked and SECRET not in leaked and "BOTQ" not in leaked

    (tmp_path / "audit.jsonl").unlink()
    broken, out, audit = night(tmp_path, Repo({}), CallerGateway({"s2"}, "BOTQ"))
    assert broken == 0
    assert [(e["kind"], e["result"]) for e in audit] == [
        ("run", "unchanged"),  # tonight's overlay already holds what was learned
        ("new_models", "failed GiteaError"),
    ]
