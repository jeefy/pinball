from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

from pinball.config import load_config
from pinball.optimize.apply import State, save_state
from pinball.optimize.db import FakeDb, PostgresDb, Window
from pinball.optimize.deps import Deps
from pinball.optimize.revert_cli import EXIT_REJECTED, check_revert
from pinball.optimize.settings import Settings
from pinball.reload import Reloader, ReloadMetrics


def test_postgres_db_query_and_methods():
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"id": "req-1", "tier": "light", "requests": 10, "errors": 1, "cost": 0.05, "priced": 10}
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("psycopg.connect", return_value=mock_conn):
        db = PostgresDb("postgresql://fake:5432/db")
        assert db._conn.read_only is True

        p = db.pool(1)
        assert len(p) == 1
        assert p[0]["id"] == "req-1"

        payloads = db.payloads(["req-1"])
        assert len(payloads) == 1

        w1 = db.window(datetime(2026, 1, 1), datetime(2026, 1, 2))
        assert w1.requests == 10
        assert w1.errors == 1

        w2 = db.window(datetime(2026, 1, 1), datetime(2026, 1, 2), caller="caller-a")
        assert w2.requests == 10

        db.close()
        mock_conn.close.assert_called_once()


def test_fake_db_window_fallback():
    fakedb = FakeDb(windows=[(datetime(2026, 1, 1), datetime(2026, 1, 2), Window(5, 0, 0.01, 5))])
    fallback = fakedb.window(datetime(2026, 2, 1), datetime(2026, 2, 2))
    assert fallback == Window(0, 0, 0.0, 0)


def test_reloader_io_error_handling(tmp_path: Path):
    r = CollectorRegistry()
    metrics = ReloadMetrics.create(r)
    base = load_config()
    target = tmp_path / "overlay.yaml"
    target.write_text("version: '2026-11-02.1'\ngenerated_at: '2026-11-02T00:00:00Z'\n")

    reloader = Reloader(base, target, metrics)
    assert reloader.active.version == "2026-11-02.1"

    target.write_text("version: '2026-11-02.2'\ngenerated_at: '2026-11-02T01:00:00Z'\n")
    with patch("pinball.reload.load_overlay", side_effect=OSError("permission denied")):
        reloader.poll()
        assert reloader.active.version == "2026-11-02.1"
        assert reloader._stamp is None


def test_revert_cli_callers_rejected_and_previous_rejected(tmp_path: Path):
    cfg_file = tmp_path / "pinball.yaml"
    cfg_file.write_text(Path("pinball.yaml").read_text())

    overlay_file = tmp_path / "overlay.yaml"
    overlay_content = "version: '2026-11-02.1'\ngenerated_at: '2026-11-02T00:00:00Z'\n"
    overlay_file.write_text(overlay_content)

    state_file = tmp_path / "state.json"
    audit_file = tmp_path / "audit.jsonl"

    state = State(
        seq=1,
        current={"version": "2026-11-02.1", "generated_at": "2026-11-02T00:00:00Z"},
        previous={"version": "2026-11-01.1", "generated_at": "2026-11-01T00:00:00Z"},
        changed_at="2026-11-02T00:00:00Z",
    )
    save_state(state_file, state)

    now = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
    settings = Settings(
        config_path=cfg_file,
        overlay_path=overlay_file,
        state_path=state_file,
        audit_path=audit_file,
    )

    out = io.StringIO()
    db = FakeDb()

    with patch(
        "pinball.optimize.revert_cli.revert_callers",
        side_effect=pytest.importorskip("pinball.overlay").OverlayRejected("bad_caller", "detail"),
    ):
        deps = Deps(
            db=db,
            gateway=None,
            now=now,
            out=out,
        )
        rc = check_revert(settings, deps)
        assert rc == 0
        assert "callers_rejected bad_caller" in out.getvalue()

    out2 = io.StringIO()
    with (
        patch("pinball.optimize.revert_cli.check") as mock_check,
        patch(
            "pinball.optimize.revert_cli.write_overlay",
            side_effect=pytest.importorskip("pinball.overlay").OverlayRejected(
                "invalid_prev", "detail"
            ),
        ),
    ):
        mock_decision = MagicMock()
        mock_decision.revert = True
        mock_decision.record.return_value = {"revert": True}
        mock_decision.reasons = ["regression"]
        mock_check.return_value = mock_decision

        deps2 = Deps(
            db=db,
            gateway=None,
            now=now,
            out=out2,
        )
        rc2 = check_revert(settings, deps2)
        assert rc2 == EXIT_REJECTED
        assert "previous_rejected invalid_prev" in out2.getvalue()
