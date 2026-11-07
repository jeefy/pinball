from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pinball.optimize.db import SqliteDb, Window, open_db

SCHEMA = """
CREATE TABLE request_logs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, http_status INTEGER,
  error TEXT, gen_ai_response_model TEXT, input_tokens INTEGER, output_tokens INTEGER,
  cost REAL, agentgateway_user TEXT, attributes_json TEXT NOT NULL);
CREATE TABLE request_log_payloads (log_id TEXT PRIMARY KEY REFERENCES request_logs(id),
  request_prompt_json TEXT, response_completion_json TEXT);
"""
NOW = datetime(2026, 11, 7, 12, tzinfo=UTC)


def add(
    c,
    id,
    *,
    age_h=1,
    status=200,
    user="alice",
    cost: float | None = 0.5,
    tier="light",
    orig="auto",
    req=None,
    resp=None,
):
    attrs = {
        "agw.ai.original_model": orig,
        "pinball.tier": tier,
        "pinball.reason": "short",
        "pinball.model": "m",
    }
    c.execute(
        "INSERT INTO request_logs VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            id,
            (NOW - timedelta(hours=age_h)).isoformat(),
            status,
            None,
            "served",
            3,
            4,
            cost,
            user,
            json.dumps(attrs),
        ),
    )
    c.execute(
        "INSERT INTO request_log_payloads VALUES (?,?,?)",
        (id, json.dumps(req or [{"role": "user", "content": "hi"}]), json.dumps(resp or ["hello"])),
    )


@pytest.fixture
def db(tmp_path: Path) -> SqliteDb:
    path = tmp_path / "data.db"
    c = sqlite3.connect(path)
    c.executescript(SCHEMA)
    add(c, "a")
    add(c, "b", tier="heavy", user="x" * 80)
    add(c, "err", status=500, cost=None)
    add(c, "eval", user="eval-1")
    add(c, "pb", user="pinball-2")
    add(c, "other", orig="gpt")
    add(c, "empty", resp=[""])
    add(c, "notuser", req=[{"role": "assistant", "content": "x"}])
    add(c, "badtier", tier="nope")
    add(c, "old", age_h=24 * 30)
    c.commit()
    c.close()
    return SqliteDb(f"sqlite:///{path}")  # absolute path => sqlite:////...


def test_pool_filters_like_postgres(db: SqliteDb):
    ids = [r["id"] for r in db.pool(3650)]
    assert ids == ["a", "b", "old"]
    assert "old" not in [r["id"] for r in db.pool(7)]
    caller = {r["id"]: r["caller"] for r in db.pool(7)}["b"]
    assert caller == "x" * 64


def test_payloads_decode_json(db: SqliteDb):
    (row,) = db.payloads(["a", "missing"])
    assert row["request"] == [{"role": "user", "content": "hi"}]
    assert row["response"] == ["hello"]
    assert (row["tier"], row["reason"], row["pinball_model"], row["served_model"]) == (
        "light",
        "short",
        "m",
        "served",
    )
    assert db.payloads([]) == []


def test_window_counts_errors_and_cost(db: SqliteDb):
    start, end = NOW - timedelta(days=2), NOW
    w = db.window(start, end)
    # a, b, err, empty, notuser, badtier: auto traffic; eval/pb/other excluded; old out of range
    assert w == Window(requests=6, errors=1, cost=2.5, priced=5)
    assert db.window(start, end, caller="alice").requests == 5
    assert db.window(start, end, caller="nobody") == Window(0, 0, 0.0, 0)


def test_read_only(db: SqliteDb):
    with pytest.raises(sqlite3.OperationalError):
        db._conn.execute("DELETE FROM request_logs")


def test_open_db_picks_backend(tmp_path: Path):
    p = tmp_path / "x.db"
    sqlite3.connect(p).executescript(SCHEMA)
    assert isinstance(open_db(f"sqlite:///{p}"), SqliteDb)
