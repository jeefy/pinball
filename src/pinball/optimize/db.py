"""Read-only access to the agentgateway request log.

`ReplayDb` is the seam: `PostgresDb` (psycopg, optional extra `optimize`) in
production, `FakeDb` in tests. Every query runs in a READ ONLY transaction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, LiteralString, Protocol

TIERS = ("light", "standard", "heavy")


@dataclass(frozen=True, slots=True)
class Window:
    """Aggregate gateway health for one time window (all callers except eval-*/pinball-*)."""

    requests: int
    errors: int  # non-2xx
    cost: float  # sum of request_logs.cost over priced rows
    priced: int  # rows with a cost

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @property
    def cost_per_request(self) -> float:
        return self.cost / self.priced if self.priced else 0.0


class ReplayDb(Protocol):
    def pool(self, days: int) -> list[dict[str, Any]]:
        """[{id, tier}] of replayable successful `auto` requests."""
        ...

    def payloads(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        """Full rows (incl. request/response JSON) for `ids`."""
        ...

    def window(self, start: datetime, end: datetime) -> Window: ...


_EXCLUDE: LiteralString = """
  AND coalesce(r.agentgateway_user, '') NOT LIKE 'eval-%%'
  AND coalesce(r.agentgateway_user, '') NOT LIKE 'pinball-%%'
"""

POOL_SQL: LiteralString = (
    """
SELECT r.id::text AS id, r.attributes_json->>'pinball.tier' AS tier
FROM request_logs r JOIN request_log_payloads p ON p.log_id = r.id
WHERE r.attributes_json->>'agw.ai.original_model' = 'auto'
  AND r.http_status BETWEEN 200 AND 299 AND r.error IS NULL
  AND r.started_at > now() - make_interval(days => %(days)s)
  AND r.attributes_json->>'pinball.tier' IN ('light', 'standard', 'heavy')
  AND jsonb_typeof(p.request_prompt_json) = 'array'
  AND p.request_prompt_json->-1->>'role' = 'user'
  AND length(btrim(coalesce(p.request_prompt_json->-1->>'content', ''))) > 0
  AND jsonb_typeof(p.response_completion_json) = 'array'
  AND length(btrim(coalesce(p.response_completion_json->>0, ''))) > 0
"""
    + _EXCLUDE
    + "ORDER BY r.id"
)

PAYLOADS_SQL: LiteralString = """
SELECT r.id::text AS id, r.attributes_json->>'pinball.tier' AS tier,
  r.attributes_json->>'pinball.reason' AS reason,
  r.attributes_json->>'pinball.model' AS pinball_model,
  r.gen_ai_response_model AS served_model, r.input_tokens, r.output_tokens,
  p.request_prompt_json AS request, p.response_completion_json AS response
FROM request_logs r JOIN request_log_payloads p ON p.log_id = r.id
WHERE r.id::text = ANY(%(ids)s)
"""

WINDOW_SQL: LiteralString = (
    """
SELECT count(*) AS requests,
  count(*) FILTER (WHERE r.http_status NOT BETWEEN 200 AND 299) AS errors,
  coalesce(sum(r.cost), 0)::float8 AS cost,
  count(r.cost) AS priced
FROM request_logs r
WHERE r.attributes_json->>'agw.ai.original_model' = 'auto'
  AND r.started_at >= %(start)s AND r.started_at < %(end)s
"""
    + _EXCLUDE
)


class PostgresDb:
    """psycopg connection in read-only mode; DSN from PINBALL_OPT_DB_DSN."""

    def __init__(self, dsn: str) -> None:
        import psycopg  # optional extra: only the nightly job needs it

        self._conn = psycopg.connect(dsn, autocommit=False)
        self._conn.read_only = True

    def _query(self, sql: LiteralString, params: dict[str, Any]) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        self._conn.rollback()
        return [dict(r) for r in rows]

    def pool(self, days: int) -> list[dict[str, Any]]:
        return self._query(POOL_SQL, {"days": int(days)})

    def payloads(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        return self._query(PAYLOADS_SQL, {"ids": list(ids)})

    def window(self, start: datetime, end: datetime) -> Window:
        row = self._query(WINDOW_SQL, {"start": start, "end": end})[0]
        return Window(
            int(row["requests"]), int(row["errors"]), float(row["cost"]), int(row["priced"])
        )

    def close(self) -> None:
        self._conn.close()


@dataclass
class FakeDb:
    """In-memory ReplayDb for tests: rows carry the PAYLOADS_SQL columns."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    windows: list[tuple[datetime, datetime, Window]] = field(default_factory=list)
    payload_calls: int = 0

    def pool(self, days: int) -> list[dict[str, Any]]:
        return [
            {"id": r["id"], "tier": r["tier"]} for r in sorted(self.rows, key=lambda r: r["id"])
        ]

    def payloads(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        self.payload_calls += 1
        wanted = set(ids)
        return [dict(r) for r in self.rows if r["id"] in wanted]

    def window(self, start: datetime, end: datetime) -> Window:
        for s, e, w in self.windows:
            if s == start and e == end:
                return w
        return Window(0, 0, 0.0, 0)
