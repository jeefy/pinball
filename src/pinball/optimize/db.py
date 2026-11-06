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
        """[{id, tier, caller}] of replayable successful `auto` requests."""
        ...

    def payloads(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        """Full rows (incl. request/response JSON) for `ids`."""
        ...

    def window(self, start: datetime, end: datetime, caller: str | None = None) -> Window:
        """Health of `auto` traffic; one caller only when `caller` is given."""
        ...


# The gateway stores the x-agw-caller header in request_logs.agentgateway_user (the
# same column the eval-*/pinball-* exclusion reads). It is cut to 64 chars here, as
# pinball does before profile lookup, so longer caller-supplied strings never leave SQL.
CALLER_COL: LiteralString = "left(coalesce(r.agentgateway_user, ''), 64)"

_EXCLUDE: LiteralString = """
  AND coalesce(r.agentgateway_user, '') NOT LIKE 'eval-%%'
  AND coalesce(r.agentgateway_user, '') NOT LIKE 'pinball-%%'
"""

POOL_SQL: LiteralString = (
    """
SELECT r.id::text AS id, r.attributes_json->>'pinball.tier' AS tier,
  """
    + CALLER_COL
    + """ AS caller
FROM request_logs r JOIN request_log_payloads p ON p.log_id = r.id
WHERE r.attributes_json->>'agw.ai.original_model' IN ('auto', 'pinball')
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

PAYLOADS_SQL: LiteralString = (
    """
SELECT r.id::text AS id, r.attributes_json->>'pinball.tier' AS tier,
  """
    + CALLER_COL
    + """ AS caller,
  r.attributes_json->>'pinball.reason' AS reason,
  r.attributes_json->>'pinball.model' AS pinball_model,
  r.gen_ai_response_model AS served_model, r.input_tokens, r.output_tokens,
  p.request_prompt_json AS request, p.response_completion_json AS response
FROM request_logs r JOIN request_log_payloads p ON p.log_id = r.id
WHERE r.id::text = ANY(%(ids)s)
"""
)

WINDOW_SQL: LiteralString = (
    """
SELECT count(*) AS requests,
  count(*) FILTER (WHERE r.http_status NOT BETWEEN 200 AND 299) AS errors,
  coalesce(sum(r.cost), 0)::float8 AS cost,
  count(r.cost) AS priced
FROM request_logs r
WHERE r.attributes_json->>'agw.ai.original_model' IN ('auto', 'pinball')
  AND r.started_at >= %(start)s AND r.started_at < %(end)s
"""
    + _EXCLUDE
)

WINDOW_CALLER_SQL: LiteralString = WINDOW_SQL + "  AND " + CALLER_COL + " = %(caller)s\n"


class PostgresDb:
    """psycopg connection in read-only mode; DSN from PINBALL_OPT_DB_DSN."""

    def __init__(self, dsn: str, retries: int = 3, retry_delay: float = 0.5) -> None:
        import time

        import psycopg  # optional extra: only the nightly job needs it

        conn = None
        for attempt in range(retries):
            try:
                conn = psycopg.connect(dsn, autocommit=False)
                break
            except psycopg.OperationalError:
                if attempt == retries - 1:
                    raise
                time.sleep(retry_delay * (2**attempt))
        if conn is None:
            conn = psycopg.connect(dsn, autocommit=False)
        self._conn = conn
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

    def window(self, start: datetime, end: datetime, caller: str | None = None) -> Window:
        if caller is None:
            row = self._query(WINDOW_SQL, {"start": start, "end": end})[0]
        else:
            params = {"start": start, "end": end, "caller": caller}
            row = self._query(WINDOW_CALLER_SQL, params)[0]
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
    caller_windows: dict[tuple[str, datetime, datetime], Window] = field(default_factory=dict)
    payload_calls: int = 0

    def pool(self, days: int) -> list[dict[str, Any]]:
        return [
            {"id": r["id"], "tier": r["tier"], "caller": str(r.get("caller") or "")[:64]}
            for r in sorted(self.rows, key=lambda r: r["id"])
        ]

    def payloads(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        self.payload_calls += 1
        wanted = set(ids)
        return [dict(r) for r in self.rows if r["id"] in wanted]

    def window(self, start: datetime, end: datetime, caller: str | None = None) -> Window:
        if caller is not None:
            return self.caller_windows.get((caller, start, end), Window(0, 0, 0.0, 0))
        for s, e, w in self.windows:
            if s == start and e == end:
                return w
        return Window(0, 0, 0.0, 0)
