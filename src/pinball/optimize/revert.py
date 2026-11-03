"""Auto-revert: compare gateway health since the last overlay change with the window before.

Source: the gateway request_logs table (same read-only DSN as the nightly run),
not Prometheus: one dependency, and the table already has http_status and cost
per request. Windows cover `auto` requests from non eval-*/pinball-* callers:

  after  = [changed_at, min(now, changed_at + hours))
  before = [changed_at - hours, changed_at)

Revert when, with at least min_requests in both windows:
  error rate:  after > before * (1 + error_margin)  AND  after - before > error_floor
  cost/req:    after > before * (1 + cost_margin)   (only when both windows are priced)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from pinball.optimize.db import ReplayDb, Window
from pinball.optimize.settings import RevertSettings


@dataclass(frozen=True, slots=True)
class RevertDecision:
    revert: bool
    reasons: tuple[str, ...]
    before: Window
    after: Window

    def record(self) -> dict[str, object]:
        def w(x: Window) -> dict[str, object]:
            return {
                "requests": x.requests,
                "errors": x.errors,
                "error_rate": round(x.error_rate, 6),
                "cost_per_request": round(x.cost_per_request, 6),
            }

        return {
            "revert": self.revert,
            "reasons": list(self.reasons),
            "before": w(self.before),
            "after": w(self.after),
        }


def windows(
    changed_at: datetime, now: datetime, hours: float
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    span = timedelta(hours=hours)
    return (changed_at - span, changed_at), (changed_at, min(now, changed_at + span))


def judge_windows(before: Window, after: Window, s: RevertSettings) -> RevertDecision:
    reasons: list[str] = []
    if before.requests < s.min_requests or after.requests < s.min_requests:
        return RevertDecision(False, ("insufficient_traffic",), before, after)
    e0, e1 = before.error_rate, after.error_rate
    if e1 > e0 * (1 + s.error_margin) and e1 - e0 > s.error_floor:
        reasons.append(f"error_rate {e0:.4f} -> {e1:.4f} (margin {s.error_margin:.0%})")
    if before.priced and after.priced:
        c0, c1 = before.cost_per_request, after.cost_per_request
        if c1 > c0 * (1 + s.cost_margin):
            reasons.append(f"cost_per_request {c0:.6f} -> {c1:.6f} (margin {s.cost_margin:.0%})")
    return RevertDecision(bool(reasons), tuple(reasons), before, after)


def check(db: ReplayDb, changed_at: datetime, now: datetime, s: RevertSettings) -> RevertDecision:
    (b0, b1), (a0, a1) = windows(changed_at, now, s.hours)
    return judge_windows(db.window(b0, b1), db.window(a0, a1), s)
