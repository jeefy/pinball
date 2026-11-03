"""Content-free outcome counts of one replay run, for the audit entry.

budget_refused is NOT an error: the pair was never sent because the cap (or its
tier's share) had no room. Real errors are gateway_errors (per candidate model,
by label: `HTTP <status>`, `transport <Exception>`, `timeout`, `empty ...`) and,
per judge, judge_unparseable / judge_failed (a gateway error on the last attempt).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pinball.optimize.replay import PairResult

BUDGET = "budget"
UNPARSEABLE = "unparseable"
NOT_ERRORS = frozenset({BUDGET, "cancelled"})  # cancelled: the pair already timed out


def outcome_counts(results: list[PairResult]) -> dict[str, Any]:
    refused = sum(r.error == BUDGET for r in results)
    gateway: dict[str, Counter[str]] = {}
    unparseable: Counter[str] = Counter()
    failed: dict[str, Counter[str]] = {}
    for r in results:
        if r.error is not None and r.error != BUDGET:
            gateway.setdefault(r.candidate, Counter())[r.error] += 1
        for judge, label in r.judge_errors.items():
            if label == UNPARSEABLE:
                unparseable[judge] += 1
            elif label not in NOT_ERRORS:
                failed.setdefault(judge, Counter())[label] += 1
    errors = (
        sum(sum(c.values()) for c in gateway.values())
        + sum(unparseable.values())
        + sum(sum(c.values()) for c in failed.values())
    )
    return {
        "pairs": len(results),
        "pairs_judged": sum(r.error is None for r in results),
        "budget_refused": refused,
        "judge_budget_refused": sum(
            label == BUDGET for r in results for label in r.judge_errors.values()
        ),
        "errors": errors,
        "gateway_errors": {m: dict(sorted(c.items())) for m, c in sorted(gateway.items())},
        "judge_unparseable": dict(sorted(unparseable.items())),
        "judge_failed": {j: dict(sorted(c.items())) for j, c in sorted(failed.items())},
    }
