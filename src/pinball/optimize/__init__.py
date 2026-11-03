"""Nightly optimizer: re-rank models within tiers from judged replays of real traffic.

Pipeline (one `pinball-optimize run`): draw a seeded sample of yesterday's
`auto` requests (sample), replay each on the other models of its tier and
have two judges compare the answers (replay, judge) under a hard dollar cap
(budget), turn agreed verdicts into a cost-adjusted, shrunk score per model
(rank), and write the reordered tiers as the overlay pinball hot-reloads
(apply). `--check-revert` restores the previous overlay when error rate or
cost per request regressed (revert).

Prompt and response text stays in memory; nothing persisted carries it.
"""
