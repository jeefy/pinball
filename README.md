# pinball

Automatic model selection for [agentgateway](https://agentgateway.dev). Clients
send `model: auto`; pinball reads the request and the gateway routes it to a
model that fits: a light model for titles and one-line summaries, a standard
model for ordinary chat and tool use, a heavy model for debugging, architecture
and high-effort reasoning, and a big-window model when the prompt will not fit
anywhere else.

pinball is an HTTP ext_authz service. It never sees the response, never
rewrites the request and never blocks it: every answer is `200`, and an empty
`model` sends the request to the gateway's fallback target.

## How it works

```
client ──model: auto──▶ agentgateway ──POST body──▶ pinball
                             │      ◀── {"model": "claude-opus-5.5", ...} ──┘
                             │  extauthz.model selects a conditional target
                             ▼
                        pinball-claude-opus-5.5 (internal model) ──▶ upstream
```

1. `llm.policies.extAuthz` runs only when `json(request.body).model == "auto"`
   and sends the buffered body (up to 16 MiB) to pinball.
2. pinball parses the body (OpenAI chat completions, Anthropic Messages or
   OpenAI Responses) and answers `{"model", "tier", "reason", "score",
   "est_tokens"}`. The `metadata` CEL map copies these into `extauthz.*`.
3. The `auto` virtual model's conditional targets match on `extauthz.model`
   and route to `visibility: internal` models. Clients cannot pick those
   directly, and request headers cannot steer routing.
4. If pinball is down or slower than the gateway's 2 s ext_authz timeout,
   `failureMode: allow` lets the request through to the last (fallback)
   target.

[`examples/agentgateway.yaml`](examples/agentgateway.yaml) is a complete,
validated gateway config.

## Choosing a tier

The score is the highest-scoring **user-authored** message in the
conversation plus request-level signals. Tool output, tool arguments,
assistant turns and system prompts count toward prompt size but are never
scored.

| Signal | Default points | `scoring` keys |
| --- | --- | --- |
| code fence, traceback, Go panic or stack frame | +1 | `code` |
| heavy keyword (`refactor`, `root cause`, `deadlock`, …), each distinct | +1 (max 2 counted) | `heavy_keyword`, `max_heavy_keywords` |
| message ≥ 4,000 chars / ≥ 12,000 chars | +1 / +2 | `long_chars`, `long`, `very_long_chars`, `very_long` |
| light keyword (`title`, `summarize`, `translate`, …) | −1 | `light_keyword` |
| message < 160 chars with no code or heavy keyword | −1 | `short_chars`, `short` |
| `tools` present | +1 | `tools` |
| reasoning effort high/xhigh/max, or thinking budget ≥ 16k | +2 | `effort_high` |
| reasoning effort none/minimal/low | −1 | `effort_low` |
| max output tokens ≤ 512 | −1 | `small_output_tokens`, `small_output` |
| JSON response format | −1 | `structured` |

`score < thresholds.standard` is light and `score >= thresholds.heavy` is
heavy. Everything in between is standard.

Clients that paste scaffolding into user messages (an agent loading a skill
file as a user turn, say) would otherwise score as if the user wrote it.
`wrapped_user_messages` lists such prefixes: the message still counts toward
prompt size, but only the user's instruction inside it is scored, and the
decision gets a `wrapped` signal.

Because the message part is a maximum over the whole history, a conversation
can move up a tier but never down. That avoids thrashing the prompt cache,
and it needs no session state, so every replica gives the same answer.

Within the tier, pinball picks the first model whose prompt window fits the
estimated prompt (chars / 3.5, plus 1,600 per image, times `headroom`) and
that supports vision if the request has images. If nothing in the tier fits,
it tries the tiers above. If nothing fits anywhere, it picks the largest
window. The `x-pinball-tier: light|standard|heavy` request header, when the
gateway forwards it, overrides the score. `x-pinball-min-tier: light|standard|heavy`
is a floor instead: the tier is raised to at least that value, and a higher score
still wins. The decision gets a `min_tier:<tier>` signal only when the floor raised
the tier. `x-pinball-tier` wins when both are present; invalid values are ignored.

All of this lives in [`pinball.yaml`](pinball.yaml): the model pool with
prompt windows, per-tier preference order, thresholds, signal points and
cutoffs (`scoring`, every key optional), wrapped user messages, headroom and
keyword lists. Mount your own copy and point `PINBALL_CONFIG` at it.

## Learned overlay

A tuning job can adjust routing without touching `pinball.yaml` (which Argo CD
self-heals). Point `PINBALL_OVERLAY_PATH` at a file (PVC, mounted file, or a
DB-synced file); pinball polls it every `PINBALL_OVERLAY_RELOAD_SECONDS`
(default 30) by inode/mtime/size and swaps the routing state atomically, with
no restart.

```yaml
version: "2026-11-02.1"            # changelog id, required
generated_at: "2026-11-02T03:00:00Z"
tier_order:                        # optional; a permutation of the base tier
  heavy: [claude-fable-5.1, claude-opus-5.5, claude-opus-5, gemini-4-argon, kimi-k3]
callers:                           # keyed by x-agw-caller (first 64 chars)
  hermes:
    tier_order:                    # optional; same permutation rule
      standard: [gemini-3.8-flash, claude-sonnet-5.5, claude-sonnet-5]
    threshold_offsets: {standard: 1, heavy: 0}   # added to base thresholds
    weight: 0.6                    # shrinkage toward global, 0..1
```

Semantics:

- The base config always wins on guardrails. The overlay cannot add models,
  move a model across tiers, change prompt windows, vision, scoring or
  headroom (unknown keys are rejected).
- A per-caller `tier_order` overrides the global (overlay) order for that
  tier, for that caller only. Its effective threshold is
  `base + round(weight * offset)`: `weight: 0` behaves exactly like the
  global router, `1` applies the learned offset in full. The tuning job
  should derive `weight` from sample size, e.g. `n / (n + k)`.
- Effective thresholds must still satisfy `heavy > standard`.
- Any violation rejects the whole file: the previous good state stays
  active, an error is logged and `pinball_overlay_reload_errors_total{reason}`
  is incremented. A missing file means base config only; deleting the file
  reverts to base.
- Each decision log line carries `overlay=<version>` and `profile=<caller>`
  (`-` when none applied).

## Nightly optimizer

`pinball-optimize` (or `python -m pinball.optimize`) re-ranks models
**within** each tier from the previous day's real traffic and writes the
overlay above. It never adds models, moves a model across tiers or touches
thresholds. Install with the `optimize` extra (`uv sync --extra optimize`,
psycopg); the serving image does not include it.

```sh
pinball-optimize run --dry-run              # sample + plan + projected cost, no model calls
pinball-optimize run --seed 20261103 --per-tier 20 --days 1 [--candidates a,b]
pinball-optimize --check-revert             # run every few hours after `run`
pinball-optimize discover [--dry-run]       # new-model PRs only (see "New models")
```

Flow of one `run`:

1. **Sample** (read-only Postgres, `PINBALL_OPT_DB_DSN`): successful `auto`
   requests from the last `--days`, excluding `eval-*`/`pinball-*` callers,
   `--per-tier` per tier with a seeded shuffle (default seed: today's date).
2. **Replay** each sample on every other model of its current tier
   (caller `eval-replay`), then ask both judges (`judge-opus`, `judge-argon`,
   caller `eval-replay-judge`) to compare served vs candidate with a seeded
   A/B order. A verdict field counts only when the judges agree.
3. **Rank** per (tier, model), using agreed fields from both sides of each pair:
   `quality = mean(acceptable_rate, win_rate)` (tie = ½ win);
   `shrunk = (n·quality + k·prior) / (n + k)` where `prior = 0.75 − 0.05·position`
   encodes the current order; `score = shrunk − cost_weight · cost_per_request / tier_median`.
   Models with fewer than `min_samples` **non-agentic** agreed comparisons are
   pinned. Moves are adjacent swaps whose score gap exceeds `margin`, at most
   `max_moves` per tier, and each slot moves at most once a night.
4. **Apply**: the candidate overlay goes through the same `merge()` pinball
   uses; an invalid one is not written (exit 3). Valid ones are written to
   `PINBALL_OVERLAY_PATH` atomically (temp file in the same dir + `os.replace`)
   with version `opt-<seq>-<UTC stamp>` (monotonic). The previous overlay is
   kept in the state file for revert.
5. **Audit**: one JSON line per run/check (`PINBALL_OPT_AUDIT_PATH`): sample
   counts, agentic share, per-model n/quality/score/cost, moves and why,
   spend, `budget_exhausted`, previous version, and outcome counts:
   `budget_refused` (pairs never sent for lack of budget; not an error),
   `errors` (real failures only), `gateway_errors` (candidate model → label:
   `HTTP <status>`, `transport <type>`, `timeout`), `judge_unparseable` and
   `judge_failed` per judge, and `tier_budget` (share and spend per tier).
   No prompt, response or error-body text is ever persisted; judge reasons
   stay in memory.

**Spend cap.** Every gateway call first reserves its worst case (input
tokens bounded at chars/2, output at `max_tokens`) at the model's price; a
call that does not fit under `PINBALL_OPT_BUDGET_USD` (default $20) is never
issued. The cap is stratified: pairs go out round-robin over (tier, candidate),
so every candidate gets a pair before any gets a second, and each tier with
pairs gets a share of the cap by `PINBALL_OPT_TIER_WEIGHTS` (default equal). A
tier cannot spend past its share while another tier has pairs left; a
finished tier's leftover rolls over. A pair running longer than
`PINBALL_OPT_PAIR_DEADLINE` is recorded as gateway error `timeout` and its
worker slot is freed. After the call the reserve is replaced by the reported usage. The run
finishes with whatever was judged and records `budget_exhausted: true`. The
default price table (`settings.py`) holds **estimates**, not invoices; override
it with `PINBALL_OPT_PRICES` (YAML `{models: {name: {input, output}}, default: {...}}`,
USD per million tokens). Unknown models are priced like the most expensive tier.

**Auto-revert** (`--check-revert`) reads request_logs (no Prometheus
dependency): for `auto` traffic it compares the `hours` after the last
optimizer write with the `hours` before. It restores the previous overlay
(or deletes the file if there was none) when error rate (non-2xx share)
rises by more than `error_margin` relative and `error_floor` absolute, or
cost per request rises by more than `cost_margin`, with at least
`min_requests` in each window. Overlays not written by the optimizer are left alone.

**Per-caller profiles.** The gateway stores the `x-agw-caller` header in
`request_logs.agentgateway_user` (the column the `eval-*`/`pinball-*`
exclusion already reads); the optimizer reads it cut to 64 chars, the same key
pinball matches. On top of the global sample, every caller with at least
`PINBALL_OPT_CALLER_MIN_SAMPLES` pool rows (busiest first, at most
`_MAX_PROFILES`) gets up to `_CALLER_PER_TIER` extra samples per tier from rows
the global sample did not take (seed `<seed>:<caller>:<tier>`). The global
ranking only ever sees the global sample. For a caller with at least
`_CALLER_MIN_SAMPLES` **judged non-agentic** samples (global + its own):

- `tier_order`: the same rerank (margin, max moves, agentic cap) over that
  caller's evidence, starting from tonight's global order; written only for
  tiers whose order differs.
- `weight = n / (n + PINBALL_OPT_CALLER_SHRINK_K)`.
- `threshold_offsets.standard = -1` only when the caller's light-tier served
  replies are judged unacceptable more often than the global light rate by
  more than `_CALLER_LIGHT_FAIL_GAP` (with at least `PINBALL_OPT_MIN_SAMPLES`
  light verdicts). Nothing else is ever learned; `heavy` stays 0.

A caller below the minimum, or whose evidence matches global, gets no profile
(it inherits global). Each profile is checked with `merge()` before the write
(an invalid one is skipped and audited as `rejected <reason>`); at most
`_CALLER_MAX_CHANGES` profiles are added or changed per night. A profile whose
caller has not appeared in the pool for `_CALLER_STALE_DAYS` is dropped.
Per-caller state (`last_seen`, `changed_at`, previous profile) lives in
`<overlay>.callers.json`. The audit line's `callers` map carries counts,
scores, moves and the result per caller, never text.
`--check-revert` also compares each changed caller's own traffic over the same
windows and margins (needs `_CALLER_REVERT_MIN_REQUESTS` per window) and
restores only that caller's previous profile, writing a new overlay version
(`result: callers_reverted`); a global regression still reverts the whole overlay.

**Blind spot: agentic traffic.** The gateway log strips tool calls, tool
results and tool definitions. Tool turns are dropped from the replay and the
sample is marked `agentic`; judges see only the text. So agentic requests are
partially visible: their evidence does not count toward `min_samples`, the
report carries the agentic share per tier, and a tier whose agentic share is
≥ `agentic_share_high` is limited to `agentic_max_moves` moves per night.

| Variable | Default |
| --- | --- |
| `PINBALL_OPT_DB_DSN` | required (read-only role recommended) |
| `PINBALL_OPT_GATEWAY_URL` | required for `run` (not for dry-run/check) |
| `PINBALL_OPT_GATEWAY_TIMEOUT` | `300` s |
| `PINBALL_OPT_BUDGET_USD` | `20.0` |
| `PINBALL_OPT_TIER_WEIGHTS` | equal (e.g. `light=1,standard=1,heavy=2`) |
| `PINBALL_OPT_PAIR_DEADLINE` | `180` s per (sample, candidate) |
| `PINBALL_OPT_PRICES` | built-in estimates |
| `PINBALL_OPT_JUDGES` | `judge-opus,judge-argon` |
| `PINBALL_OPT_WORKERS` | `4` |
| `PINBALL_OPT_MODEL_PREFIX` | empty (candidates are sent as `<prefix><pool-id>`, e.g. `opt-`; judges, reports, overlay and prices keep bare ids) |
| `PINBALL_OVERLAY_PATH` | unset (nothing written) |
| `PINBALL_CONFIG` | packaged `pinball.yaml` |
| `PINBALL_OPT_AUDIT_PATH` | `optimize-audit.jsonl` next to the overlay |
| `PINBALL_OPT_STATE_PATH` | `<overlay>.state.json` |
| `PINBALL_OPT_MIN_SAMPLES` / `_SHRINK_K` / `_COST_WEIGHT` / `_SWAP_MARGIN` | `10` / `20` / `0.1` / `0.02` |
| `PINBALL_OPT_MAX_MOVES` / `_AGENTIC_SHARE_HIGH` / `_AGENTIC_MAX_MOVES` | `2` / `0.5` / `1` |
| `PINBALL_OPT_CALLER_MIN_SAMPLES` / `_PER_TIER` / `_SHRINK_K` | `30` / `15` / `30` |
| `PINBALL_OPT_CALLER_MAX_PROFILES` / `_MAX_CHANGES` / `_STALE_DAYS` | `20` / `5` / `14` |
| `PINBALL_OPT_CALLER_LIGHT_FAIL_GAP` / `_REVERT_MIN_REQUESTS` | `0.15` / `20` |
| `PINBALL_OPT_REVERT_HOURS` / `_ERROR_MARGIN` / `_ERROR_FLOOR` / `_COST_MARGIN` / `_MIN_REQUESTS` | `6` / `0.5` / `0.01` / `0.25` / `50` |
| `PINBALL_GITEA_TOKEN` | unset: the pass is skipped (never logged) |
| `PINBALL_GITEA_URL` / `PINBALL_GITEA_REPO` | `https://gitea.grizzlygeek.net` / `jeefy/carrier` |
| `PINBALL_OPT_MODELS_URL` | `http://copilot-api.carrier.svc.cluster.local:4141/v1/models` |
| `PINBALL_OPT_CANDIDATES` | unset: no evaluation |
| `PINBALL_OPT_CANDIDATE_BUDGET_USD` | `5.0` (separate from the ranking budget) |
| `PINBALL_OPT_MERGE_WITHOUT_CHECKS` | off: a PR with no status is never merged |
| `PINBALL_OPT_MAX_PRS` / `_CANDIDATE_NIGHTS` | `3` / `7` |
| `PINBALL_OPT_DISCOVERY_STRICT` | off: a failed discovery pass never changes the exit code |

CronJob note: build the image with `uv sync --locked --no-dev --extra optimize`
and use `command: ["pinball-optimize", "run"]` (nightly) plus
`["pinball-optimize", "--check-revert"]` (a few hours later), sharing the
overlay volume with pinball.

## New models

The nightly `run` (and `pinball-optimize discover [--dry-run]`) also looks for
new Copilot models and wires winners into the pool through bot PRs on the
deployment repo (agentgateway cannot route to a model chosen at runtime, so
every pool model needs literal gateway config). Two steps, both auto-merged:

1. **Candidate PR** (`pinball-bot/candidate-<id>`): a chat model in
   `PINBALL_OPT_MODELS_URL` (copilot-api `/v1/models`; embeddings, dated
   snapshots and legacy GPT-3.5/4/4o ignored) that is not in the pool, the
   candidate list or its `dropped` list gets ONLY a public `opt-<id>` gateway
   alias and a `candidates` entry in `k8s/pinball-candidates.yaml`. Nothing
   live changes.
2. **Promotion PR** (`pinball-bot/promote-<id>`): on later nights every
   candidate in the mounted list (`PINBALL_OPT_CANDIDATES`) is replayed on the
   night's samples of every tier and judged against what the pool served.
   It wins a tier when it has `PINBALL_OPT_MIN_SAMPLES` non-agentic agreed
   comparisons and, scored as if appended last, beats the weakest incumbent
   by the swap margin (largest gap wins). The PR adds it to `models`
   (`prompt_tokens` = catalogue window capped at 250k, `vision` from the
   catalogue; missing values default to 100k / false and are flagged in the
   PR body), at the END of that tier, plus `pinball-<id>`, the `auto` target
   and placeholder cost rates. The overlay may then move it up.
   A candidate that won nothing after `PINBALL_OPT_CANDIDATE_NIGHTS` (7) is
   dropped by a PR (`pinball-bot/drop-<id>`) that removes the alias.

Safety: every edit is a line insertion/rewrite next to an anchor (comments
untouched); before committing, the edited files must pass
`optimize/invariants.py`, a copy of the deployment's pool drift test. The bot
merges only after the PR head has a GREEN commit status (it does not rely on
Gitea's merge-when-checks-succeed, which merges at once without branch
protection); failed checks leave the PR open, pending ones are retried the
next night, and PRs are serial (one unmerged PR blocks further ones). At most
`PINBALL_OPT_MAX_PRS` (3) new PRs per night; an existing branch or PR for the
same step is never duplicated. A discovery failure is logged and reported
but never fails the ranking unless `PINBALL_OPT_DISCOVERY_STRICT=1`.

The variables are in the optimizer table above.

## Running

```sh
uv run python -m pinball                   # :9201
curl -s localhost:9201/check -d '{"model":"auto","messages":[{"role":"user","content":"Generate a title"}],"max_tokens":20}'
# {"model":"claude-haiku-5.5","tier":"light","reason":"light:title,short,small_output","score":-3,"est_tokens":4}
```

| Variable | Default |
| --- | --- |
| `PINBALL_HOST` | `0.0.0.0` |
| `PINBALL_PORT` | `9201` |
| `PINBALL_CONFIG` | packaged `pinball.yaml` |
| `PINBALL_OVERLAY_PATH` | unset (base config only) |
| `PINBALL_OVERLAY_RELOAD_SECONDS` | `30` |
| `LOG_LEVEL` | `INFO` |

| Endpoint | |
| --- | --- |
| `POST /<any path>` | the check (agentgateway sends to `protocol.http.path`) |
| `GET /healthz` | liveness/readiness |
| `GET /metrics` | `pinball_decisions_total{tier,model}`, `pinball_signals_total{signal}`, `pinball_skipped_total{reason}`, `pinball_errors_total`, `pinball_decide_seconds`, `pinball_body_bytes`, `pinball_overlay_info{version}`, `pinball_overlay_generated_timestamp_seconds`, `pinball_overlay_reload_total{result}`, `pinball_overlay_reload_errors_total{reason}` |

Set `reply_signals: true` in the config to add `"signals": [...]` to the reply:
the signal names behind the decision (e.g. `["short", "min_tier"]`, no values, no
request content), so the gateway can copy them into `extauthz.signals`. Off by default.

Each decision is logged as one line with tier, model, score, estimated tokens,
body size, latency and signal names. Message content and credentials are
never logged.

The container image (`ghcr.io/jeefy/pinball`) runs as uid 1000 with a
read-only root filesystem.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest --cov
scripts/twin.sh        # agentgateway v1.6.0 + pinball + stub upstream, end to end
```

`scripts/twin.sh` runs `examples/agentgateway.yaml` in podman or docker
against a local pinball and a chat-completions-only stub upstream. It checks
routing for all three API formats, the access-log attributes, that internal
models are hidden, metrics, clean shutdown, and the fail-open fallback.

## Limits

- The upstream must speak chat completions. Models that only serve the
  Responses API need a translating upstream (the homelab's copilot-api
  0ea08fe-p5 does this for Copilot's GPT-5.x/6.x, grok and mai models).
- Scoring is heuristic. An embedding classifier is planned once the decision
  logs show where the heuristics miss.
- Clients that compress or truncate history can lose the message that set
  the tier, so the tier may drop.
