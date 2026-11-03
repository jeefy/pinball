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
gateway forwards it, overrides the score.

All of this lives in [`pinball.yaml`](pinball.yaml): the model pool with
prompt windows, per-tier preference order, thresholds, signal points and
cutoffs (`scoring`, every key optional), wrapped user messages, headroom and
keyword lists. Mount your own copy and point `PINBALL_CONFIG` at it.

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
| `LOG_LEVEL` | `INFO` |

| Endpoint | |
| --- | --- |
| `POST /<any path>` | the check (agentgateway sends to `protocol.http.path`) |
| `GET /healthz` | liveness/readiness |
| `GET /metrics` | `pinball_decisions_total{tier,model}`, `pinball_signals_total{signal}`, `pinball_skipped_total{reason}`, `pinball_errors_total`, `pinball_decide_seconds`, `pinball_body_bytes` |

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
  Responses API cannot be in the pool behind copilot-api.
- Scoring is heuristic. An embedding classifier is planned once the decision
  logs show where the heuristics miss.
- Clients that compress or truncate history can lose the message that set
  the tier, so the tier may drop.
