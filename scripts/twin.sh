#!/usr/bin/env bash
# Local end-to-end twin for pinball: agentgateway v1.6.0 (podman or docker,
# --network host) running examples/agentgateway.yaml, with its extAuthz host
# pointed at a locally running pinball and its completions provider pointed at
# a stub upstream started here. Nothing talks to a real LLM provider.
#
# Checks: `auto` on /v1/chat/completions, /v1/messages and /v1/responses
# reaches upstream /v1/chat/completions as the model pinball picked (light,
# heavy, x-pinball-tier hint, a ~2.4 MiB prompt that only kimi-k3 fits);
# the access log carries pinball.* attributes; the internal pinball-* models
# are 404 to clients; pinball's metrics count the decisions; pinball exits on
# SIGTERM without logging an error; with pinball down `auto` still answers via
# the fallback target (failureMode: allow).
#
# Usage: scripts/twin.sh        (needs podman or docker, uv, curl, python3)
#
# CONTAINER_RUNTIME picks podman or docker (default: podman if installed, else
# docker). Logs go to TWIN_LOG_DIR, default ${RUNNER_TEMP:-/tmp}/pinball-twin-logs.
# Exits non-zero if any check fails or the twin cannot start.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${AGW_IMAGE:-cr.agentgateway.dev/agentgateway:v1.6.0}"
GW_PORT="${TWIN_GW_PORT:-24000}"
ADMIN_PORT="${TWIN_ADMIN_PORT:-25100}"
STATS_PORT="${TWIN_STATS_PORT:-25120}"
READY_PORT="${TWIN_READY_PORT:-25121}"
PINBALL_PORT="${TWIN_PINBALL_PORT:-29201}"
STUB_PORT="${TWIN_STUB_PORT:-28180}"
CONTAINER="pinball-twin-agw"
LOG_DIR="${TWIN_LOG_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/pinball-twin-logs}"

RUNTIME="${CONTAINER_RUNTIME:-}"
if [[ -z "${RUNTIME}" ]]; then
    if command -v podman >/dev/null 2>&1; then
        RUNTIME=podman
    elif command -v docker >/dev/null 2>&1; then
        RUNTIME=docker
    else
        echo "FAIL: neither podman nor docker found (or set CONTAINER_RUNTIME)" >&2
        exit 1
    fi
fi
for tool in "${RUNTIME}" uv curl python3; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "FAIL: ${tool} not found" >&2
        exit 1
    fi
done

mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}"/{agentgateway.log,agentgateway-config.yaml,stub.log,pinball.log,metrics.txt}
WORK="$(mktemp -d "${TMPDIR:-/tmp}/pinball-twin.XXXXXX")"

STUB_PID=""
PIN_PID=""
PASSED=0
FAILED=0
FAILURES=()
FINISHED=0

pass() {
    PASSED=$((PASSED + 1))
    echo "PASS: $1"
}

fail() {
    FAILED=$((FAILED + 1))
    FAILURES+=("$1")
    echo "FAIL: $1"
}

kill_tree() {
    local child
    for child in $(pgrep -P "$2" 2>/dev/null || true); do
        kill_tree "$1" "${child}"
    done
    kill "-$1" "$2" 2>/dev/null || true
}

# stop_pid <pid>: SIGTERM, up to 10 s to exit, then SIGKILL the tree.
# Returns 1 if it had to be killed.
stop_pid() {
    local pid="$1"
    [[ -n "${pid}" ]] || return 0
    kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 100); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            wait "${pid}" 2>/dev/null || true
            return 0
        fi
        sleep 0.1
    done
    kill_tree KILL "${pid}"
    wait "${pid}" 2>/dev/null || true
    return 1
}

cleanup() {
    local rc=$? log
    trap - ERR
    set +e
    "${RUNTIME}" container inspect "${CONTAINER}" >/dev/null 2>&1 &&
        "${RUNTIME}" logs "${CONTAINER}" >"${LOG_DIR}/agentgateway.log" 2>&1
    "${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1
    stop_pid "${PIN_PID}"
    stop_pid "${STUB_PID}"
    cp "${WORK}/pinball.log" "${WORK}/stub.log" "${WORK}/metrics.txt" "${LOG_DIR}/" 2>/dev/null
    cp "${WORK}/agw/config.yaml" "${LOG_DIR}/agentgateway-config.yaml" 2>/dev/null
    if [[ "${FINISHED}" == 0 ]]; then
        echo "RESULT: twin ABORTED (exit ${rc}) after ${PASSED} passed and ${FAILED} failed checks"
        for log in "${LOG_DIR}/pinball.log" "${LOG_DIR}/stub.log" "${LOG_DIR}/agentgateway.log"; do
            [[ -s "${log}" ]] || continue
            echo "--- tail ${log}"
            tail -n 20 "${log}"
        done
    fi
    echo "logs: ${LOG_DIR}"
    if [[ "${KEEP_TWIN_WORKDIR:-0}" == 1 ]]; then
        echo "workdir kept: ${WORK}"
    else
        rm -rf "${WORK}"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'rc=$?; [[ $- == *e* ]] && echo "ERROR: line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

# wait_port <port> <what> [pid]: up to ~20 s, or until pid exits.
wait_port() {
    local port="$1" what="$2" pid="${3:-}"
    for _ in $(seq 1 100); do
        if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
            return 0
        fi
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "FAIL: ${what} exited before opening 127.0.0.1:${port}" >&2
            exit 1
        fi
        sleep 0.2
    done
    echo "FAIL: ${what} did not open 127.0.0.1:${port}" >&2
    exit 1
}

"${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
for port in "${GW_PORT}" "${ADMIN_PORT}" "${STATS_PORT}" "${READY_PORT}" "${PINBALL_PORT}" \
    "${STUB_PORT}"; do
    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
        echo "FAIL: 127.0.0.1:${port} already in use (leftover twin? set TWIN_*_PORT)" >&2
        exit 1
    fi
done
if ! "${RUNTIME}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "pulling ${IMAGE} with ${RUNTIME}"
    "${RUNTIME}" pull -q "${IMAGE}" >/dev/null
fi

# --- stub upstream ----------------------------------------------------------
# Chat completions only (like copilot-api): appends {"path", "model", "bytes"}
# per request to upstream.jsonl and answers "served-by:<model>".
cat >"${WORK}/stub.py" <<'PY'
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT = sys.argv[2]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_body(self):
        if "chunked" in (self.headers.get("transfer-encoding") or "").lower():
            body = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return body
                body += self.rfile.read(size)
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("content-length") or 0))

    def _send(self, code, obj):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        body = self._read_body()
        try:
            model = json.loads(body).get("model")
        except ValueError:
            model = None
        with open(f"{OUT}/upstream.jsonl", "a") as fh:
            fh.write(json.dumps({"path": self.path, "model": model, "bytes": len(body)}) + "\n")
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {
            "id": "chatcmpl-twin", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": f"served-by:{model}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
PY
touch "${WORK}/upstream.jsonl"
python3 "${WORK}/stub.py" "${STUB_PORT}" "${WORK}" 2>"${WORK}/stub.log" &
STUB_PID=$!
wait_port "${STUB_PORT}" "stub upstream" "${STUB_PID}"

# --- agentgateway config -------------------------------------------------------
# examples/agentgateway.yaml itself; only ports, the extAuthz host and the
# provider baseUrl are rewritten for localhost.
mkdir -p "${WORK}/agw"
uv run --project "${ROOT}" --quiet python - "${ROOT}/examples/agentgateway.yaml" \
    "${WORK}/agw/config.yaml" "${ADMIN_PORT}" "${STATS_PORT}" "${READY_PORT}" "${GW_PORT}" \
    "${PINBALL_PORT}" "${STUB_PORT}" <<'PY'
import sys

import yaml

src, dst, admin, stats, ready, gw, pinball, stub = sys.argv[1:]
cfg = yaml.safe_load(open(src))
llm = cfg["llm"]
llm["port"] = int(gw)
(provider,) = llm["providers"]
provider["params"]["baseUrl"] = f"http://127.0.0.1:{stub}/v1"
(authz,) = llm["policies"]["extAuthz"]["conditional"]
authz["host"] = f"127.0.0.1:{pinball}"
cfg["config"] = {
    "adminAddr": f"127.0.0.1:{admin}",
    "statsAddr": f"127.0.0.1:{stats}",
    "readinessAddr": f"127.0.0.1:{ready}",
}
with open(dst, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
PY
# The image runs as uid 65532; keep the bind-mounted config readable.
chmod 755 "${WORK}/agw"
chmod 644 "${WORK}/agw/config.yaml"

if "${RUNTIME}" run --rm --network host -v "${WORK}/agw:/config:Z" "${IMAGE}" \
    -f /config/config.yaml --validate-only >"${WORK}/validate.log" 2>&1; then
    pass "agentgateway --validate-only accepts examples/agentgateway.yaml"
else
    fail "agentgateway --validate-only rejected examples/agentgateway.yaml"
    cat "${WORK}/validate.log"
fi

PINBALL_PORT="${PINBALL_PORT}" PINBALL_HOST=127.0.0.1 \
    uv run --project "${ROOT}" --quiet python -m pinball >"${WORK}/pinball.log" 2>&1 &
PIN_PID=$!
wait_port "${PINBALL_PORT}" "pinball" "${PIN_PID}"

"${RUNTIME}" run -d --name "${CONTAINER}" --network host \
    -v "${WORK}/agw:/config:Z" "${IMAGE}" -f /config/config.yaml >/dev/null
wait_port "${GW_PORT}" "agentgateway"

# --- requests -------------------------------------------------------------------
python3 - "${WORK}" <<'PY'
import json, sys
work = sys.argv[1]
tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
debug = "Find the root cause of this deadlock:\n```go\nmu.Lock()\nch <- v\n```"
bodies = {
    "light": {"model": "auto", "max_tokens": 30,
              "messages": [{"role": "user", "content": "Generate a short title for this chat"}]},
    "heavy": {"model": "auto", "tools": tools,
              "messages": [{"role": "system", "content": "You are a coding agent."},
                           {"role": "user", "content": debug}]},
    "messages": {"model": "auto", "max_tokens": 8000,
                 "thinking": {"type": "enabled", "budget_tokens": 32000},
                 "messages": [{"role": "user", "content": [{"type": "text", "text": debug}]}]},
    "responses": {"model": "auto", "input": "Summarize this in one line: it rained.",
                  "max_output_tokens": 50},
    "hint": {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    # ~2.4 MiB of prompt (~690k estimated tokens): only kimi-k3's window fits.
    "big": {"model": "auto",
            "messages": [{"role": "user", "content": "lorem ipsum " * (200 << 10)}]},
    "internal": {"model": "pinball-claude-opus-5.5",
                 "messages": [{"role": "user", "content": "hi"}]},
}
for name, body in bodies.items():
    with open(f"{work}/req-{name}.json", "w") as fh:
        json.dump(body, fh)
PY

# post <request> <path> [curl args]: sets CODE and UP (the stub's record of this
# request as "path model", or "none" if it never reached upstream).
post() {
    local name="$1" path="$2" seen
    shift 2
    seen="$(wc -l <"${WORK}/upstream.jsonl")"
    CODE="$(curl -sS --max-time 30 -o "${WORK}/out-${name}.json" -w '%{http_code}' \
        -H 'content-type: application/json' -H 'authorization: Bearer twin-dummy' \
        -H 'x-agw-caller: twin' "$@" \
        --data-binary "@${WORK}/req-${name}.json" "http://127.0.0.1:${GW_PORT}${path}")" || true
    UP="$(python3 - "${WORK}/upstream.jsonl" "${seen}" <<'PY'
import json, sys
lines = open(sys.argv[1]).read().splitlines()[int(sys.argv[2]):]
print(" ".join((lambda r: (r["path"], str(r["model"])))(json.loads(lines[-1]))) if lines else "none")
PY
)"
}

# expect <desc> <http code> <upstream "path model">
expect() {
    if [[ "${CODE}" == "$2" && "${UP}" == "$3" ]]; then
        pass "$1: HTTP ${CODE}, upstream ${UP}"
    else
        fail "$1: HTTP ${CODE:-000}, upstream ${UP}; expected HTTP $2, upstream $3"
    fi
}

# client_has <desc> <request name> <python expr over `out` (parsed JSON)>
client_has() {
    if python3 -c "import json,sys; out=json.load(open(sys.argv[1])); sys.exit(0 if ($3) else 1)" \
        "${WORK}/out-$2.json" 2>/dev/null; then
        pass "$1"
    else
        fail "$1"
    fi
}

echo "== pinball up"
post light /v1/chat/completions
expect "light chat completion" 200 "/v1/chat/completions claude-haiku-5.5"
client_has "light: client sees the served model" light \
    'out["choices"][0]["message"]["content"] == "served-by:claude-haiku-5.5"'

post heavy /v1/chat/completions
expect "heavy agentic debugging" 200 "/v1/chat/completions claude-opus-5.5"

post messages /v1/messages
expect "anthropic /v1/messages translated to completions" 200 \
    "/v1/chat/completions claude-opus-5.5"
client_has "messages: client gets an Anthropic message" messages \
    'out["type"] == "message" and out["content"][0]["text"] == "served-by:claude-opus-5.5"'

post responses /v1/responses
expect "openai /v1/responses translated to completions" 200 \
    "/v1/chat/completions claude-haiku-5.5"
client_has "responses: client gets a Responses object" responses \
    'out["object"] == "response" and "served-by:claude-haiku-5.5" in json.dumps(out["output"])'

post hint /v1/chat/completions -H 'x-pinball-tier: heavy'
expect "x-pinball-tier hint" 200 "/v1/chat/completions claude-opus-5.5"

post big /v1/chat/completions
expect "2.4 MiB prompt goes to the largest window" 200 "/v1/chat/completions kimi-k3"

post internal /v1/chat/completions
expect "internal pinball-* model is not client-reachable" 404 "none"

curl -sS "http://127.0.0.1:${PINBALL_PORT}/metrics" >"${WORK}/metrics.txt" || true
grep -E '^pinball_decisions_total' "${WORK}/metrics.txt" || true
if python3 - "${WORK}/metrics.txt" <<'PY'; then
import re, sys
text = open(sys.argv[1]).read()
total = sum(float(v) for v in re.findall(r"^pinball_decisions_total\{[^}]*\} (\S+)", text, re.M))
sys.exit(0 if total == 6 and "pinball_errors_total 0.0" in text else 1)
PY
    pass "metrics: 6 decisions, no errors"
else
    fail "metrics: expected 6 decisions and pinball_errors_total 0"
fi

# Access-log lines are written after the response ends.
for _ in $(seq 1 25); do
    "${RUNTIME}" logs "${CONTAINER}" >"${WORK}/agw.log" 2>&1 || true
    if grep -q 'pinball.model="kimi-k3"' "${WORK}/agw.log"; then
        break
    fi
    sleep 0.2
done
if grep -q 'pinball.model="claude-haiku-5.5" pinball.tier="light"' "${WORK}/agw.log" &&
    grep -q 'pinball.model="kimi-k3" pinball.tier="standard" pinball.reason=".*fit:kimi-k3' \
        "${WORK}/agw.log"; then
    pass "access log carries pinball.model and pinball.tier"
else
    fail "access log is missing pinball.* attributes"
fi

if grep -q "decide tier=light model=claude-haiku-5.5 .*listener=${GW_PORT} caller=twin " \
    "${WORK}/pinball.log"; then
    pass "pinball sees the listener port and X-AGW-Caller"
else
    fail "pinball log lacks listener=${GW_PORT} caller=twin"
fi

started="${SECONDS}"
if stop_pid "${PIN_PID}"; then
    pass "pinball exited on SIGTERM in $((SECONDS - started))s"
else
    fail "pinball still running 10s after SIGTERM; killed"
fi
PIN_PID=""
if grep -qE 'ERROR|Traceback' "${WORK}/pinball.log"; then
    fail "pinball logged an error"
else
    pass "pinball logged no errors"
fi

echo "== pinball down (failureMode: allow)"
post heavy /v1/chat/completions
expect "auto falls back when pinball is unreachable" 200 \
    "/v1/chat/completions claude-sonnet-5.5"

FINISHED=1
if [[ "${FAILED}" == 0 ]]; then
    echo "RESULT: all ${PASSED} twin checks passed"
else
    echo "RESULT: ${FAILED} of $((PASSED + FAILED)) twin checks FAILED:"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi
