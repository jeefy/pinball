"""In-memory Gitea 1.2x over a real local HTTP server (the subset the bot calls)."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


@dataclass
class Repo:
    files: dict[str, str]  # main
    branches: dict[str, dict[str, str]] = field(default_factory=dict)
    pulls: list[dict[str, Any]] = field(default_factory=list)
    status: str = "success"  # combined state reported for every PR head
    statuses: int = 1  # how many statuses (0: no PR checks at all)
    requests: list[tuple[str, str, str]] = field(default_factory=list)  # method, path, auth
    merged: list[int] = field(default_factory=list)

    @property
    def writes(self) -> list[tuple[str, str, str]]:
        return [r for r in self.requests if r[0] != "GET"]


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class FakeGitea:
    def __init__(self, repo: Repo) -> None:
        self.repo = repo
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> FakeGitea:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        repo = self.repo

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send(self, code: int, doc: Any = None) -> None:
                raw = b"" if doc is None else json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _route(self, method: str) -> None:
                url = urlparse(self.path)
                path = url.path.removeprefix("/api/v1/repos/jeefy/carrier")
                repo.requests.append((method, path, self.headers.get("authorization", "")))
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                self._send(*dispatch(method, unquote(path), q, body))

            def do_GET(self) -> None:
                self._route("GET")

            def do_POST(self) -> None:
                self._route("POST")

        def dispatch(method: str, path: str, q: dict[str, str], body: Any) -> tuple[int, Any]:
            if method == "GET" and path.startswith("/contents/"):
                text = repo.files.get(path.removeprefix("/contents/"))
                if text is None:
                    return 404, None
                return 200, {"content": base64.b64encode(text.encode()).decode(), "sha": _sha(text)}
            if method == "GET" and path.startswith("/branches/"):
                return (
                    (200, {}) if path.removeprefix("/branches/") in repo.branches else (404, None)
                )
            if method == "GET" and path == "/pulls":
                page, limit = int(q.get("page", 1)), int(q.get("limit", 50))
                return 200, repo.pulls[(page - 1) * limit : page * limit]
            if method == "POST" and path == "/contents":
                if body["new_branch"] in repo.branches:
                    return 422, None
                files = dict(repo.files)
                for f in body["files"]:
                    if f["sha"] != _sha(repo.files[f["path"]]):
                        return 409, None
                    files[f["path"]] = base64.b64decode(f["content"]).decode()
                repo.branches[body["new_branch"]] = files
                return 201, {}
            if method == "POST" and path == "/pulls":
                number = len(repo.pulls) + 1
                head = {"ref": body["head"], "sha": f"sha{number}"}
                repo.pulls.append(
                    {"number": number, "state": "open", "head": head, "body": body["body"]}
                )
                return 201, {"number": number}
            m = re.fullmatch(r"/commits/(\w+)/status", path)
            if method == "GET" and m:
                return 200, {"state": repo.status, "statuses": [{}] * repo.statuses}
            m = re.fullmatch(r"/pulls/(\d+)/merge", path)
            if method == "POST" and m:
                pr = repo.pulls[int(m.group(1)) - 1]
                repo.files = repo.branches.pop(pr["head"]["ref"])
                pr.update(state="closed", merged=True)
                repo.merged.append(pr["number"])
                return 200, None
            return 404, None

        return Handler
