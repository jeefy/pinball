"""Minimal Gitea (1.2x) REST client over stdlib HTTP for the carrier bot PRs.

The token is sent only in the Authorization header and never appears in an
exception, a log line or a repr. `read_only=True` (dry-run) refuses every
non-GET request before it reaches the socket.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Final

DEFAULT_URL: Final = "https://gitea.grizzlygeek.net"
DEFAULT_REPO: Final = "jeefy/carrier"
BASE_BRANCH: Final = "main"


class GiteaError(RuntimeError):
    """Content-free: method, path and status only."""

    def __init__(self, label: str, status: int | None = None) -> None:
        super().__init__(label)
        self.status = status


@dataclass(frozen=True, slots=True)
class PullRequest:
    number: int
    branch: str
    state: str  # open | closed
    merged: bool
    head_sha: str


@dataclass(frozen=True, slots=True)
class FileEdit:
    path: str
    content: str
    sha: str  # blob sha the edit is based on (optimistic concurrency)


@dataclass
class Gitea:
    base_url: str
    repo: str
    token: str = field(repr=False)
    read_only: bool = False
    timeout: float = 30.0

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        if self.read_only and method != "GET":
            raise GiteaError(f"read-only client refused {method} {path}")
        url = f"{self.base_url.rstrip('/')}/api/v1/repos/{self.repo}{path}"
        data = None if body is None else json.dumps(body).encode()
        headers = {"accept": "application/json", "content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"token {self.token}"
        request = urllib.request.Request(url, data, headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as e:
            raise GiteaError(f"{method} {path}: HTTP {e.code}", e.code) from None
        except (urllib.error.URLError, OSError) as e:
            raise GiteaError(f"{method} {path}: transport {type(e).__name__}") from None
        return json.loads(raw) if raw else None

    def file(self, path: str, ref: str = BASE_BRANCH) -> tuple[str, str]:
        """(text, blob sha) of `path` at `ref`."""
        q = urllib.parse.urlencode({"ref": ref})
        doc = self._call("GET", f"/contents/{urllib.parse.quote(path)}?{q}")
        return base64.b64decode(doc["content"]).decode("utf-8"), str(doc["sha"])

    def branch_exists(self, branch: str) -> bool:
        try:
            self._call("GET", f"/branches/{urllib.parse.quote(branch, safe='')}")
        except GiteaError as e:
            if e.status == 404:
                return False
            raise
        return True

    def pulls(self, branch: str) -> list[PullRequest]:
        """Every PR (open or closed) whose head is `branch`."""
        out: list[PullRequest] = []
        for page in range(1, 51):
            q = urllib.parse.urlencode({"state": "all", "page": page, "limit": 50})
            batch = self._call("GET", f"/pulls?{q}") or []
            out.extend(
                PullRequest(
                    number=int(p["number"]),
                    branch=str(p["head"]["ref"]),
                    state=str(p["state"]),
                    merged=bool(p.get("merged")),
                    head_sha=str(p["head"]["sha"]),
                )
                for p in batch
                if p["head"]["ref"] == branch
            )
            if len(batch) < 50:
                break
        return out

    def commit_files(self, branch: str, message: str, edits: list[FileEdit]) -> None:
        """One commit on a NEW branch cut from main (contents API, Gitea >= 1.20)."""
        files = [
            {
                "operation": "update",
                "path": e.path,
                "sha": e.sha,
                "content": base64.b64encode(e.content.encode()).decode(),
            }
            for e in edits
        ]
        body = {"branch": BASE_BRANCH, "new_branch": branch, "message": message, "files": files}
        self._call("POST", "/contents", body)

    def open_pull(self, branch: str, title: str, body: str) -> int:
        doc = self._call(
            "POST", "/pulls", {"head": branch, "base": BASE_BRANCH, "title": title, "body": body}
        )
        return int(doc["number"])

    def checks_state(self, sha: str) -> tuple[str, int]:
        """Combined commit status of `sha`: (state, number of statuses)."""
        doc = self._call("GET", f"/commits/{sha}/status")
        return str(doc.get("state") or ""), len(doc.get("statuses") or [])

    def merge(self, number: int) -> None:
        # Called only after the bot saw green checks (bot.Bot._merge); without branch
        # protection merge_when_checks_succeed would not wait anyway.
        body = {"Do": "squash", "delete_branch_after_merge": True}
        self._call("POST", f"/pulls/{number}/merge", body)
