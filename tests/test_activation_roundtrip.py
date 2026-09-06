"""Test that the activation -> deactivation handoff works in BOTH directions.

A backend whose service was stopped by another backend's activation (its
on_deactivate ran) must be re-activated when the next request arrives for
it: its on_activate must run again, and its own locked backends must be
stopped first.

Before the fix, activated_backends was a one-way latch (only ever added
to), so a deactivated backend could never be started again - the VRAM
handoff only worked one direction, and the first request back to a
stopped backend hit a dead service.

The test drives _handle_request directly (same event loop, same pattern
as test_lock_lifecycle.py) and uses declarative shell lifecycle actions
that append markers to a file, so the exact start/stop sequence is
assertable without systemd.
"""

import httpx
import pytest
from fastapi import Request
from fastapi.responses import Response

from exrouter.config import Config
from exrouter.proxy import LockProxy


class FakeBufferedResponse:
    """Mimics an httpx non-streaming JSON response."""

    def __init__(self, body: bytes):
        self._body = body
        self.status_code = 200
        self.headers = httpx.Headers({"content-type": "application/json"})
        self.closed = False

    async def aread(self):
        return self._body

    def close(self):
        self.closed = True


class FakeHttpClient:
    """Stand-in for the proxy's httpx.AsyncClient; every request succeeds."""

    def __init__(self):
        self.requests: list[tuple[str, str]] = []

    def build_request(self, method, url, headers=None, content=None):
        self.requests.append((method, str(url)))
        return ("req", str(url), method, headers, content)

    async def send(self, req, stream=False):
        return FakeBufferedResponse(b'{"ok": true}')


def make_request(method: str, path: str, body: bytes = b"{}") -> Request:
    """Build a Starlette Request for direct _handle_request calls."""

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1:4001")],
        "client": ("127.0.0.1", 51000),
    }
    return Request(scope, receive)


def read_markers(path) -> list[str]:
    return path.read_text().split()


@pytest.fixture
def proxy(tmp_path):
    markers = tmp_path / "lifecycle.log"
    config = Config.from_dict({
        "server": {"host": "127.0.0.1", "port": 9999},
        "backends": {
            "compute": {
                "llm": {
                    "url": "http://backend-llm:8080",
                    "paths": ["/v1/chat/completions"],
                    "locks": ["stt"],
                    "lifecycle": {
                        "on_activate": {"shell": [f'echo llm-start >> "{markers}"']},
                        "on_deactivate": {"shell": [f'echo llm-stop >> "{markers}"']},
                    },
                },
                "stt": {
                    "url": "http://backend-stt:8084",
                    "paths": ["/transcribe"],
                    "locks": ["llm"],
                    "lifecycle": {
                        "on_activate": {"shell": [f'echo stt-start >> "{markers}"']},
                        "on_deactivate": {"shell": [f'echo stt-stop >> "{markers}"']},
                    },
                },
            }
        },
        "global_lock": {"enabled": True, "timeout": 5},
    })
    p = LockProxy(config)
    p.httpx_client = FakeHttpClient()
    p.markers = markers
    return p


async def test_deactivated_backend_is_reactivated_on_next_request(proxy):
    """llm -> stt -> llm must start/stop services in both directions."""
    # 1. First request to llm: llm activates (and first stops its lock
    #    target stt, which was never started - a no-op in real deployments).
    r1 = await proxy._handle_request(
        make_request("POST", "/v1/chat/completions"), "v1/chat/completions"
    )
    assert r1.status_code == 200
    assert read_markers(proxy.markers) == ["stt-stop", "llm-start"]

    # 2. Request to stt: stt activates, which stops llm's service
    #    (llm-stop) and starts stt's own (stt-start).
    r2 = await proxy._handle_request(make_request("POST", "/transcribe"), "transcribe")
    assert r2.status_code == 200
    assert read_markers(proxy.markers) == [
        "stt-stop", "llm-start", "llm-stop", "stt-start",
    ]
    assert "llm" not in proxy.activated_backends

    # 3. Request back to llm: llm must be RE-activated - stop stt
    #    (stt-stop) and run llm's on_activate again (llm-start). Before the
    #    fix this step did nothing: llm stayed in activated_backends
    #    forever, so its (stopped) service was never restarted.
    r3 = await proxy._handle_request(
        make_request("POST", "/v1/chat/completions"), "v1/chat/completions"
    )
    assert r3.status_code == 200
    assert read_markers(proxy.markers) == [
        "stt-stop", "llm-start", "llm-stop", "stt-start", "stt-stop", "llm-start",
    ]
    assert proxy.activated_backends == {"llm"}
