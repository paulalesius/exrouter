"""Test that Python scripts are a first-class lifecycle action type.

Per the design, activation and deactivation are a single mechanism
configured in the YAML lifecycle: block; per phase the user picks
systemd, shell, or Python. This test drives the Python path end to end:
a full llm -> stt -> llm round trip where both backends' start/stop
logic lives in Python scripts (one async, one sync, so both forms are
exercised), plus robustness checks that a failing or malformed script
never wedges the proxy (consistent with shell/systemd error handling).
"""

import httpx
import pytest
from fastapi import Request

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
    if not path.exists():
        return []
    return path.read_text().split()


LLM_SCRIPT = """
MARKERS = {markers!r}

async def activate():
    with open(MARKERS, "a") as f:
        f.write("llm-start\\n")

def deactivate():
    with open(MARKERS, "a") as f:
        f.write("llm-stop\\n")
"""

STT_SCRIPT = """
MARKERS = {markers!r}

def activate():
    with open(MARKERS, "a") as f:
        f.write("stt-start\\n")

async def deactivate():
    with open(MARKERS, "a") as f:
        f.write("stt-stop\\n")
"""


def build_proxy(tmp_path, markers, llm_lifecycle, stt_lifecycle):
    config = Config.from_dict({
        "server": {"host": "127.0.0.1", "port": 9999},
        "backends": {
            "compute": {
                "llm": {
                    "url": "http://backend-llm:8080",
                    "paths": ["/v1/chat/completions"],
                    "locks": ["stt"],
                    "lifecycle": llm_lifecycle,
                },
                "stt": {
                    "url": "http://backend-stt:8084",
                    "paths": ["/transcribe"],
                    "locks": ["llm"],
                    "lifecycle": stt_lifecycle,
                },
            }
        },
        "global_lock": {"enabled": True, "timeout": 5},
    })
    p = LockProxy(config)
    p.httpx_client = FakeHttpClient()
    p.markers = markers
    return p


@pytest.fixture
def proxy(tmp_path):
    markers = tmp_path / "lifecycle.log"
    llm_script = tmp_path / "llm.py"
    stt_script = tmp_path / "stt.py"
    llm_script.write_text(LLM_SCRIPT.format(markers=str(markers)))
    stt_script.write_text(STT_SCRIPT.format(markers=str(markers)))
    return build_proxy(
        tmp_path, markers,
        {"on_activate": {"python": str(llm_script)}, "on_deactivate": {"python": str(llm_script)}},
        {"on_activate": {"python": str(stt_script)}, "on_deactivate": {"python": str(stt_script)}},
    )


async def test_python_lifecycle_roundtrip(proxy):
    """llm -> stt -> llm with Python scripts: both directions work.

    llm.py has an async activate() and a sync deactivate(); stt.py is
    the reverse, so both sync and async callables are exercised. Each
    phase points at the same file, proving one script can serve both
    phases of a backend.
    """
    r1 = await proxy._handle_request(
        make_request("POST", "/v1/chat/completions"), "v1/chat/completions"
    )
    assert r1.status_code == 200
    assert read_markers(proxy.markers) == ["stt-stop", "llm-start"]

    r2 = await proxy._handle_request(make_request("POST", "/transcribe"), "transcribe")
    assert r2.status_code == 200
    assert read_markers(proxy.markers) == [
        "stt-stop", "llm-start", "llm-stop", "stt-start",
    ]

    r3 = await proxy._handle_request(
        make_request("POST", "/v1/chat/completions"), "v1/chat/completions"
    )
    assert r3.status_code == 200
    assert read_markers(proxy.markers) == [
        "stt-stop", "llm-start", "llm-stop", "stt-start", "stt-stop", "llm-start",
    ]
    assert proxy.activated_backends == {"llm"}


async def test_failing_python_script_does_not_break_proxy(tmp_path):
    """A raising lifecycle script is logged, not fatal: consistent with
    the shell/systemd action error handling."""
    markers = tmp_path / "lifecycle.log"
    bad = tmp_path / "bad.py"
    bad.write_text("def activate():\n    raise RuntimeError('boom')\n")
    p = build_proxy(tmp_path, markers, {"on_activate": {"python": str(bad)}}, {})

    r = await p._handle_request(
        make_request("POST", "/v1/chat/completions"), "v1/chat/completions"
    )
    assert r.status_code == 200
    assert read_markers(markers) == []


async def test_python_script_missing_callable_does_not_break_proxy(tmp_path):
    """A script that does not define the phase callable is logged, not fatal."""
    markers = tmp_path / "lifecycle.log"
    bad = tmp_path / "bad.py"
    bad.write_text("# no activate() here\n")
    p = build_proxy(tmp_path, markers, {"on_activate": {"python": str(bad)}}, {})

    r = await p._handle_request(
        make_request("POST", "/v1/chat/completions"), "v1/chat/completions"
    )
    assert r.status_code == 200
    assert read_markers(markers) == []
