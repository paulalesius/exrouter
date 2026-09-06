"""Test that locks are held for the ENTIRE duration of a request.

The README promises: "Locks are held for the *entire duration* of the
request (including full streaming responses). Locks are only released
after the request (including streaming) has completed."

These tests drive _handle_request directly (same event loop) so lock state
can be asserted while an SSE stream is mid-flight, and verify:

1. The lock on a locked backend stays held until the SSE stream is
   fully delivered, then is released.
2. The lock is also released if the client disconnects mid-stream
   (the body iterator is closed early).
3. Buffered (non-streaming) responses release their locks immediately
   after the response is produced (regression guard).
4. While a backend's stream is in flight, a request to the backend it
   locks gets a 503 (exclusivity), and succeeds after the stream ends.
"""

import asyncio

import httpx
import pytest
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from exrouter.config import Config
from exrouter.proxy import LockProxy


SSE_LINES = [
    'data: {"choices": [{"delta": {"content": "hel"}}]}',
    "",
    'data: {"choices": [{"delta": {"content": "lo"}}]}',
    "",
    "data: [DONE]",
    "",
]


class FakeSSEResponse:
    """Mimics an httpx streaming response with content-type text/event-stream.

    After yielding the first line, the stream pauses on `gate` so the test
    can assert lock state deterministically while the stream is mid-flight.
    """

    def __init__(self, gate: asyncio.Event):
        self._gate = gate
        self.status_code = 200
        self.headers = httpx.Headers({"content-type": "text/event-stream"})
        self.closed = False

    async def aiter_lines(self):
        for i, line in enumerate(SSE_LINES):
            yield line
            if i == 0:
                await self._gate.wait()

    async def aread(self):
        raise AssertionError("aread() called on an SSE response")

    def close(self):
        self.closed = True


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
    """Stand-in for the proxy's httpx.AsyncClient.

    Routes by URL substring: "llm" URLs get an SSE response (gated),
    "embed" URLs get a buffered JSON response.
    """

    def __init__(self, gate: asyncio.Event):
        self._gate = gate
        self.sse_response = FakeSSEResponse(gate)
        self.embed_response = FakeBufferedResponse(b'{"embeddings": [[0.1, 0.2]]}')
        self.requests: list[tuple[str, str]] = []

    def build_request(self, method, url, headers=None, content=None):
        self.requests.append((method, str(url)))
        return ("req", str(url), method, headers, content)

    async def send(self, req, stream=False):
        _, url, _, _, _ = req
        if "llm" in url:
            return self.sse_response
        return self.embed_response


@pytest.fixture
def proxy():
    config = Config.from_dict({
        "server": {"host": "127.0.0.1", "port": 9999},
        "backends": {
            "compute": {
                "llm": {
                    "url": "http://backend-llm:8080",
                    "paths": ["/v1/chat/completions"],
                    "locks": ["embed"],
                },
                "embed": {
                    "url": "http://backend-embed:8082",
                    "paths": ["/v1/embeddings"],
                    "locks": ["llm"],  # mutual exclusion, as in real configs
                },
            }
        },
        "global_lock": {"enabled": True, "timeout": 1},
    })
    p = LockProxy(config)
    gate = asyncio.Event()
    p.httpx_client = FakeHttpClient(gate)
    p.gate = gate  # test handle to resume the in-flight stream
    return p


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


async def _start_sse_request(proxy):
    """Start an SSE request to llm and return its (gated, mid-flight) body
    iterator plus the already-consumed first chunk."""
    request = make_request("POST", "/v1/chat/completions", b'{"stream": true}')
    response = await proxy._handle_request(request, "v1/chat/completions")
    assert isinstance(response, StreamingResponse)
    it = response.body_iterator
    first = await it.__anext__()
    assert "hel" in first  # we have the first SSE event; stream is now paused on the gate
    return it, first


async def test_lock_held_during_sse_stream_and_released_after(proxy):
    """Locks must be held while the stream is in flight, released after completion."""
    lm = proxy.domains["compute"].lock_manager
    it, first = await _start_sse_request(proxy)

    # Stream is mid-flight (paused on the gate): locks must still be held.
    assert lm.is_locked("embed") is True, "lock on locked backend must survive the stream"
    assert lm.is_locked("llm") is True, "self-lock must survive the stream"
    assert proxy.active_counts["llm"] == 1

    # Let the in-flight stream resume and finish.
    proxy.gate.set()
    chunks = []
    async for chunk in it:
        chunks.append(chunk)
    stream = first + "".join(chunks)
    assert "hel" in stream and "lo" in stream and "[DONE]" in stream

    # Stream complete: everything released, backend response closed.
    assert lm.is_locked("embed") is False
    assert lm.is_locked("llm") is False
    assert proxy.active_counts["llm"] == 0
    assert proxy.httpx_client.sse_response.closed is True


async def test_lock_released_when_client_disconnects_mid_stream(proxy):
    """If the client goes away, closing the body iterator must release the locks."""
    lm = proxy.domains["compute"].lock_manager
    it, _ = await _start_sse_request(proxy)

    assert lm.is_locked("embed") is True

    # Simulate client disconnect: the ASGI server stops consuming the body.
    await it.aclose()

    assert lm.is_locked("embed") is False
    assert lm.is_locked("llm") is False
    assert proxy.active_counts["llm"] == 0


async def test_lock_released_after_buffered_response(proxy):
    """Regression guard: non-streaming responses keep the old (immediate) release."""
    # Point the llm URL at the buffered fake instead of the SSE one.
    proxy.httpx_client.sse_response = FakeBufferedResponse(b'{"choices": []}')

    request = make_request("POST", "/v1/chat/completions", b'{"stream": false}')
    response = await proxy._handle_request(request, "v1/chat/completions")
    assert isinstance(response, Response)
    assert response.status_code == 200
    assert response.body == b'{"choices": []}'

    lm = proxy.domains["compute"].lock_manager
    assert lm.is_locked("embed") is False
    assert lm.is_locked("llm") is False
    assert proxy.active_counts["llm"] == 0


async def test_locked_backend_gets_503_while_stream_in_flight(proxy):
    """Exclusivity guarantee: a request to the locked backend must fail while
    the locking backend's stream is still in flight, and succeed after."""
    proxy.domains["compute"].lock_manager.timeout = 0.2  # keep the 503 wait short

    it, _ = await _start_sse_request(proxy)

    # Request to 'embed' while llm's stream holds the lock on it.
    embed_request = make_request("POST", "/v1/embeddings", b'{"input": "hi"}')
    blocked = await proxy._handle_request(embed_request, "v1/embeddings")
    assert isinstance(blocked, Response)
    assert blocked.status_code == 503
    assert blocked.headers.get("retry-after") == "10"

    # Finish the stream, then the same request must succeed.
    proxy.gate.set()
    async for _ in it:
        pass

    embed_request = make_request("POST", "/v1/embeddings", b'{"input": "hi"}')
    ok = await proxy._handle_request(embed_request, "v1/embeddings")
    assert isinstance(ok, Response)
    assert ok.status_code == 200
    assert ok.body == b'{"embeddings": [[0.1, 0.2]]}'
