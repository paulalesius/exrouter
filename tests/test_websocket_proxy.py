"""Tests for direct per-session WebSocket legs (replaces test_websocket_pool.py).

Design (nginx/Caddy model: one dedicated upstream leg per downstream client
connection, no pooling):
- Each concurrent client gets its own backend connection (no sharing,
  no cross-delivery).
- WebSocket sessions do NOT acquire or hold cross-backend locks for the
  session lifetime (a parked chat tab must not block the rest of the
  domain for hours), but they DO take the full activation path:
  activating a WebSocket backend deactivates the backends it locks
  (the same VRAM handoff as the HTTP path).
- When a backend is deactivated, its open legs are closed and the client
  sees a clean disconnect.
- When a session ends, the leg is closed and the self-lock released.

These tests drive `_handle_websocket` directly (same event loop, so state
can be asserted mid-session) with fake Starlette WebSocket objects, using
a real `websockets` echo server on loopback as the backend.
"""

import asyncio
import socket

import pytest
import websockets
from fastapi import Request
from starlette.websockets import WebSocketState

from exrouter.config import Config
from exrouter.proxy import LockProxy


DISCONNECT = object()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeClientWS:
    """Just enough of the Starlette WebSocket interface for _handle_websocket."""

    def __init__(self, tag: str):
        self.tag = tag
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.out: list[tuple[str, str | bytes]] = []
        self.accepted_subprotocol = None
        self.closed = False
        self.client_state = WebSocketState.CONNECTING
        self.headers = {"host": "127.0.0.1:4001"}

        class _Url:
            query = ""
            scheme = "ws"

        class _Client:
            host = "127.0.0.1"

        self.url = _Url()
        self.client = _Client()

    async def accept(self, subprotocol=None):
        self.accepted_subprotocol = subprotocol
        self.client_state = WebSocketState.CONNECTED

    async def close(self, code=1000, reason=None):
        self.closed = True
        self.client_state = WebSocketState.DISCONNECTED

    async def receive(self):
        msg = await self.inbox.get()
        if msg is DISCONNECT:
            return {"type": "websocket.disconnect", "code": 1000}
        return {"type": "websocket.receive", "text": msg}

    async def send_text(self, text):
        self.out.append(("text", text))

    async def send_bytes(self, data):
        self.out.append(("bytes", data))


async def wait_for(pred, timeout: float = 5.0):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


class EchoServer:
    """Real websockets echo server; records every connection and close."""

    def __init__(self):
        self.port = free_port()
        self.connections: list = []
        self.closed: list = []
        self._server = None

    async def handler(self, websocket):
        self.connections.append(websocket)
        try:
            async for msg in websocket:
                await websocket.send(msg)
        finally:
            self.closed.append(websocket)

    async def start(self):
        self._server = await websockets.serve(self.handler, "127.0.0.1", self.port)

    async def stop(self):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
async def echo_server():
    server = EchoServer()
    await server.start()
    yield server
    await server.stop()


def make_proxy(
    echo: EchoServer,
    extra_backends: dict | None = None,
    ws_locks: list[str] | None = None,
    ws_lifecycle: dict | None = None,
    timeout: int = 2,
) -> LockProxy:
    ws_backend: dict = {"url": echo.url, "paths": ["/ws"]}
    if ws_locks:
        ws_backend["locks"] = ws_locks
    if ws_lifecycle:
        ws_backend["lifecycle"] = ws_lifecycle
    backends = {"d": {"ws": ws_backend}}
    if extra_backends:
        backends["d"].update(extra_backends)
    return LockProxy(Config.from_dict({
        "backends": backends,
        "global_lock": {"timeout": timeout},
    }))


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


async def open_session(
    proxy: LockProxy, tag: str = "c", path: str = "ws"
) -> tuple[FakeClientWS, asyncio.Task]:
    """Start a WS session and wait until it echoes one message back."""
    fake = FakeClientWS(tag)
    task = asyncio.create_task(proxy._handle_websocket(fake, path))
    await fake.inbox.put(f"{tag}-ping")
    await wait_for(lambda: ("text", f"{tag}-ping") in fake.out)
    return fake, task


async def test_each_client_gets_dedicated_backend_leg(echo_server):
    """Concurrent sessions must get distinct backend connections (no
    pooling, no sharing, no cross-delivery between clients)."""
    proxy = make_proxy(echo_server)
    fakes = [FakeClientWS(f"c{i}") for i in range(3)]
    tasks = [asyncio.create_task(proxy._handle_websocket(f, "ws")) for f in fakes]

    for i, fake in enumerate(fakes):
        await fake.inbox.put(f"m{i}")
        await wait_for(lambda i=i: ("text", f"m{i}") in fake.out)

    # Three concurrent sessions => three distinct backend connections.
    assert len(echo_server.connections) == 3

    for fake in fakes:
        await fake.inbox.put(DISCONNECT)
    await asyncio.gather(*tasks)

    # All legs closed at the backend and deregistered; locks released.
    await wait_for(lambda: len(echo_server.closed) == 3)
    assert proxy.ws_legs.get("ws") in (None, set())
    assert proxy.active_counts["ws"] == 0
    assert proxy.domains["d"].lock_manager.is_locked("ws") is False


async def test_ws_session_skips_cross_locks_but_activates(echo_server, tmp_path):
    """A WS session must not take the backend's locks: targets, but must
    run the self-lock + activation path."""
    marker = tmp_path / "activated"
    proxy = make_proxy(
        echo_server,
        extra_backends={"other": {"url": "http://127.0.0.1:1", "paths": ["/other"]}},
        ws_locks=["other"],
        ws_lifecycle={"on_activate": {"shell": [f'echo on >> "{marker}"']}},
    )
    lm = proxy.domains["d"].lock_manager

    fake, task = await open_session(proxy)

    # Cross-backend lock NOT held for the session (the pool code held it
    # for the entire connection lifetime).
    assert lm.is_locked("other") is False
    # Self-lock held, backend marked active, on_activate ran.
    assert lm.is_locked("ws") is True
    assert proxy.active_counts["ws"] == 1
    assert "ws" in proxy.activated_backends
    assert marker.exists()

    # Session ends: leg closed, self-lock released, count back to zero.
    await fake.inbox.put(DISCONNECT)
    await task
    await wait_for(lambda: len(echo_server.closed) == 1)
    assert proxy.ws_legs.get("ws") in (None, set())
    assert proxy.active_counts["ws"] == 0
    assert lm.is_locked("ws") is False


async def test_ws_self_lock_blocks_locking_http_backend(echo_server, tmp_path):
    """While a WS session is open, the backend is in use (self-lock held),
    so an HTTP backend that locks it must 503. After the session ends it
    can proceed, and its activation deactivates the WS backend."""
    deact_marker = tmp_path / "deactivated"
    proxy = make_proxy(
        echo_server,
        extra_backends={
            "heavy": {
                "url": "http://127.0.0.1:1",
                "paths": ["/heavy"],
                "locks": ["ws"],
            }
        },
        ws_lifecycle={"on_deactivate": {"shell": [f'echo off >> "{deact_marker}"']}},
        timeout=1,
    )

    fake, task = await open_session(proxy)

    # Session open: "heavy" locks "ws" and must time out => 503.
    blocked = await proxy._handle_request(make_request("GET", "/heavy"), "heavy")
    assert blocked.status_code == 503

    # Session ends: the self-lock is released, "heavy" can now proceed.
    # Its activation deactivates "ws" (marker written, activation cleared).
    # The 502 is expected: "heavy"'s URL is a dead port.
    await fake.inbox.put(DISCONNECT)
    await task

    ok = await proxy._handle_request(make_request("GET", "/heavy"), "heavy")
    assert ok.status_code == 502
    assert deact_marker.exists()
    assert "ws" not in proxy.activated_backends


async def test_ws_activation_deactivates_locked_backend(echo_server, tmp_path):
    """Opening a WS session must run the same VRAM handoff as the HTTP
    path: the backends the session's backend locks are deactivated
    (service stopped, dropped from activated_backends, open legs closed)
    before the session's own backend is activated. Without this, the
    session starts its service while a locked backend's service is still
    running: both mutually exclusive services up at once."""
    handoff = tmp_path / "handoff"
    proxy = make_proxy(
        echo_server,
        extra_backends={
            "other": {
                "url": echo_server.url,
                "paths": ["/other"],
                "lifecycle": {
                    "on_deactivate": {"shell": [f'echo other-off >> "{handoff}"']}
                },
            }
        },
        ws_locks=["other"],
        ws_lifecycle={"on_activate": {"shell": [f'echo ws-on >> "{handoff}"']}},
    )
    lm = proxy.domains["d"].lock_manager

    # "other" is up and in use: hold a session to it.
    other_fake, other_task = await open_session(proxy, "o", path="other")
    assert lm.is_locked("other") is True
    assert "other" in proxy.activated_backends

    # A WS session to the backend that locks "other" must stop it first.
    fake, task = await open_session(proxy, "c")

    assert handoff.read_text().split() == ["other-off", "ws-on"]
    assert "other" not in proxy.activated_backends

    # "other"'s open leg was closed: its session's backend side ends and
    # the client is told to disconnect.
    await wait_for(lambda: other_fake.closed)
    await other_fake.inbox.put(DISCONNECT)  # unblock the client-side loop
    await other_task
    await wait_for(lambda: lm.is_locked("other") is False)

    # Session ends: leg closed, self-lock released.
    await fake.inbox.put(DISCONNECT)
    await task
    await wait_for(lambda: len(echo_server.closed) == 2)
    assert proxy.active_counts["ws"] == 0
    assert lm.is_locked("ws") is False


async def test_deactivation_closes_open_ws_legs(echo_server):
    """_close_backend_ws_legs (called from the deactivation path) must close
    live legs so the client sees a clean disconnect and the registry drains."""
    proxy = make_proxy(echo_server)
    fake, task = await open_session(proxy)
    assert len(proxy.ws_legs["ws"]) == 1

    await proxy._close_backend_ws_legs("ws")

    await wait_for(lambda: fake.closed)
    assert proxy.ws_legs.get("ws") in (None, set())
    await wait_for(lambda: len(echo_server.closed) == 1)

    await fake.inbox.put(DISCONNECT)
    await task
    assert proxy.active_counts["ws"] == 0
    assert proxy.domains["d"].lock_manager.is_locked("ws") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
