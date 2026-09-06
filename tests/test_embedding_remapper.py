"""Regression tests for the TEI embedding remapper sample.

The old remapper opened its own httpx connection to the embedding GPU
and short-circuited the response. remap() runs before the proxy
acquires locks, so an in-flight embedding call held no locks and the
other GPU backends could run at the same time.

The remapper now only translates: the request phase rewrites the
request (TEI "inputs" -> OpenAI "input", path -> /v1/embeddings) and
the proxy forwards it under the backend's locks; the response phase
unwraps the OpenAI answer back into a bare TEI vector list when the
client asked for TEI format.

These tests cover the translation in isolation and, end to end, that
the GPU call now runs with the backend's locks held: a concurrent LLM
request gets 503 while an embedding is in flight and succeeds again
after it completes.
"""

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from fastapi import Request
from fastapi.responses import Response

from exrouter.config import Config
from exrouter.hooks import HookContext
from exrouter.proxy import LockProxy

SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "llama-server-embedding-tei-remapper.py"
)

OPENAI_EMBED_BODY = (
    b'{"data": [{"embedding": [0.1, 0.2], "index": 0},'
    b' {"embedding": [0.3, 0.4], "index": 1}]}'
)


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


def make_context(path: str, body: bytes, *, headers=None, response_status=None,
                 response_body=None) -> HookContext:
    return HookContext(
        backend_name="embed",
        request_method="POST",
        request_path=path,
        request_headers={"host": "127.0.0.1:4001", **(headers or {})},
        request_body=body,
        response_status=response_status,
        response_body=response_body,
    )


async def wait_for(pred, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() >= deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


@pytest.fixture
def remapper():
    spec = importlib.util.spec_from_file_location("sample_tei_embed_remap", SAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RequestRemapper()


# --- Request phase: translation to OpenAI, no short-circuit -------------


async def test_tei_request_is_rewritten_to_openai(remapper):
    result = await remapper.remap(
        make_context("/v1/embed", b'{"inputs": ["hello", "world"]}')
    )
    assert result is not None
    # Must NOT short-circuit: the proxy has to forward this under lock.
    assert result.status_code is None
    assert result.path == "/v1/embeddings"
    assert json.loads(result.body) == {"input": ["hello", "world"]}
    assert result.headers.get("x-exrouter-embed-format") == "tei"


async def test_openai_request_passes_through_without_marker(remapper):
    result = await remapper.remap(make_context("/v1/embeddings", b'{"input": "hi"}'))
    assert result is not None
    assert result.status_code is None
    assert result.path == "/v1/embeddings"
    assert json.loads(result.body) == {"input": "hi"}
    assert not result.headers


async def test_openai_path_with_tei_body_gets_marker(remapper):
    result = await remapper.remap(make_context("/embeddings", b'{"inputs": "hi"}'))
    assert json.loads(result.body) == {"input": "hi"}
    assert result.headers.get("x-exrouter-embed-format") == "tei"


async def test_empty_body_is_rejected(remapper):
    result = await remapper.remap(make_context("/v1/embed", b""))
    assert result.status_code == 400


async def test_invalid_json_body_is_rejected(remapper):
    result = await remapper.remap(make_context("/v1/embed", b"not json"))
    assert result.status_code == 400


async def test_info_and_models_still_short_circuit(remapper):
    info = await remapper.remap(make_context("/v1/info", b""))
    assert info.status_code == 200
    assert json.loads(info.content)["model_type"] == "text-embeddings"

    models = await remapper.remap(make_context("/v1/models", b""))
    assert models.status_code == 200
    assert json.loads(models.content)["data"][0]["id"]


# --- Response phase: unwrap OpenAI answer into bare TEI list -------------


async def test_response_unwraps_to_bare_tei_list(remapper):
    context = make_context(
        "/v1/embeddings",
        b'{"input": ["a", "b"]}',
        headers={"x-exrouter-embed-format": "tei"},
        response_status=200,
        response_body=OPENAI_EMBED_BODY,
    )
    result = await remapper.remap(context)
    assert result is not None
    assert result.status_code == 200
    assert json.loads(result.content) == [[0.1, 0.2], [0.3, 0.4]]
    assert result.response_headers == {"content-type": "application/json"}


async def test_response_passes_through_for_openai_clients(remapper):
    context = make_context(
        "/v1/embeddings",
        b'{"input": "hi"}',
        response_status=200,
        response_body=OPENAI_EMBED_BODY,
    )
    assert await remapper.remap(context) is None


async def test_response_passes_through_on_backend_error(remapper):
    context = make_context(
        "/v1/embeddings",
        b'{"input": "hi"}',
        headers={"x-exrouter-embed-format": "tei"},
        response_status=500,
        response_body=b"internal error",
    )
    assert await remapper.remap(context) is None


# --- End to end: the GPU call runs under the backend's locks -------------


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


class GatedBufferedResponse:
    """A buffered response that pauses inside aread() (while the proxy
    holds the locks) until the gate is set, recording the lock state at
    that moment."""

    def __init__(self, body: bytes, gate: asyncio.Event, snapshot):
        self._body = body
        self._gate = gate
        self._snapshot = snapshot
        self.status_code = 200
        self.headers = httpx.Headers({"content-type": "application/json"})
        self.closed = False
        self.lock_state_at_read = None

    async def aread(self):
        self.lock_state_at_read = self._snapshot()
        await self._gate.wait()
        return self._body

    def close(self):
        self.closed = True


class EmbedFakeClient:
    """Stand-in for the proxy's httpx.AsyncClient. The rewritten embed
    request reaches the (gated) embed response; llm requests get a plain
    buffered response."""

    def __init__(self, embed_response: GatedBufferedResponse):
        self.embed_response = embed_response
        self.requests: list[tuple[str, str, dict, bytes]] = []

    def build_request(self, method, url, headers=None, content=None):
        self.requests.append((method, str(url), dict(headers or {}), content))
        return (str(url), content, headers)

    async def send(self, req, stream=False):
        url, _, _ = req
        if "backend-embed" in url:
            return self.embed_response
        return FakeBufferedResponse(b'{"choices": []}')


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
                    "paths": [
                        "/v1/embed", "/embed",
                        "/v1/embeddings", "/embeddings",
                        "/v1/info", "/v1/models",
                    ],
                    "locks": ["llm"],  # mutual exclusion, as in real configs
                },
            }
        },
        "global_lock": {"enabled": True, "timeout": 1},
    })
    p = LockProxy(config)
    p.backends["embed"].remapper = str(SAMPLE)
    assert p.remapper_loader.load_script("embed", str(SAMPLE)) is not None
    gate = asyncio.Event()

    def snapshot():
        lm = p.domains["compute"].lock_manager
        return (lm.is_locked("embed"), lm.is_locked("llm"))

    p.gate = gate
    p.httpx_client = EmbedFakeClient(
        GatedBufferedResponse(OPENAI_EMBED_BODY, gate, snapshot)
    )
    return p


async def test_embedding_gpu_call_runs_under_locks(proxy):
    """An in-flight embedding must hold the backend's locks: the proxy
    forwards the rewritten request (the remapper itself never calls the
    GPU), a concurrent LLM request gets 503, the client still receives
    the bare TEI vector list, and the locks are released afterwards."""
    lm = proxy.domains["compute"].lock_manager
    lm.timeout = 0.2  # keep the 503 wait short

    request = make_request("POST", "/v1/embed", b'{"inputs": ["a", "b"]}')
    task = asyncio.create_task(proxy._handle_request(request, "v1/embed"))

    # Wait until the backend read is in flight (paused on the gate).
    await wait_for(
        lambda: proxy.httpx_client.embed_response.lock_state_at_read is not None
    )

    # The proxy forwarded the rewritten request under the backend URL.
    method, url, headers, content = proxy.httpx_client.requests[0]
    assert method == "POST"
    assert url.endswith("/v1/embeddings")
    assert headers.get("x-exrouter-embed-format") == "tei"
    assert json.loads(content) == {"input": ["a", "b"]}

    # While the embedding is in flight, both locks are held.
    assert proxy.httpx_client.embed_response.lock_state_at_read == (True, True)

    # A concurrent LLM request must be rejected (GPU exclusivity).
    blocked = await proxy._handle_request(
        make_request("POST", "/v1/chat/completions"), "v1/chat/completions"
    )
    assert isinstance(blocked, Response)
    assert blocked.status_code == 503

    # Finish the embedding: the client gets the bare TEI vector list.
    proxy.gate.set()
    response = await task
    assert isinstance(response, Response)
    assert response.status_code == 200
    assert json.loads(response.body) == [[0.1, 0.2], [0.3, 0.4]]

    # Everything is released; the GPU is free for the LLM again.
    assert lm.is_locked("embed") is False
    assert lm.is_locked("llm") is False
    assert proxy.active_counts["embed"] == 0

    after = await proxy._handle_request(
        make_request("POST", "/v1/chat/completions"), "v1/chat/completions"
    )
    assert after.status_code == 200


async def test_openai_embedding_request_roundtrip(proxy):
    """An OpenAI-format client also goes through the proxy under lock and
    gets the untouched OpenAI answer back (no marker, no unwrap)."""
    lm = proxy.domains["compute"].lock_manager
    lm.timeout = 0.2

    request = make_request("POST", "/v1/embeddings", b'{"input": "hi"}')
    task = asyncio.create_task(
        proxy._handle_request(request, "v1/embeddings")
    )
    await wait_for(
        lambda: proxy.httpx_client.embed_response.lock_state_at_read is not None
    )
    assert proxy.httpx_client.embed_response.lock_state_at_read == (True, True)

    blocked = await proxy._handle_request(
        make_request("POST", "/v1/chat/completions"), "v1/chat/completions"
    )
    assert blocked.status_code == 503

    proxy.gate.set()
    response = await task
    assert response.status_code == 200
    assert json.loads(response.body) == json.loads(OPENAI_EMBED_BODY)

    assert lm.is_locked("embed") is False
    assert lm.is_locked("llm") is False
