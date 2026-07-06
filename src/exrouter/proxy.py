"""EXRouter - routes requests to backends with global locking and optional request remapping.

Architecture:
- LockDomain: Top-level domain containing backends (e.g. "compute", "frontend")
- Backend: Individual backend within a domain, can only lock other backends in same domain
- LockProxy: Main proxy that routes requests and manages multiple domains
"""

import asyncio
import json
import logging
import time
import re
from urllib.parse import urlparse
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import StreamingResponse
import httpx

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.websockets import WebSocketState

from .backend import Backend
from .config import Config, LifecycleConfig
from .domain import LockDomain
from .hooks import HookLoader, HookContext
from .lifecycle import LifecycleExecutor
from .remapper import RemapperLoader, RemapResult

logger = logging.getLogger("exrouter")


def strip_thinking_at_start(text: str) -> str:
    """Strip thinking blocks only at the START of text.
    
    Handles both formats:
    - <think>...</think>
    - <think>...</think>
    - <reasoning>...</reasoning>
    
    Only removes if the tag is at position 0 (after optional whitespace).
    """
    patterns = [
        r'^\s*<think>(.*?)\s*</think>\s*(.*)',
        r'^\s*<think\b[^>]*>(.*?)\s*</think>\s*(.*)',
        r'^\s*<reasoning\b[^>]*>(.*?)\s*</reasoning>\s*(.*)',
        r'^\s*</?think\s*>(.*?)\s*</?think\s*>\s*(.*)',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(2).strip()
    
    return text


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all incoming HTTP requests and their final responses."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()
        client_host = request.client.host if request.client else "-"
        forwarded = request.headers.get("x-forwarded-for", client_host)

        logger.info(f"→ {request.method} {request.url.path} from {forwarded}")

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"← {request.method} {request.url.path} "
            f"status={response.status_code} ({process_time:.0f}ms)"
        )

        return response


class LockProxy:
    """Main proxy server with connection pooling, locking, hooks, and request remapping.
    
    Architecture:
    - Multiple LockDomains, each with its own LockManager
    - Backends can only lock other backends in the same domain
    - Cross-domain requests don't cause deadlocks
    """

    def __init__(self, config: Config):
        self.config = config

        # Build hierarchical domain structure
        # domains: domain_name -> LockDomain
        self.domains: dict[str, LockDomain] = {}
        
        # Flat lookup for routing: backend_name -> Backend
        # (backend names must be globally unique across all domains)
        self.backends: dict[str, Backend] = {}
        
        # Track which domain each backend belongs to
        self.backend_to_domain: dict[str, str] = {}
        
        # Lifecycle configs per backend
        self.lifecycle_configs: dict[str, Optional[LifecycleConfig]] = {}

        for domain_name, domain_config in config.backends.items():
            # Build Backend instances for this domain
            domain_backends: dict[str, Backend] = {}
            
            for backend_name, backend_config in domain_config.backends.items():
                backend = Backend(
                    name=backend_name,
                    domain_name=domain_name,  # ← track domain membership
                    url=str(backend_config.url),
                    paths=backend_config.paths,
                    locks=backend_config.locks,
                    domain=backend_config.domain,
                    script=backend_config.script,
                    remapper=backend_config.remapper,
                )
                domain_backends[backend_name] = backend
                self.backends[backend_name] = backend
                self.backend_to_domain[backend_name] = domain_name
                self.lifecycle_configs[backend_name] = backend_config.lifecycle
            
            # Create LockDomain with its own LockManager
            self.domains[domain_name] = LockDomain(
                name=domain_name,
                backends=domain_backends,
                timeout=config.global_lock.timeout
            )

        # Initialize hook loader
        self.hook_loader = HookLoader()
        for backend in self.backends.values():
            if backend.script:
                self.hook_loader.load_script(backend.name, backend.script)

        # Initialize remapper loader
        self.remapper_loader = RemapperLoader()
        for backend in self.backends.values():
            if backend.remapper:
                self.remapper_loader.load_script(backend.name, backend.remapper)

        # Initialize declarative lifecycle executor
        self.lifecycle_executor = LifecycleExecutor()

        # Track in-flight requests per backend
        self.active_counts: dict[str, int] = {name: 0 for name in self.backends}

        # Clean activation tracking
        self.activated_backends: set[str] = set()

        # Shared httpx client
        self.httpx_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            timeout=httpx.Timeout(300.0, connect=30.0)
        )

        # Create FastAPI app
        self.app = FastAPI(title="EXRouter")
        self.app.add_middleware(RequestLoggingMiddleware)
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

        @self.app.get("/config")
        async def get_config() -> dict:
            """Return full configuration as JSON.
            
            This endpoint exposes the complete config in JSON format,
            useful for debugging and monitoring.
            """
            return self.config.model_dump()

        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy_request(request: Request, path: str):
            return await self._handle_request(request, path)

        @self.app.websocket("/{path:path}")
        async def proxy_websocket(websocket: WebSocket, path: str):
            await self._handle_websocket(websocket, path)

    def _find_backend(self, host: str, path: str) -> Optional[Backend]:
        """Robust domain + path matching.
        
        Searches all backends across all domains. Returns the first match.
        """
        import fnmatch

        raw_host = host or ""
        host_no_port = raw_host.lower().split(":")[0]
        full_path = f"/{path}" if not path.startswith("/") else path

        domain_matched = []
        path_only = []

        for backend in self.backends.values():
            domains = backend.domain
            if isinstance(domains, str):
                domains = [domains]
            if not domains:
                domains = []

            matched_domain = False
            for d in domains:
                d_clean = str(d).lower().split(":")[0]
                if fnmatch.fnmatch(host_no_port, d_clean):
                    domain_matched.append(backend)
                    matched_domain = True
                    break

            if not matched_domain:
                path_only.append(backend)

        candidates = domain_matched if domain_matched else path_only

        for backend in candidates:
            if backend.matches_path(full_path):
                return backend

        logger.warning(
            f"No backend matched host+path for Host={raw_host} path={full_path}"
        )
        return None

    async def _handle_request(self, request: Request, path: str) -> Response:
        full_path = f"/{path}"
        start_time = time.time()

        # 1. Find matching backend by domain + path
        host = request.headers.get("host", "")
        backend: Optional[Backend] = self._find_backend(host, full_path)

        if not backend:
            logger.warning(
                f"No backend matched host+path for Host={request.headers.get('host', '-')} path={full_path}"
            )
            return Response(status_code=404, content=f"Unknown path: {full_path}")

        # 2. Run request remapper (if configured) BEFORE acquiring locks
        request_body = await request.body()
        hook_context = HookContext(
            backend_name=backend.name,
            request_method=request.method,
            request_path=full_path,
            request_headers=dict(request.headers),
            request_body=request_body
        )

        remapped = False
        if backend.remapper:
            remapper_instance = self.remapper_loader.get_remapper(backend.name)
            if remapper_instance:
                result: Optional[RemapResult] = await self.remapper_loader.call_remap(
                    remapper_instance, hook_context
                )
                if result:
                    remapped = True
                    
                    if result.status_code is not None:
                        content = result.content
                        if isinstance(content, str):
                            content = content.encode("utf-8")
                        return Response(
                            status_code=result.status_code,
                            content=content or b"",
                            headers=result.response_headers or {}
                        )

                    if result.backend and result.backend in self.backends:
                        backend = self.backends[result.backend]
                        hook_context.backend_name = backend.name
                        logger.info(f"Remapped request to backend '{backend.name}'")

                    if result.path is not None:
                        full_path = result.path
                        hook_context.request_path = full_path

                    if result.headers:
                        hook_context.request_headers.update(result.headers)
                    if result.body is not None:
                        hook_context.request_body = result.body
                        request_body = result.body
                        hook_context.request_headers["content-length"] = str(len(result.body))

        # 3. Get locks for the (possibly remapped) backend
        # IMPORTANT: Only lock backends in the SAME domain
        domain = self.domains[backend.domain_name]
        lock_targets = backend.get_lock_targets(domain.backends)

        # 4. Acquire locks using domain-specific LockManager
        acquired = True
        if lock_targets:
            logger.info(f"Acquiring locks {lock_targets} for backend '{backend.name}' (domain: {backend.domain_name})")
            acquired = await domain.lock_manager.acquire(backend.name, lock_targets)

        if not acquired:
            logger.warning(
                f"Lock timeout waiting for {lock_targets} → returning 503 for {full_path}"
            )
            return Response(
                status_code=503,
                content=f"Backend {backend.name} is locked by another backend",
                headers={"Retry-After": "10"}
            )

        # 5. Clean activation using activated_backends
        if backend.name not in self.activated_backends:
            self.activated_backends.add(backend.name)

            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name),
                    "on_backend_activated",
                    hook_context
                )

            # Self-lock within domain
            await domain.lock_manager.acquire(backend.name, [backend.name])

            # Stop backends this one locks
            for locked_name in lock_targets:
                locked_lifecycle = self.lifecycle_configs.get(locked_name)
                if locked_lifecycle and locked_lifecycle.on_deactivate:
                    logger.info(
                        f"[{backend.name}] Stopping locked backend '{locked_name}' first"
                    )
                    await self.lifecycle_executor.execute(
                        locked_name, locked_lifecycle.on_deactivate, is_activate=False
                    )

            # Activate self
            lifecycle = self.lifecycle_configs.get(backend.name)
            if lifecycle and lifecycle.on_activate:
                await self.lifecycle_executor.execute(
                    backend.name, lifecycle.on_activate, is_activate=True
                )

        # Increment in-flight count
        self.active_counts[backend.name] = self.active_counts.get(backend.name, 0) + 1

        try:
            # Call lifecycle hooks
            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name),
                    "on_locks_acquired",
                    hook_context
                )
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name),
                    "on_before_request",
                    hook_context
                )

            logger.info(f"Forwarding {request.method} {full_path} → {backend.name} ({backend.url})")

            query_string = f"?{request.url.query}" if request.url.query else ""
            target_url = f"{str(backend.url).rstrip('/')}{full_path}{query_string}"

            hop_by_hop = {"connection", "keep-alive", "transfer-encoding", "upgrade", "trailers"}
            filtered_headers = {
                k: v for k, v in hook_context.request_headers.items()
                if k.lower() not in hop_by_hop
            }

            client_ip = request.client.host if request.client else "unknown"

            existing_xff = filtered_headers.get("x-forwarded-for", "")
            if existing_xff:
                filtered_headers["x-forwarded-for"] = f"{existing_xff}, {client_ip}"
            else:
                filtered_headers["x-forwarded-for"] = client_ip

            original_host = request.headers.get("host", "")
            if original_host:
                filtered_headers["x-forwarded-host"] = original_host

            proto = request.headers.get("x-forwarded-proto") or getattr(request.url, "scheme", "http")
            filtered_headers["x-forwarded-proto"] = proto

            filtered_headers["x-real-ip"] = client_ip
            filtered_headers["accept-encoding"] = "identity"

            req = self.httpx_client.build_request(
                method=request.method,
                url=target_url,
                headers=filtered_headers,
                content=hook_context.request_body or request_body
            )

            response = await self.httpx_client.send(req, stream=True)

            elapsed = time.time() - start_time
            status_code = response.status_code
            content_type = response.headers.get("content-type", "").lower()

            hook_context.response_status = status_code
            hook_context.response_headers = dict(response.headers)

            if "text/event-stream" not in content_type:
                response_body = await response.aread()
                hook_context.response_body = response_body
            else:
                response_body = b""
                hook_context.response_body = None

            # Run response remapper
            if backend.remapper:
                remapper_instance = self.remapper_loader.get_remapper(backend.name)
                if remapper_instance:
                    result: Optional[RemapResult] = await self.remapper_loader.call_remap(
                        remapper_instance, hook_context
                    )
                    if result:
                        content = result.content
                        if isinstance(content, str):
                            content = content.encode("utf-8")
                        return Response(
                            status_code=result.status_code or status_code,
                            content=content or b"",
                            headers=result.response_headers or {}
                        )

            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name),
                    "on_response",
                    hook_context
                )

            if status_code >= 500:
                logger.error(f"Backend '{backend.name}' returned {status_code} for {full_path} (took {elapsed:.3f}s)")
            elif status_code >= 400:
                logger.warning(f"Backend '{backend.name}' returned {status_code} for {full_path} (took {elapsed:.3f}s)")
            else:
                logger.info(f"Backend '{backend.name}' responded {status_code} for {full_path} (took {elapsed:.3f}s)")

            if "text/event-stream" in content_type:
                return StreamingResponse(
                    self._stream_sse(response.aiter_lines(), backend.name),
                    status_code=status_code,
                    media_type="text/event-stream",
                    headers=dict(response.headers)
                )
            else:
                clean_response_headers = {
                    k: v for k, v in response.headers.items()
                    if k.lower() not in {
                        "content-length", "transfer-encoding", "content-encoding",
                        "connection", "keep-alive"
                    }
                }
                return Response(
                    status_code=status_code,
                    content=response_body,
                    headers=clean_response_headers
                )

        except httpx.TimeoutException as e:
            hook_context.error = f"Timeout: {str(e)}"
            logger.error(f"Backend '{backend.name}' timed out for {full_path}: {str(e)}")
            return Response(status_code=504, content=f"Backend {backend.name} timed out: {str(e)}")

        except httpx.RequestError as e:
            hook_context.error = f"RequestError: {str(e)}"
            logger.error(f"Backend '{backend.name}' connection error for {full_path}: {str(e)}")
            return Response(status_code=502, content=f"Backend {backend.name} error: {str(e)}")

        finally:
            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name),
                    "on_after_request",
                    hook_context
                )

            if lock_targets:
                await domain.lock_manager.release(backend.name, lock_targets)
                logger.info(f"Released locks {lock_targets} for backend '{backend.name}'")

            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name),
                    "on_locks_released",
                    hook_context
                )

            self.active_counts[backend.name] = max(0, self.active_counts.get(backend.name, 1) - 1)
            if self.active_counts[backend.name] <= 0:
                self.active_counts[backend.name] = 0
                await domain.lock_manager.release(backend.name, [backend.name])

    async def _handle_websocket(self, websocket: WebSocket, path: str) -> None:
        full_path = f"/{path}"
        host = websocket.headers.get("host", "")

        backend: Optional[Backend] = self._find_backend(host, full_path)
        if not backend:
            await websocket.close(code=1008, reason=f"Unknown path: {full_path}")
            return

        hook_context = HookContext(
            backend_name=backend.name,
            request_method="GET",
            request_path=full_path,
            request_headers=dict(websocket.headers),
            request_body=None
        )

        if backend.remapper:
            remapper_instance = self.remapper_loader.get_remapper(backend.name)
            if remapper_instance:
                result = await self.remapper_loader.call_remap(remapper_instance, hook_context)
                if result and result.status_code is not None:
                    await websocket.close(code=1011, reason="Remapper closed connection")
                    return
                if result and result.backend and result.backend in self.backends:
                    backend = self.backends[result.backend]
                if result and result.path:
                    full_path = result.path

        # Get locks within same domain
        domain = self.domains[backend.domain_name]
        lock_targets = backend.get_lock_targets(domain.backends)

        acquired = True
        if lock_targets:
            acquired = await domain.lock_manager.acquire(backend.name, lock_targets)
        if not acquired:
            await websocket.close(code=1013, reason="Locked")
            return

        if backend.name not in self.activated_backends:
            self.activated_backends.add(backend.name)

            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name), "on_backend_activated", hook_context
                )
            await domain.lock_manager.acquire(backend.name, [backend.name])

            for locked_name in lock_targets:
                locked_lifecycle = self.lifecycle_configs.get(locked_name)
                if locked_lifecycle and locked_lifecycle.on_deactivate:
                    logger.info(f"[{backend.name}] Stopping locked backend '{locked_name}'")
                    await self.lifecycle_executor.execute(
                        locked_name, locked_lifecycle.on_deactivate, is_activate=False
                    )

            lifecycle = self.lifecycle_configs.get(backend.name)
            if lifecycle and lifecycle.on_activate:
                await self.lifecycle_executor.execute(
                    backend.name, lifecycle.on_activate, is_activate=True
                )

        self.active_counts[backend.name] = self.active_counts.get(backend.name, 0) + 1

        try:
            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name), "on_locks_acquired", hook_context
                )
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name), "on_before_request", hook_context
                )

            backend_url_str = str(backend.url).rstrip("/")
            
            query_string = f"?{websocket.url.query}" if websocket.url.query else ""
            ws_url = ("wss://" if backend_url_str.startswith("https://") else "ws://") + \
                     backend_url_str.split("://", 1)[-1] + full_path + query_string

            HOP_BY_HOP = {
                "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade",
                "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions",
                "sec-websocket-accept", "sec-websocket-protocol"
            }

            filtered_headers = {
                k: v for k, v in hook_context.request_headers.items()
                if k.lower() not in HOP_BY_HOP
            }

            client_ip = websocket.client.host if websocket.client else "unknown"

            if "x-forwarded-for" in filtered_headers:
                filtered_headers["x-forwarded-for"] += f", {client_ip}"
            else:
                filtered_headers["x-forwarded-for"] = client_ip

            filtered_headers["x-forwarded-host"] = host or websocket.headers.get("host", "")
            filtered_headers["x-real-ip"] = client_ip

            proto = websocket.headers.get("x-forwarded-proto")
            if not proto:
                proto = "https" if websocket.url.scheme == "wss" else "http"
            filtered_headers["x-forwarded-proto"] = proto

            origin_host = host or websocket.headers.get("host", "")
            if origin_host:
                scheme = "https" if proto == "https" or websocket.url.scheme == "wss" else "http"
                filtered_headers["origin"] = f"{scheme}://{origin_host}"

            if "referer" not in filtered_headers and "origin" in filtered_headers:
                filtered_headers["referer"] = filtered_headers["origin"] + full_path

            logger.info(f"Forwarding WebSocket {full_path} → {backend.name} ({ws_url})")

            ws_uri = ws_url
            if origin_host:
                ws_uri = ws_url.replace("127.0.0.1:9090", origin_host)
            
            backend_ws = await websockets.connect(
                ws_uri,
                additional_headers=filtered_headers,
            )

            await websocket.accept(
                subprotocol=backend_ws.subprotocol if hasattr(backend_ws, "subprotocol") else None
            )

            async def client_to_backend():
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        if "text" in msg:
                            await backend_ws.send(msg["text"])
                        elif "bytes" in msg:
                            await backend_ws.send(msg["bytes"])
                except Exception:
                    pass
                finally:
                    if not backend_ws.closed:
                        await backend_ws.close()

            async def backend_to_client():
                try:
                    async for message in backend_ws:
                        if isinstance(message, str):
                            await websocket.send_text(message)
                        else:
                            await websocket.send_bytes(message)
                except Exception:
                    pass
                finally:
                    if websocket.client_state != WebSocketState.DISCONNECTED:
                        try:
                            await websocket.close()
                        except Exception:
                            pass

            await asyncio.gather(client_to_backend(), backend_to_client(), return_exceptions=True)

        except Exception as e:
            logger.error(f"WebSocket proxy error for '{backend.name}' {full_path}: {e}")
            if websocket.client_state == WebSocketState.CONNECTING:
                await websocket.close(code=1011)
        finally:
            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name), "on_after_request", hook_context
                )
            if lock_targets:
                await domain.lock_manager.release(backend.name, lock_targets)
            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name), "on_locks_released", hook_context
                )

            self.active_counts[backend.name] = max(0, self.active_counts.get(backend.name, 1) - 1)
            if self.active_counts[backend.name] <= 0:
                self.active_counts[backend.name] = 0
                await domain.lock_manager.release(backend.name, [backend.name])

    async def _stream_sse(self, aiter_lines, backend_name: str = None):
        stripper = None
        if backend_name:
            remapper = self.remapper_loader.get_remapper(backend_name)
            if remapper and hasattr(remapper, "get_streaming_stripper"):
                stripper = remapper.get_streaming_stripper()

        async for line in aiter_lines:
            if not line.startswith("data: "):
                yield line + "\n"
                continue

            json_part = line[6:].strip()
            if not json_part or json_part == "[DONE]":
                yield line + "\n"
                continue

            try:
                chunk = json.loads(json_part)

                if "choices" in chunk:
                    for choice in chunk.get("choices", []):
                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"] or ""
                            if stripper:
                                stripped = stripper.process_chunk(content)
                                if stripped is not None:
                                    choice["delta"]["content"] = stripped
                                else:
                                    continue

                yield "data: " + json.dumps(chunk) + "\n\n"

            except json.JSONDecodeError:
                yield line + "\n"

    async def _stream_response(self, aiter_bytes, backend_name: str = None):
        stripper = None
        if backend_name:
            remapper = self.remapper_loader.get_remapper(backend_name)
            if remapper and hasattr(remapper, 'get_streaming_stripper'):
                stripper = remapper.get_streaming_stripper()
        
        buffer = ""
        async for chunk in aiter_bytes:
            chunk_str = chunk.decode('utf-8', errors='replace')
            buffer += chunk_str
            
            try:
                data = json.loads(buffer)
                if "choices" in data:
                    for choice in data.get("choices", []):
                        if "text" in choice:
                            choice["text"] = strip_thinking_at_start(choice["text"])
                        elif "message" in choice and "content" in choice["message"]:
                            choice["message"]["content"] = strip_thinking_at_start(
                                choice["message"]["content"]
                            )
                buffer = json.dumps(data)
            except json.JSONDecodeError:
                pass
            
            if stripper:
                stripped = stripper.process_chunk(buffer)
                if stripped is not None:
                    yield stripped.encode('utf-8')
                    buffer = ""
            else:
                yield chunk

    async def shutdown(self):
        await self.httpx_client.aclose()

    async def run(self) -> None:
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.config.server.host,
            port=self.config.server.port,
            log_level="info",
            access_log=False,
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        finally:
            await self.shutdown()
