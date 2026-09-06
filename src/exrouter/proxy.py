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

        # Debug: log request headers
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Request headers: {dict(request.headers)}")
            try:
                body = await request.body()
                if body:
                    logger.debug(f"  Request body length: {len(body)} bytes")
                    # Log first 512 bytes of body for debugging
                    body_preview = body[:512].decode("utf-8", errors="replace")
                    logger.debug(f"  Request body preview: {body_preview!r}")
            except Exception:
                pass  # Body might be streaming

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"← {request.method} {request.url.path} "
            f"status={response.status_code} ({process_time:.0f}ms)"
        )

        # Debug: log response headers
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Response headers: {dict(response.headers)}")

        return response


class LockProxy:
    """Main proxy server with HTTP connection pooling, WebSocket legs, locking, hooks, and request remapping.
    
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

        for domain_name, domain_config in config.domains.items():
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
        
        # Open WebSocket legs to backends, by backend name. One dedicated leg
        # per downstream client connection (nginx/Caddy style: no pooling).
        # A WebSocket is a stateful bidirectional session, so a shared leg
        # would misdeliver messages and an idle leg would hold backend
        # resources needlessly. Legs are closed when the session ends or the
        # backend is deactivated.
        self.ws_legs: dict[str, set] = {}

        # Backends for which we already warned that locks: does not apply to
        # their WebSocket paths.
        self._ws_lock_warned: set[str] = set()

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
        """Domain-aware routing.
        
        1. Extract domain from host (remove port)
        2. Filter backends by domain match:
           - If host matches a domain pattern: only match backends with that domain
           - If host matches no domain: only match backends with empty domain list
        3. Then match by path
        4. Return first match
        
        Example:
        - Request to openwebui.unnsvc.org → only open-webui matches (domain: ["openwebui.unnsvc.org"])
        - Request to 127.0.0.1:4001/v1/tokenize → only llm matches (domain: [])
        """
        import fnmatch

        raw_host = host or ""
        host_no_port = raw_host.lower().split(":")[0]
        full_path = f"/{path}" if not path.startswith("/") else path

        # Step 1: Find which domain pattern (if any) the host matches
        matched_domain_pattern: Optional[str] = None
        for backend in self.backends.values():
            domains = backend.domain
            if isinstance(domains, str):
                domains = [domains]
            if not domains:
                continue

            for d in domains:
                d_clean = str(d).lower().split(":")[0]
                if fnmatch.fnmatch(host_no_port, d_clean):
                    matched_domain_pattern = d_clean
                    break
            if matched_domain_pattern:
                break

        # Step 2: Filter backends by domain
        candidates: list[Backend] = []
        for backend in self.backends.values():
            domains = backend.domain
            if isinstance(domains, str):
                domains = [domains]
            if not domains:
                domains = []

            if matched_domain_pattern is not None:
                # Host matched a domain → only match backends with that domain
                for d in domains:
                    d_clean = str(d).lower().split(":")[0]
                    if fnmatch.fnmatch(host_no_port, d_clean):
                        candidates.append(backend)
                        break
            else:
                # Host matched no domain → only match backends with empty domain list
                if not domains:
                    candidates.append(backend)

        # Step 3: Match by path within filtered candidates
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Lock targets: {lock_targets}")
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

            # Self-lock within domain
            await domain.lock_manager.acquire(backend.name, [backend.name])

            # Stop backends this one locks (VRAM handoff)
            await self._deactivate_locked_backends(backend.name, lock_targets)

            # Activate self
            lifecycle = self.lifecycle_configs.get(backend.name)
            if lifecycle and lifecycle.on_activate:
                await self.lifecycle_executor.execute(
                    backend.name, lifecycle.on_activate, is_activate=True
                )

        # Increment in-flight count
        self.active_counts[backend.name] = self.active_counts.get(backend.name, 0) + 1
        
        response = None  # Track response for cleanup in finally block
        streaming = False  # True once an SSE response is returned: cleanup moves to the stream wrapper

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

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Backend URL: {target_url}")
                logger.debug(f"  Backend headers: {filtered_headers}")
                body_len = len(hook_context.request_body or request_body or b"")
                logger.debug(f"  Backend request body: {body_len} bytes")

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
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  Backend response: {status_code}, {len(response_body)} bytes, content-type={content_type}")
            else:
                response_body = b""
                hook_context.response_body = None
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  Backend response: {status_code}, SSE stream, content-type={content_type}")

            # Run response remapper
            if backend.remapper:
                logger.debug(f"Response remapper configured for backend '{backend.name}'")
                remapper_instance = self.remapper_loader.get_remapper(backend.name)
                logger.debug(f"Remapper instance: {remapper_instance}")
                if remapper_instance:
                    logger.debug("Calling remap for response...")
                    result: Optional[RemapResult] = await self.remapper_loader.call_remap(
                        remapper_instance, hook_context
                    )
                    logger.debug(f"Remap result: {result}")
                    if result:
                        content = result.content
                        if isinstance(content, str):
                            content = content.encode("utf-8")
                        return Response(
                            status_code=result.status_code or status_code,
                            content=content or b"",
                            headers=result.response_headers or {}
                        )
            else:
                logger.debug(f"No remapper configured for backend '{backend.name}'")

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
                # Locks must be held for the entire duration of the stream.
                # The cleanup cannot run in this function's finally (which fires
                # before the first streamed byte reaches the client); instead it
                # runs in the stream wrapper's finally, when the stream is fully
                # delivered, the client disconnects, or the stream raises.
                sse_response = StreamingResponse(
                    self._stream_with_cleanup(
                        self._stream_sse(response.aiter_lines(), backend.name),
                        domain,
                        backend,
                        lock_targets,
                        hook_context,
                        response,
                    ),
                    status_code=status_code,
                    media_type="text/event-stream",
                    headers=dict(response.headers),
                )
                streaming = True
                return sse_response
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
            # For streaming responses the cleanup is deferred to the stream
            # wrapper (_stream_with_cleanup) so locks are held until the stream
            # is fully delivered. For everything else it runs here, as before.
            if not streaming:
                await self._finalize_request(domain, backend, lock_targets, hook_context, response)

    async def _finalize_request(
        self,
        domain: LockDomain,
        backend: Backend,
        lock_targets: list[str],
        hook_context: HookContext,
        response: Optional[httpx.Response] = None,
    ) -> None:
        """Release locks, decrement the in-flight count, run post-request hooks.

        Must only run once the request has *completed*: immediately for
        buffered responses, and after the SSE stream has been fully delivered
        (or the client went away) for streaming responses. Holding the locks
        for the entire response duration is the core VRAM-exclusivity guarantee.
        """
        if response is not None and hook_context.response_headers:
            try:
                # Only close if not already consumed/streaming
                # (streaming responses are closed by _stream_with_cleanup)
                content_type_lower = hook_context.response_headers.get('content-type', '').lower()
                if 'text/event-stream' not in content_type_lower:
                    response.close()
            except Exception as e:
                logger.debug(f"Error closing response in finally: {e}")

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

    async def _stream_with_cleanup(self, aiter, domain: LockDomain, backend: Backend,
                                   lock_targets: list[str], hook_context: HookContext,
                                   response: Optional[httpx.Response]):
        """Wrap an SSE stream so locks are held until the stream is consumed.

        The finally block runs when the stream completes normally, when the
        consumer stops early (client disconnect), or when the stream raises.
        In every case the locks are released, the in-flight count is
        decremented, post-request hooks fire, and the backend response is
        closed so its connection returns to the pool.
        """
        try:
            async for line in aiter:
                yield line
        finally:
            await self._finalize_request(domain, backend, lock_targets, hook_context)
            if response is not None:
                try:
                    response.close()
                except Exception as e:
                    logger.debug(f"Error closing streaming response: {e}")

    async def _close_backend_ws_legs(self, backend_name: str) -> None:
        """Close all open WebSocket legs to a backend.

        Called when a backend is deactivated (its service is stopped).
        Without this, legs would keep a dead service's address open and
        clients would hang instead of seeing a clean disconnect (and
        reconnecting once the backend is re-activated).
        """
        legs = self.ws_legs.get(backend_name)
        if not legs:
            return
        for leg in list(legs):
            try:
                await leg.close()
            except Exception:
                pass
        self.ws_legs.pop(backend_name, None)
        logger.info(
            f"Closed {len(legs)} WebSocket leg(s) to deactivated backend '{backend_name}'"
        )

    async def _deactivate_locked_backends(self, backend_name: str, lock_targets: list[str]) -> None:
        """Deactivate the backends that ``backend_name`` locks (same domain).

        Shared by the HTTP and WebSocket activation paths: before a
        backend's own service is started, the services of the backends it
        locks must be stopped (VRAM handoff). Each such target is dropped
        from activated_backends so its next use re-runs the full activation
        block (and its on_activate, which restarts the service): without
        this the VRAM handoff is one-way and a deactivated backend could
        never be started again. Open WebSocket legs to a stopped target are
        closed so clients see a clean disconnect (and typically reconnect
        once the backend is re-activated) instead of a connection hanging
        against a dead service.
        """
        for locked_name in lock_targets:
            locked_lifecycle = self.lifecycle_configs.get(locked_name)
            if locked_lifecycle and locked_lifecycle.on_deactivate:
                logger.info(
                    f"[{backend_name}] Stopping locked backend '{locked_name}' first"
                )
                await self.lifecycle_executor.execute(
                    locked_name, locked_lifecycle.on_deactivate, is_activate=False
                )
                self.activated_backends.discard(locked_name)
                await self._close_backend_ws_legs(locked_name)

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

        # Locking policy for WebSocket sessions:
        # - Cross-backend locks (locks:) are NEVER acquired or held for a WS
        #   session. A WS session can stay open for hours (a parked chat tab,
        #   a phone left on a TTS page); waiting on them at connect time or
        #   holding them for the session lifetime would exclusive-lock the
        #   rest of the domain the whole time.
        # - The self-lock + activation path IS taken: the backend is marked
        #   in-use (its service is up and a client is connected) for the
        #   duration of the session, and activation includes the same VRAM
        #   handoff as the HTTP path: the backends this one locks are
        #   deactivated before its own service starts.
        domain = self.domains[backend.domain_name]
        lock_targets = []  # WS sessions never acquire cross-backend locks

        if backend.locks and backend.name not in self._ws_lock_warned:
            self._ws_lock_warned.add(backend.name)
            logger.warning(
                f"Backend '{backend.name}' declares locks: {backend.locks}. "
                f"WebSocket sessions to this backend do not wait for those "
                f"backends to become idle before connecting (the HTTP "
                f"wait/503 gate does not apply to WebSocket upgrades); the "
                f"locked backends are deactivated when this backend is "
                f"activated instead."
            )

        if backend.name not in self.activated_backends:
            self.activated_backends.add(backend.name)

            # Self-lock: marks the backend in-use while the session is open
            # (released in the finally block when the last session ends).
            await domain.lock_manager.acquire(backend.name, [backend.name])

            # VRAM handoff, same as the HTTP activation path: stop the
            # backends this one locks before its own service starts. (The
            # session does not acquire or hold their locks for its lifetime;
            # it only deactivates them here, at activation time.)
            await self._deactivate_locked_backends(
                backend.name, backend.get_lock_targets(domain.backends)
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

            # One dedicated backend leg per client connection (nginx/Caddy
            # style). No pooling: the leg's lifetime matches the session's.
            backend_ws = await websockets.connect(
                ws_url,
                additional_headers=filtered_headers
            )
            self.ws_legs.setdefault(backend.name, set()).add(backend_ws)

            await websocket.accept(
                subprotocol=backend_ws.subprotocol if backend_ws.subprotocol else None
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
                    # Close this session's dedicated backend leg
                    try:
                        await backend_ws.close()
                    except Exception:
                        pass
                    self.ws_legs.get(backend.name, set()).discard(backend_ws)

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
                    self.ws_legs.get(backend.name, set()).discard(backend_ws)
                    # Log when the backend side closed the leg (backend went
                    # down, or was deactivated): clients typically reconnect,
                    # so this is the line that makes that loop visible.
                    if getattr(backend_ws, "close_code", None) is not None:
                        logger.info(
                            f"Backend WebSocket leg to '{backend.name}' closed "
                            f"(code {backend_ws.close_code}) for {full_path}; "
                            f"client session ending (clients typically reconnect)"
                        )
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
        for name in list(self.ws_legs):
            await self._close_backend_ws_legs(name)
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
