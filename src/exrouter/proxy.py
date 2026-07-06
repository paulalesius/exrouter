"""EXRouter - routes requests to backends with global locking and optional request remapping."""

import asyncio
import json
import logging
import time
import re

from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from .backend import Backend
from .config import Config, LifecycleConfig
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
    # Pattern matches opening tag at start, captures everything after closing tag
    # Note: No \s* required AFTER opening tag (e.g.,  not  )
    patterns = [
        #  format (closing tag is )
        r'^\s*<think>(.*?)\s*</think>\s*(.*)',
        #  format  (closing tag is </reasoning>)
        r'^\s*<think\b[^>]*>(.*?)\s*</think>\s*(.*)',
        #  format
        r'^\s*<reasoning\b[^>]*>(.*?)\s*</reasoning>\s*(.*)',
        #  format (self-closing or paired tags)
        r'^\s*</?think\s*>(.*?)\s*</?think\s*>\s*(.*)',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # Return content after the closing tag
            return match.group(2).strip()
    
    # No thinking tag at start, return as-is
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


@dataclass
class LockState:
    """Track which backend holds a lock."""
    locked_by: str


class LockManager:
    """Manages global locks across backends using Condition for proper waiting."""

    def __init__(self, backends: dict[str, Backend], timeout: int = 300):
        self.backends = backends
        self.locks: dict[str, LockState] = {}
        self.holder_counts: dict[str, int] = {}  # target -> number of in-flight holders from locked_by
        self.condition = asyncio.Condition()
        self.timeout = timeout

    async def acquire(self, backend_name: str, lock_targets: list[str]) -> bool:
        async with self.condition:
            try:
                await asyncio.wait_for(
                    self._wait_until_free(backend_name, lock_targets),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                return False

            for target in lock_targets:
                if target not in self.locks:
                    self.locks[target] = LockState(locked_by=backend_name)
                    self.holder_counts[target] = 1
                else:
                    # Re-acquire by same backend (concurrent requests) — just increment holder count
                    self.holder_counts[target] += 1
            return True

    async def _wait_until_free(self, backend_name: str, lock_targets: list[str]) -> None:
        while any(
            target in self.locks and self.locks[target].locked_by != backend_name
            for target in lock_targets
        ):
            await self.condition.wait()

    async def release(self, backend_name: str, lock_targets: list[str]) -> None:
        async with self.condition:
            for target in lock_targets:
                if target in self.locks and self.locks[target].locked_by == backend_name:
                    self.holder_counts[target] -= 1
                    if self.holder_counts[target] <= 0:
                        del self.locks[target]
                        del self.holder_counts[target]
            self.condition.notify_all()

    def is_locked(self, backend_name: str) -> bool:
        return backend_name in self.locks


class LockProxy:
    """Main proxy server with connection pooling, locking, hooks, and request remapping."""

    def __init__(self, config: Config):
        self.config = config

        # Convert backend configs to Backend instances
        self.backends: dict[str, Backend] = {}
        for name, backend_config in config.backends.items():
            self.backends[name] = Backend(
                name=name,
                url=str(backend_config.url),
                paths=backend_config.paths,
                locks=backend_config.locks,
                domain=backend_config.domain,
                script=backend_config.script,
                remapper=backend_config.remapper,
            )

        # Initialize lock manager
        self.lock_manager = LockManager(
            self.backends,
            timeout=config.global_lock.timeout
        ) if config.global_lock.enabled else None

        # Initialize hook loader
        self.hook_loader = HookLoader()
        for backend in self.backends.values():
            if backend.script:
                self.hook_loader.load_script(backend.name, backend.script)

        # NEW: Initialize remapper loader
        self.remapper_loader = RemapperLoader()
        for backend in self.backends.values():
            if backend.remapper:
                self.remapper_loader.load_script(backend.name, backend.remapper)

        # Initialize declarative lifecycle executor (systemd/shell/wait)
        self.lifecycle_executor = LifecycleExecutor()
        self.lifecycle_configs: dict[str, Optional[LifecycleConfig]] = {}
        for name, backend_config in config.backends.items():
            self.lifecycle_configs[name] = backend_config.lifecycle

        # Track in-flight requests per backend
        self.active_counts: dict[str, int] = {name: 0 for name in self.backends}

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
        # Note: We deliberately do NOT register a literal "/" route here.
        # The catch-all "/{path:path}" below handles root requests so that
        # domain-based backends with paths: ["/"] or paths: ["*"] can serve
        # the homepage and all sub-paths of their domain.
        # /health remains internal to EXRouter for monitoring.

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy_request(request: Request, path: str):
            return await self._handle_request(request, path)

    def _find_backend(self, request: Request, path: str) -> Optional[Backend]:
        """Select backend using combined domain + path matching.
        
        Rules (as requested):
        - If the incoming request's Host header matches ANY backend's declared `domain`,
          then ONLY backends that declare a matching domain are eligible.
        - Pure path-based backends (no `domain:` declared) are ignored in that case.
        - Multiple backends can share the same domain but use different `paths:`.
        - If no declared domain matches the request Host, fall back to pure path matching
          (original behavior for stt_custom, llm, etc.).
        """
        host = request.headers.get("host", "")
        
        # Step 1: Does this request's Host match any backend that declares domains?
        request_matches_some_domain = any(
            b.domain and b.matches_domain(host) for b in self.backends.values()
        )
        
        for b in self.backends.values():
            domain_ok = True
            if b.domain:
                # Backend declares domains → must match the request Host
                domain_ok = b.matches_domain(host)
            else:
                # Backend has NO domain declared (pure path backend like stt_custom)
                if request_matches_some_domain:
                    # A domain was specified in the request → skip pure path backends
                    domain_ok = False
            
            path_ok = b.matches_path(path)
            if domain_ok and path_ok:
                return b
        return None

    async def _handle_request(self, request: Request, path: str) -> Response:
        full_path = f"/{path}"
        start_time = time.time()

        # 1. Find matching backend by domain (if declared) + path.
        #    - If backend declares `domain:`, the Host header must match one of its patterns
        #      AND the path must match one of its path patterns.
        #    - If backend has no `domain:` (empty), only path matching is required (legacy behavior).
        #    This design allows multiple backends to share the same domain name but serve
        #    different paths (e.g. api.example.com/v1/chat vs api.example.com/v1/embed).
        backend: Optional[Backend] = self._find_backend(request, full_path)

        if not backend:
            logger.warning(
                f"No backend matched host+path for Host={request.headers.get('host', '-')} path={full_path} "
                f"→ returning 404 from EXRouter"
            )
            return Response(status_code=404, content=f"Unknown path: {full_path}")

        # 2. NEW: Run request remapper (if configured) BEFORE acquiring locks
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
                    # Short-circuit with a direct response?
                    if result.status_code is not None:
                        content = result.content
                        if isinstance(content, str):
                            content = content.encode("utf-8")
                        return Response(
                            status_code=result.status_code,
                            content=content or b"",
                            headers=result.response_headers or {}
                        )

                    # Switch backend?
                    if result.backend and result.backend in self.backends:
                        backend = self.backends[result.backend]
                        hook_context.backend_name = backend.name  # update context
                        logger.info(f"Remapped request to backend '{backend.name}'")

                    # Rewrite path?
                    if result.path is not None:
                        full_path = result.path
                        hook_context.request_path = full_path

                    # Apply other modifications
                    if result.method:
                        # We can't easily change method after body read, but we can log it
                        logger.info(f"Remapper requested method change to {result.method} (not fully supported yet)")
                    if result.headers:
                        hook_context.request_headers.update(result.headers)
                    if result.body is not None:
                        hook_context.request_body = result.body
                        request_body = result.body
                        hook_context.request_headers["content-length"] = str(len(result.body))

        # 3. Get locks for the (possibly remapped) backend
        lock_targets = backend.get_lock_targets(self.backends)

        # 4. Acquire locks
        acquired = True
        if self.lock_manager and lock_targets:
            logger.info(f"Acquiring locks {lock_targets} for backend '{backend.name}'")
            acquired = await self.lock_manager.acquire(backend.name, lock_targets)

        if not acquired:
            logger.warning(
                f"Lock timeout waiting for {lock_targets} → returning 503 for {full_path}"
            )
            return Response(
                status_code=503,
                content=f"Backend {backend.name} is locked by another backend",
                headers={"Retry-After": "10"}
            )

        # 5. Backend activation tracking
        was_active = self.active_counts.get(backend.name, 0) > 0
        self.active_counts[backend.name] = self.active_counts.get(backend.name, 0) + 1
        if not was_active:
            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name),
                    "on_backend_activated",
                    hook_context
                )

            # Mark this backend as busy in the lock manager (self-lock).
            # This ensures that any other backend that declares this one in its `locks:`
            # list will properly wait in acquire() until we go idle.
            # Without this, a contender could start while we still have in-flight requests.
            if self.lock_manager:
                await self.lock_manager.acquire(backend.name, [backend.name])

            # === IMPORTANT ORDERING FOR RESOURCE CONTENTION ===
            # When activating a backend that locks other backends, we MUST:
            # 1. FIRST stop the locked backends (free their resources / VRAM / GPU)
            # 2. THEN start/activate the current backend (its on_activate + wait_for its port)
            #
            # This prevents the new service from starting while the conflicting one
            # is still holding resources (the previous bug).
            # The systemctl stop waits for systemd to process the stop (up to 30s),
            # and the activating backend's own wait_for: port will further wait until
            # its port is ready (implicitly waiting for resources to be truly free).

            # 1. Stop all locked backends first (auto on_deactivate)
            for locked_name in lock_targets:
                locked_lifecycle = self.lifecycle_configs.get(locked_name)
                if locked_lifecycle and locked_lifecycle.on_deactivate:
                    logger.info(
                        f"[{backend.name}] Stopping locked backend '{locked_name}' first "
                        f"(to free resources before activating)"
                    )
                    await self.lifecycle_executor.execute(
                        locked_name, locked_lifecycle.on_deactivate, is_activate=False
                    )

            # 2. THEN activate self (start own service + its wait_for conditions)
            # Declarative lifecycle actions (runs in addition to hook if both configured)
            lifecycle = self.lifecycle_configs.get(backend.name)
            if lifecycle and lifecycle.on_activate:
                await self.lifecycle_executor.execute(
                    backend.name, lifecycle.on_activate, is_activate=True
                )

        try:
            # Call lifecycle hooks on the final backend
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

            target_url = f"{str(backend.url).rstrip('/')}{full_path}"

            hop_by_hop = {"connection", "keep-alive", "transfer-encoding", "upgrade", "trailers"}
            filtered_headers = {
                k: v for k, v in hook_context.request_headers.items()
                if k.lower() not in hop_by_hop
            }

            # === Proper reverse proxy headers (X-Forwarded-*) ===
            # These are important for backends that log client info, do rate limiting,
            # virtual hosting internally, or need to know original request context.
            client_ip = request.client.host if request.client else "unknown"

            # X-Forwarded-For: append client IP (preserve chain if already present)
            existing_xff = filtered_headers.get("x-forwarded-for", "")
            if existing_xff:
                filtered_headers["x-forwarded-for"] = f"{existing_xff}, {client_ip}"
            else:
                filtered_headers["x-forwarded-for"] = client_ip

            # X-Forwarded-Host: original Host header from client
            original_host = request.headers.get("host", "")
            if original_host:
                filtered_headers["x-forwarded-host"] = original_host

            # X-Forwarded-Proto: scheme (https/http). Prefer any already-forwarded value.
            proto = request.headers.get("x-forwarded-proto") or getattr(request.url, "scheme", "http")
            filtered_headers["x-forwarded-proto"] = proto

            # X-Real-IP: common convention for the immediate client IP
            filtered_headers["x-real-ip"] = client_ip

            # Prevent backends from returning compressed responses (gzip/deflate/br).
            # httpx auto-decompresses when it sees Content-Encoding, but then we were
            # forwarding the original Content-Encoding + Content-Length, causing
            # "Failed to uncompress gzip stream" (wget) and h11 Content-Length errors.
            # Forcing "identity" makes backends send plain text (or we strip it anyway).
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

            # Only fully buffer non-streaming responses.
            # For SSE (text/event-stream), we must NOT call aread() so that
            # aiter_lines() can stream chunks incrementally from the backend
            # as they arrive. This enables true streaming to clients.
            if "text/event-stream" not in content_type:
                response_body = await response.aread()
                hook_context.response_body = response_body
            else:
                response_body = b""
                hook_context.response_body = None

            # Run response remapper (if configured).
            # For SSE responses, response_body is None so remappers that
            # check `if context.response_body:` (like the think-strip one)
            # will skip full-body processing and fall through to the
            # StreamingResponse path (which uses incremental stripper).
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
                # Clean hop-by-hop and length-related headers from backend so Starlette
                # can set correct Content-Length based on the body we actually return.
                # This prevents "Too much data for declared Content-Length" errors
                # with some backends (especially modern web UIs).
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

            if self.lock_manager and lock_targets:
                await self.lock_manager.release(backend.name, lock_targets)
                logger.info(f"Released locks {lock_targets} for backend '{backend.name}'")

            if backend.script:
                await self.hook_loader.call_hook(
                    self.hook_loader.get_hook(backend.name),
                    "on_locks_released",
                    hook_context
                )

            # Deactivation tracking
            self.active_counts[backend.name] = self.active_counts.get(backend.name, 1) - 1
            if self.active_counts[backend.name] <= 0:
                self.active_counts[backend.name] = 0
                if self.lock_manager:
                    # Release our self-lock so that any backend locking us can now acquire
                    await self.lock_manager.release(backend.name, [backend.name])

                if backend.script:
                    await self.hook_loader.call_hook(
                        self.hook_loader.get_hook(backend.name),
                        "on_backend_deactivated",
                        hook_context
                    )
                # Declarative lifecycle actions
                lifecycle = self.lifecycle_configs.get(backend.name)
                if lifecycle and lifecycle.on_deactivate:
                    # For backends that declare `locks:` (mutually exclusive group like llm <-> stt_custom),
                    # do NOT automatically stop them when their request count goes to zero.
                    # They should remain running ("stay in this mode") until another conflicting
                    # backend activates (which will stop them first via the "stop locked first" logic).
                    #
                    # This matches the expectation: once stt_custom is active, it stays
                    # active (service running) until something that locks it needs to run.
                    if not backend.locks:
                        await self.lifecycle_executor.execute(
                            backend.name, lifecycle.on_deactivate, is_activate=False
                        )

                # Note: We intentionally do NOT auto-trigger on_activate for the previously
                # locked backends here. 
                # 
                # Reason: On quick failures (connection error, timeout, etc.) this would
                # immediately restart the previously stopped service (as you saw with llama-server
                # coming back after the failed /transcribe). 
                # 
                # The critical & safe behavior is only the other direction:
                #   "When I activate and I lock something → first stop the locked ones"
                # This prevents resource contention and matches the spirit of your original hook.py.
                #
                # If you want the "switch back to alternative when this one goes idle" behavior,
                # you can add it in a custom hook's on_backend_deactivated, or we can make it
                # optional via a config flag later.

    async def _stream_sse(self, aiter_lines, backend_name: str = None):
        """Stream SSE, using remapper's stripper if available."""
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
                                    continue  # still buffering think block
                            # else: no stripper, pass through

                yield "data: " + json.dumps(chunk) + "\n\n"

            except json.JSONDecodeError:
                yield line + "\n"

    async def _stream_response(self, aiter_bytes, backend_name: str = None):
        """Stream regular responses, optionally stripping thinking tags."""
        stripper = None
        if backend_name:
            remapper = self.remapper_loader.get_remapper(backend_name)
            if remapper and hasattr(remapper, 'get_streaming_stripper'):
                stripper = remapper.get_streaming_stripper()
        
        buffer = ""
        async for chunk in aiter_bytes:
            chunk_str = chunk.decode('utf-8', errors='replace')
            buffer += chunk_str
            
            # Try to parse as JSON and strip thinking tags
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
                # Not complete JSON yet, continue buffering
                pass
            
            # Apply streaming stripper if available
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
