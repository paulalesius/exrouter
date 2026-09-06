"""Request remapping - allows rewriting requests or short-circuiting before backend forwarding.

This is especially useful when:
- Clients expect TEI-compatible endpoints (/v1/info, etc.) that the real server doesn't implement.
- You want to rewrite paths or route to a different backend dynamically.
- You need to return synthetic responses for certain paths.

Note: remap() runs BEFORE the proxy acquires locks. Never perform direct
backend I/O from a remapper: it would bypass the lock system and let the
other backends share the GPU mid-request. Either rewrite the request
(path, body, headers) and let the proxy forward it under lock, or
short-circuit with a static response that needs no GPU.
"""

from dataclasses import dataclass
from typing import Optional, Any
import logging
import asyncio

from .hooks import HookContext

logger = logging.getLogger("exrouter.remapper")


@dataclass
class RemapResult:
    """Result returned from a remapper's remap() method.

    You can either:
    - Modify the request (change backend, path, method, headers, body)
    - Short-circuit and return a direct HTTP response (status_code + content)
    """

    # --- Request modification ---
    backend: Optional[str] = None          # Switch to a different backend by name
    path: Optional[str] = None             # Rewrite the request path
    method: Optional[str] = None           # Change HTTP method
    headers: Optional[dict[str, str]] = None
    body: Optional[bytes] = None

    # --- Short-circuit response (return without hitting any backend) ---
    status_code: Optional[int] = None
    content: Optional[bytes | str] = None
    response_headers: Optional[dict[str, str]] = None


class RequestRemapper:
    """Base class that user scripts should implement.

    Define a class named `RequestRemapper` in your script that inherits from this
    (or just duck-types the `remap` method).
    """

    def remap(self, context: HookContext) -> Optional[RemapResult]:
        """Called after initial backend matching but before locking and forwarding.

        Return None to proceed normally.
        Return a RemapResult to modify routing or short-circuit.
        """
        return None


class RemapperLoader:
    """Loads and manages per-backend remapper scripts (similar to HookLoader)."""

    def __init__(self):
        self._remappers: dict[str, RequestRemapper] = {}
        self._modules: dict[str, object] = {}

    def load_script(self, backend_name: str, script_path: str) -> Optional[RequestRemapper]:
        """Load a remapper script from disk."""
        import importlib.util
        from pathlib import Path

        path = Path(script_path)
        if not path.exists():
            logger.error(f"Remapper script not found for backend '{backend_name}': {script_path}")
            return None

        try:
            module_name = f"exrouter_remapper_{backend_name}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to load spec for remapper: {script_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, 'RequestRemapper'):
                logger.error(f"Remapper script must define 'RequestRemapper' class: {script_path}")
                return None

            remapper_class = getattr(module, 'RequestRemapper')
            instance = remapper_class()

            # Duck-type check
            if not hasattr(instance, 'remap'):
                logger.error(f"RequestRemapper in {script_path} must implement 'remap' method")
                return None

            self._remappers[backend_name] = instance
            self._modules[backend_name] = module
            logger.info(f"Loaded remapper for backend '{backend_name}': {script_path}")
            return instance

        except Exception as e:
            logger.error(f"Error loading remapper for backend '{backend_name}': {e}")
            return None

    def get_remapper(self, backend_name: str) -> Optional[RequestRemapper]:
        return self._remappers.get(backend_name)

    def get_module(self, backend_name: str) -> Optional[object]:
        return self._modules.get(backend_name)

    async def call_remap(
        self, remapper: Optional[RequestRemapper], context: HookContext
    ) -> Optional[RemapResult]:
        """Call the remap method (supports both sync and async)."""
        if remapper is None:
            return None

        method = getattr(remapper, "remap", None)
        if method is None:
            return None

        try:
            result = method(context)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            logger.error(f"Remapper for backend '{context.backend_name}' raised error: {e}")
            return None
