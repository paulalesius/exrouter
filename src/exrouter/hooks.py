"""Backend hooks - lifecycle callbacks for request processing."""

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import asyncio

logger = logging.getLogger("exrouter.hooks")


@dataclass
class HookContext:
    """Context passed to hook callbacks."""
    backend_name: str
    request_method: str
    request_path: str
    request_headers: dict[str, str]
    request_body: Optional[bytes]
    response_status: Optional[int] = None
    response_headers: Optional[dict[str, str]] = None
    response_body: Optional[bytes] = None
    error: Optional[str] = None


class BackendHook:
    """Base hook class that script must implement.
    
    All methods receive HookContext and can be sync or async.
    Return None or a dict to merge into request/response context.
    """
    
    def on_locks_acquired(self, context: HookContext) -> Optional[dict]:
        """Called after global locks are acquired, before request to backend."""
        pass
    
    def on_before_request(self, context: HookContext) -> Optional[dict]:
        """Called right before request is sent to backend."""
        pass
    
    def on_response(self, context: HookContext) -> Optional[dict]:
        """Called after response is received from backend."""
        pass
    
    def on_after_request(self, context: HookContext) -> Optional[dict]:
        """Called after request processing is complete, before locks are released."""
        pass
    
    def on_locks_released(self, context: HookContext) -> Optional[dict]:
        """Called after locks are released."""
        pass


class HookLoader:
    """Loads and manages hook scripts for backends."""
    
    def __init__(self):
        self._hooks: dict[str, BackendHook] = {}
        self._modules: dict[str, object] = {}  # Store module references
    
    def load_script(self, backend_name: str, script_path: str) -> Optional[BackendHook]:
        """Load a hook script from file path.
        
        Returns BackendHook instance if successful, None on error.
        """
        path = Path(script_path)
        if not path.exists():
            logger.error(f"Hook script not found for backend '{backend_name}': {script_path}")
            return None
        
        try:
            module_name = f"exrouter_hook_{backend_name}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to load spec for hook script: {script_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'BackendHook'):
                logger.error(f"Hook script must define 'BackendHook' class: {script_path}")
                return None
            
            hook_class = getattr(module, 'BackendHook')
            hook_instance = hook_class()
            
            # Verify hook has required methods (duck typing)
            required_methods = [
                'on_locks_acquired',
                'on_before_request',
                'on_response',
                'on_after_request',
                'on_locks_released'
            ]
            for method in required_methods:
                if not hasattr(hook_instance, method):
                    logger.error(f"BackendHook in {script_path} must implement '{method}' method")
                    return None
            
            self._hooks[backend_name] = hook_instance
            self._modules[backend_name] = module  # Store module reference
            logger.info(f"Loaded hook script for backend '{backend_name}': {script_path}")
            return hook_instance
            
        except Exception as e:
            logger.error(f"Error loading hook script for backend '{backend_name}': {e}")
            return None
    
    def get_hook(self, backend_name: str) -> Optional[BackendHook]:
        """Get loaded hook for backend."""
        return self._hooks.get(backend_name)
    
    def get_module(self, backend_name: str) -> Optional[object]:
        """Get loaded module for backend (for testing)."""
        return self._modules.get(backend_name)
    
    async def call_hook(self, hook: Optional[BackendHook], method_name: str, context: HookContext) -> Optional[dict]:
        """Call a hook method, handling both sync and async.
        
        Returns the hook's return value (dict or None).
        """
        if hook is None:
            return None
        
        method = getattr(hook, method_name, None)
        if method is None:
            return None
        
        try:
            result = method(context)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            logger.error(f"Hook {method_name} for backend '{context.backend_name}' raised error: {e}")
            return None
