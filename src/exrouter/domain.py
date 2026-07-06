"""Locking domain - a group of backends that can lock each other."""

from dataclasses import dataclass
from typing import Optional
import asyncio

from .backend import Backend


@dataclass
class LockState:
    """Track which backend holds a lock."""
    locked_by: str


class LockManager:
    """Manages locks within a single domain.
    
    This lock manager is domain-specific - it only knows about backends
    within one locking domain. Cross-domain locking is not possible.
    """

    def __init__(self, domain_backends: dict[str, Backend], timeout: int = 300):
        """Initialize lock manager for a domain.
        
        Args:
            domain_backends: Dict of backend_name -> Backend for THIS domain only
            timeout: Timeout in seconds when waiting for locks
        """
        self.domain_backends = domain_backends
        self.locks: dict[str, LockState] = {}
        self.holder_counts: dict[str, int] = {}  # target -> number of in-flight holders
        self.condition = asyncio.Condition()
        self.timeout = timeout

    async def acquire(self, backend_name: str, lock_targets: list[str]) -> bool:
        """Acquire locks on target backends.
        
        Args:
            backend_name: Name of backend acquiring locks
            lock_targets: List of backend names to lock (must be in same domain)
        
        Returns:
            True if locks acquired, False if timeout
        """
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
                    # Re-acquire by same backend (concurrent requests)
                    self.holder_counts[target] += 1
            return True

    async def _wait_until_free(self, backend_name: str, lock_targets: list[str]) -> None:
        """Wait until all targets are free or owned by this backend."""
        while any(
            target in self.locks and self.locks[target].locked_by != backend_name
            for target in lock_targets
        ):
            await self.condition.wait()

    async def release(self, backend_name: str, lock_targets: list[str]) -> None:
        """Release locks on target backends.
        
        Args:
            backend_name: Name of backend releasing locks
            lock_targets: List of backend names to unlock
        """
        async with self.condition:
            for target in lock_targets:
                if target in self.locks and self.locks[target].locked_by == backend_name:
                    self.holder_counts[target] -= 1
                    if self.holder_counts[target] <= 0:
                        del self.locks[target]
                        del self.holder_counts[target]
            self.condition.notify_all()

    def is_locked(self, backend_name: str) -> bool:
        """Check if a backend is currently locked."""
        return backend_name in self.locks


class LockDomain:
    """A locking domain containing backends that can lock each other.
    
    This class represents a top-level domain in the config hierarchy.
    It holds:
    - A set of backends that belong to this domain
    - A LockManager for coordinating locks within this domain
    
    Example usage:
        compute_domain = LockDomain(
            name="compute",
            backends={
                "llm": Backend(name="llm", domain_name="compute", ...),
                "stt": Backend(name="stt", domain_name="compute", ...),
            },
            timeout=300
        )
        
        # llm can lock stt, stt can lock llm
        # but neither can lock backends in other domains
    """
    
    def __init__(
        self,
        name: str,
        backends: dict[str, Backend],
        timeout: int = 300
    ):
        """Initialize a locking domain.
        
        Args:
            name: Domain name (e.g. "compute", "frontend", "audio")
            backends: Dict of backend_name -> Backend for this domain
            timeout: Lock timeout in seconds
        """
        self.name = name
        self.backends = backends
        self.lock_manager = LockManager(backends, timeout)
    
    def get_backend(self, name: str) -> Optional[Backend]:
        """Get a backend by name from this domain.
        
        Args:
            name: Backend name
        
        Returns:
            Backend if found, None otherwise
        """
        return self.backends.get(name)
    
    def backend_exists(self, name: str) -> bool:
        """Check if a backend exists in this domain."""
        return name in self.backends
