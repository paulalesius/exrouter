"""Backend component - represents a single backend with paths and locks."""

from dataclasses import dataclass, field
from typing import Optional
import fnmatch


@dataclass
class Backend:
    """A backend component.
    
    Holds:
    - name: Backend identifier (unique within its domain)
    - domain_name: Name of the locking domain this backend belongs to
    - url: Backend server URL
    - paths: List of path patterns this backend handles
    - locks: List of other backend names to lock while processing (same domain only)
    - domain: List of domain/Host patterns (optional). If non-empty, BOTH domain and path must match.
    - script: Optional path to hook script
    - remapper: Optional path to request remapper script
    """
    name: str
    domain_name: str  # ← NEW: which locking domain this backend belongs to
    url: str
    paths: list[str]
    locks: list[str]
    domain: list[str] = field(default_factory=list)
    script: Optional[str] = None
    remapper: Optional[str] = None

    def matches_path(self, path: str) -> bool:
        """Check if this backend handles the given path.

        Special case: if "/" is listed in paths (common when using domain: for full
        virtual hosting of a web UI), it matches *any* path under that domain.
        This makes configs like paths: ["/"] + domain: ["foo.example.com"] work
        as "this entire domain belongs to this backend".
        """
        if not self.paths:
            return False
        for pattern in self.paths:
            if pattern == "/":
                return True
            if "*" in pattern:
                if fnmatch.fnmatch(path, pattern):
                    return True
            else:
                if path == pattern:
                    return True
        return False

    def matches_domain(self, host: str) -> bool:
        """Check if this backend handles the given Host header (domain matching).
        
        Supports exact match and fnmatch wildcards (e.g. '*.example.com').
        Case-insensitive. Port in Host header is ignored.
        """
        if not self.domain:
            return False
        # Normalize: strip port, lower case
        host = host.split(":")[0].lower().strip()
        for pattern in self.domain:
            p = pattern.lower().strip()
            if not p:
                continue
            if "*" in p:
                if fnmatch.fnmatch(host, p):
                    return True
            else:
                if host == p:
                    return True
        return False

    def get_lock_targets(self, domain_backends: dict[str, "Backend"]) -> list[str]:
        """Get list of backends this backend locks (within the same domain).
        
        Args:
            domain_backends: Dict of backend_name -> Backend for THIS domain only.
                            Backends from other domains are not returned.
        
        Returns:
            List of backend names that exist in the same domain and are in locks list.
        """
        return [name for name in self.locks if name in domain_backends]
