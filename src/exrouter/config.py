"""Configuration loading from YAML with Pydantic validation."""

from pydantic import BaseModel, Field, AnyHttpUrl, field_validator, model_validator, ConfigDict
from typing import Any, Literal, Optional
import yaml


class WaitForConfig(BaseModel):
    """Configuration for waiting after starting services (e.g. port ready)."""
    type: Literal["port"] = Field(default="port", description="Wait type (only 'port' supported currently)")
    host: str = Field(default="127.0.0.1", description="Host to connect to")
    port: int = Field(..., ge=1, le=65535, description="TCP port to wait for")
    timeout: int = Field(default=30, gt=0, description="Max seconds to wait")


class SystemdConfig(BaseModel):
    """Systemd units to stop/start as part of lifecycle actions."""
    stop: list[str] = Field(default_factory=list, description="Units/targets to stop (e.g. conflicting heavy service)")
    start: list[str] = Field(default_factory=list, description="Units/targets to start (e.g. this backend's service)")


class ActionSet(BaseModel):
    """Set of actions to run on activate or deactivate."""
    systemd: Optional[SystemdConfig] = Field(default=None, description="Systemd start/stop actions")
    shell: list[str] = Field(default_factory=list, description="Shell commands to execute (each via bash -c)")
    wait_for: list[WaitForConfig] = Field(default_factory=list, description="Wait conditions after actions (usually on activate)")


class LifecycleConfig(BaseModel):
    """Declarative lifecycle management (systemd/shell/wait) as alternative to hook scripts.

    Runs on backend activation (first in-flight request after idle) and deactivation
    (last in-flight request completes). This is ideal for VRAM/resource management
    without writing Python hook code.
    """
    on_activate: Optional[ActionSet] = Field(default=None, description="Actions when this backend becomes active")
    on_deactivate: Optional[ActionSet] = Field(default=None, description="Actions when this backend becomes idle")


class BackendConfig(BaseModel):
    """Backend configuration from YAML.
    
    Note: This backend config does NOT include domain_name here - that is set
    by the LockDomain it belongs to. This keeps the config flat and simple.
    """
    url: AnyHttpUrl = Field(..., description="Backend server URL (must be http/https)")
    paths: list[str] = Field(
        default_factory=list,
        description="Path patterns this backend handles. "
                    "Use ['*'] or ['/'] (special-cased to match everything) when combining with domain: to own an entire virtual host."
    )
    locks: list[str] = Field(default_factory=list, description="Other backends to lock while processing (must be in same domain)")
    domain: list[str] = Field(
        default_factory=list,
        description="Domain / Host header patterns this backend handles (supports fnmatch wildcards like '*.example.com'). "
                    "If specified, BOTH domain and path must match. "
                    "Combine with paths: ['*'] or paths: ['/'] to give one backend full ownership of that domain."
    )
    script: Optional[str] = Field(default=None, description="Path to Python hook script")
    remapper: Optional[str] = Field(default=None, description="Path to Python request remapper script")
    lifecycle: Optional[LifecycleConfig] = Field(
        default=None,
        description="Declarative lifecycle actions (systemd + shell + wait_for) on activate/deactivate. "
                    "Alternative to writing a full hook script for common resource management use cases."
    )

    @field_validator('paths', 'locks', 'domain', mode='before')
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        return v or []


class LockDomain(BaseModel):
    """A locking domain - a group of backends that can lock each other.
    
    Backends in one domain can only lock other backends in the same domain.
    This prevents cross-domain deadlocks and enables independent resource management.
    
    In YAML, this appears as:
    backends:
      compute:  # ← domain name
        llm:    # ← backend name (directly under domain)
          url: http://localhost:8080
        stt:
          url: http://localhost:7301
    """
    
    model_config = ConfigDict(extra="allow")
    
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    
    def __init__(self, **data):
        """Parse backends from extra fields.
        
        Any field that is a dict is treated as a BackendConfig.
        """
        # Extract known fields
        known_fields = {'backends'}
        extra_data = {k: v for k, v in data.items() if k not in known_fields}
        
        # Initialize with explicit backends if provided
        backends_dict = data.get('backends', {})
        
        # Add extra fields as backends
        for name, config_data in extra_data.items():
            if isinstance(config_data, dict):
                backends_dict[name] = config_data
        
        super().__init__(backends=backends_dict)


class GlobalLockConfig(BaseModel):
    """Global lock settings."""
    enabled: bool = Field(default=True, description="Enable global locking")
    timeout: int = Field(default=300, gt=0, description="Timeout in seconds when waiting for locks")


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=4001, ge=1, le=65535, description="Port to listen on")


class Config(BaseModel):
    """Full proxy configuration with validation.
    
    The backends field is now hierarchical:
    - Top level: domain names (e.g. "compute", "frontend", "audio")
    - Second level: backend configs within each domain
    
    Locks are validated to only reference backends within the same domain.
    """
    server: ServerConfig = Field(default_factory=ServerConfig)
    backends: dict[str, LockDomain] = Field(
        ...,
        description="Locking domains. Each domain contains backends that can lock each other."
    )
    global_lock: GlobalLockConfig = Field(default_factory=GlobalLockConfig)

    @model_validator(mode='after')
    def validate_lock_targets_exist(self) -> "Config":
        """Ensure that all lock targets exist within the same domain.
        
        Validates:
        1. Each backend's locks reference backends that exist
        2. Lock targets are in the SAME domain (cross-domain locks not allowed)
        """
        for domain_name, domain in self.backends.items():
            domain_backend_names = set(domain.backends.keys())
            
            for backend_name, backend in domain.backends.items():
                for lock_target in backend.locks:
                    # Check if lock target exists ANYWHERE
                    found_anywhere = False
                    found_in_same_domain = False
                    
                    for other_domain_name, other_domain in self.backends.items():
                        if lock_target in other_domain.backends:
                            found_anywhere = True
                            if other_domain_name == domain_name:
                                found_in_same_domain = True
                    
                    if not found_anywhere:
                        raise ValueError(
                            f"Backend '{backend_name}' in domain '{domain_name}' "
                            f"tries to lock '{lock_target}', but no backend named '{lock_target}' exists anywhere."
                        )
                    
                    if not found_in_same_domain:
                        # Find which domain it's in for better error message
                        lock_domain = None
                        for dn, d in self.backends.items():
                            if lock_target in d.backends:
                                lock_domain = dn
                                break
                        raise ValueError(
                            f"Backend '{backend_name}' in domain '{domain_name}' "
                            f"tries to lock '{lock_target}', but '{lock_target}' is in domain '{lock_domain}'. "
                            f"Locks can only target backends within the same domain."
                        )
        
        return self

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load config from YAML file with validation."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Load config from dict with validation."""
        return cls.model_validate(data or {})
