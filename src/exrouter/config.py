"""Configuration loading from YAML with Pydantic validation."""

import logging
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
    """Set of actions to run on an activate or deactivate phase.

    systemd, shell and python are interchangeable choices for the same job:
    declare any combination per phase and the executor runs them in the fixed
    order systemd -> shell -> python -> wait_for.
    """
    systemd: Optional[SystemdConfig] = Field(default=None, description="Systemd start/stop actions")
    shell: list[str] = Field(default_factory=list, description="Shell commands to execute (each via /bin/sh -c)")
    python: Optional[str] = Field(default=None, description="Path to a Python script for this phase. Must define a callable named after the phase: activate() or deactivate() (sync or async)")
    wait_for: list[WaitForConfig] = Field(default_factory=list, description="Wait conditions after actions (usually on activate)")


class LifecycleConfig(BaseModel):
    """The single activation/deactivation mechanism for a backend's own service.

    on_activate runs the first time a request reaches the backend after it was
    deactivated; on_deactivate runs when another backend that locks this one
    activates and stops it. Ideal for VRAM/resource management: start and stop
    services with systemd units, shell commands, or Python scripts.
    """
    on_activate: Optional[ActionSet] = Field(default=None, description="Actions when this backend activates")
    on_deactivate: Optional[ActionSet] = Field(default=None, description="Actions when this backend is deactivated by a conflicting backend")


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
    script: Optional[str] = Field(default=None, description="Path to Python hook script for request-level callbacks (not activation/deactivation: that is lifecycle:)")
    remapper: Optional[str] = Field(default=None, description="Path to Python request remapper script")
    lifecycle: Optional[LifecycleConfig] = Field(
        default=None,
        description="The single activation/deactivation mechanism for this backend's own service. "
                    "Per phase: systemd units, shell commands, Python scripts, and wait conditions."
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
    
    When serialized to JSON, this becomes:
    {"domains": {"compute": {"backends": {"llm": {...}, "stt": {...}}}}}
    """
    
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    
    @model_validator(mode='before')
    @classmethod
    def parse_backends_from_yaml(cls, data: Any) -> Any:
        """Parse backends from YAML structure.
        
        YAML structure:
        compute:
          llm:
            url: ...
          stt:
            url: ...
        
        This validator extracts all dict fields (backend configs) and puts them
        into the 'backends' dict.
        """
        if not isinstance(data, dict):
            return data
        
        # Extract backend configs (all dict fields except 'backends')
        backend_configs = {}
        for key, value in data.items():
            if key == 'backends':
                continue
            if isinstance(value, dict):
                backend_configs[key] = BackendConfig.model_validate(value)
        
        # Return data with 'backends' field populated
        result = dict(data)
        if backend_configs:
            result['backends'] = backend_configs
        
        return result
    
    def model_dump(self, *args, **kwargs) -> dict:
        """Custom serialization to avoid nested 'backends' key.
        
        Returns: {"backends": {...}} instead of {"name": "...", "backends": {...}}
        """
        return {"backends": self.backends}


class GlobalLockConfig(BaseModel):
    """Global lock settings."""
    timeout: int = Field(default=300, gt=0, description="Timeout in seconds when waiting for locks")


class ServerConfig(BaseModel):
    """Server configuration.
    
    Configured via environment variables: EXROUTER_HOST, EXROUTER_PORT
    
    Defaults: host=0.0.0.0, port=4001
    """
    model_config = ConfigDict(extra="allow")
    
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=4001, ge=1, le=65535, description="Port to listen on")
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create ServerConfig from environment variables.
        
        Reads EXROUTER_HOST and EXROUTER_PORT if set.
        Falls back to defaults if not set.
        """
        import os
        
        host = os.environ.get("EXROUTER_HOST", "0.0.0.0")
        port_str = os.environ.get("EXROUTER_PORT", "4001")
        
        try:
            port = int(port_str)
        except ValueError:
            logger = logging.getLogger("exrouter")
            logger.warning(f"Invalid EXROUTER_PORT '{port_str}', using default 4001")
            port = 4001
        
        return cls(host=host, port=port)


class Config(BaseModel):
    """Full proxy configuration with validation.
    
    The domains field is hierarchical:
    - Top level: domain names (e.g. "compute", "frontend", "audio")
    - Second level: backend configs within each domain
    
    Locks are validated to only reference backends within the same domain.
    """
    server: ServerConfig = Field(default_factory=ServerConfig)
    domains: dict[str, LockDomain] = Field(
        ...,
        alias="backends",
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
        for domain_name, domain in self.domains.items():
            domain_backend_names = set(domain.backends.keys())
            
            for backend_name, backend in domain.backends.items():
                for lock_target in backend.locks:
                    # Check if lock target exists ANYWHERE
                    found_anywhere = False
                    found_in_same_domain = False
                    
                    for other_domain_name, other_domain in self.domains.items():
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
                        for dn, d in self.domains.items():
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
