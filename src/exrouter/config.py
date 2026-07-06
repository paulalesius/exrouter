"""Configuration loading from YAML with Pydantic validation."""

from pydantic import BaseModel, Field, AnyHttpUrl, field_validator, model_validator
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
    """Backend configuration from YAML."""
    url: AnyHttpUrl = Field(..., description="Backend server URL (must be http/https)")
    paths: list[str] = Field(default_factory=list, description="Path patterns this backend handles")
    locks: list[str] = Field(default_factory=list, description="Other backends to lock while processing")
    script: Optional[str] = Field(default=None, description="Path to Python hook script")
    remapper: Optional[str] = Field(default=None, description="Path to Python request remapper script")
    lifecycle: Optional[LifecycleConfig] = Field(
        default=None,
        description="Declarative lifecycle actions (systemd + shell + wait_for) on activate/deactivate. "
                    "Alternative to writing a full hook script for common resource management use cases."
    )

    @field_validator('paths', 'locks', mode='before')
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        return v or []


class GlobalLockConfig(BaseModel):
    """Global lock settings."""
    enabled: bool = Field(default=True, description="Enable global locking")
    timeout: int = Field(default=300, gt=0, description="Timeout in seconds when waiting for locks")


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=4001, ge=1, le=65535, description="Port to listen on")


class Config(BaseModel):
    """Full proxy configuration with validation."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    global_lock: GlobalLockConfig = Field(default_factory=GlobalLockConfig)

    @model_validator(mode='after')
    def validate_lock_targets_exist(self) -> "Config":
        """Ensure that all lock targets actually exist as backends."""
        backend_names = set(self.backends.keys())
        for name, backend in self.backends.items():
            for lock in backend.locks:
                if lock not in backend_names:
                    raise ValueError(
                        f"Backend '{name}' tries to lock '{lock}', "
                        f"but no backend named '{lock}' exists."
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
