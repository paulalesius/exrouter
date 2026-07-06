"""Hook script for managing systemd services based on backend lifecycle.

This hook ensures only ONE of the heavy services runs at a time by using
BLProxy's backend activation lifecycle (not per-request).

- When the first request arrives for a backend after idle → activate it
  (start its service, stop conflicting ones)
- When the LAST in-flight request finishes → deactivate (stop service to free resources)

This is much more efficient than running systemctl on every single request/path.

Usage in config.yaml (both backends can share the same hook file):
  backends:
    llm:
      url: http://127.0.0.1:8080
      paths:
        - /v1/chat/completions
        - /v1/models
        # ... etc
      script: /src/exrouter/samples/hook.py
      locks: [stt_custom]   # optional but recommended for serialization
    
    stt_custom:
      url: http://127.0.0.1:8091
      paths:
        - /transcribe
      script: /src/exrouter/samples/hook.py
      locks: [llm]
"""

import subprocess
import time
import socket
from exrouter.hooks import BackendHook, HookContext


class BackendHook:
    """Hook that starts/stops systemd services based on backend activate/deactivate lifecycle."""
    
    def on_backend_activated(self, context: HookContext) -> None:
        backend = context.backend_name

        if backend == "stt_custom":
            print("[HOOK] stt_custom ACTIVATED → starting stt_custom.service, stopping llm.service")
            self._switch_services(active="stt-custom.target", inactive="llama-server.service")
            self._wait_for_port("127.0.0.1", 8091, timeout=20)

        elif backend == "llm":
            print("[HOOK] llm ACTIVATED → starting llama-server.service")
            self._switch_services(active="llama-server.service", inactive="stt-custom.target")
            self._wait_for_port("127.0.0.1", 8080, timeout=20)

    def _switch_services(self, active: str, inactive: str) -> None:
        self._stop_service(inactive)
        self._start_service(active)

    def _wait_for_port(self, host: str, port: int, timeout: int = 120) -> None:
        """Väntar tills porten faktiskt tar emot connections."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection((host, port), timeout=2):
                    print(f"  ✓ {host}:{port} is ready")
                    return
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(1.5)
        print(f"  ⚠ Timeout waiting for {host}:{port} to become ready")

    def _start_service(self, service: str) -> None:
        """Start a systemd service (idempotent if already running)."""
        try:
            print(f"  → Starting {service}...")
            result = subprocess.run(
                ["systemctl", "start", service],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"  ✓ {service} started")
            else:
                print(f"  ⚠ {service} start error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠ Timeout starting {service}")
        except Exception as e:
            print(f"  ✗ Error starting {service}: {e}")
    
    def _stop_service(self, service: str) -> None:
        """Stop a systemd service (idempotent if already stopped)."""
        try:
            print(f"  → Stopping {service}...")
            result = subprocess.run(
                ["systemctl", "stop", service],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"  ✓ {service} stopped")
            else:
                print(f"  ⚠ {service} stop error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠ Timeout stopping {service}")
        except Exception as e:
            print(f"  ✗ Error stopping {service}: {e}")

    # Legacy per-request hooks kept for compatibility but left empty.
    # All service logic now lives in the activation lifecycle hooks above.
    def on_locks_acquired(self, context: HookContext) -> None:
        pass
    
    def on_before_request(self, context: HookContext) -> None:
        pass
    
    def on_response(self, context: HookContext) -> None:
        pass
    
    def on_after_request(self, context: HookContext) -> None:
        pass
    
    def on_locks_released(self, context: HookContext) -> None:
        pass
