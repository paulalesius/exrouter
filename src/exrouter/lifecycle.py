"""Declarative lifecycle actions executor (systemd, shell commands, wait conditions).

This provides a native, YAML-driven way to manage backend services (e.g. start/stop
systemd units when a backend goes from idle → active or active → idle) without
requiring users to write Python hook scripts for the common VRAM/resource switching
pattern.

It is executed at the same points as the existing hook on_backend_activated /
on_backend_deactivated callbacks, and runs *in addition to* any hook script if both
are configured.
"""

import asyncio
import logging
import time
from typing import Optional

from .config import ActionSet, WaitForConfig

logger = logging.getLogger("exrouter.lifecycle")


class LifecycleExecutor:
    """Executes on_activate / on_deactivate action sets for a backend."""

    async def execute(
        self, backend_name: str, action_set: Optional[ActionSet], *, is_activate: bool
    ) -> None:
        """Run the actions for activate or deactivate phase."""
        if action_set is None:
            return

        phase = "activate" if is_activate else "deactivate"
        logger.info(f"[{backend_name}] Executing declarative lifecycle {phase} actions")

        # 1. Systemd actions first (stops before starts is usually what we want)
        if action_set.systemd:
            for unit in action_set.systemd.stop:
                await self._systemctl("stop", unit, backend_name)
            for unit in action_set.systemd.start:
                await self._systemctl("start", unit, backend_name)

        # 2. Arbitrary shell commands
        for cmd in action_set.shell:
            await self._run_shell(cmd, backend_name)

        # 3. Wait conditions (most useful after starting on activate)
        for wait in action_set.wait_for:
            if wait.type == "port":
                await self._wait_for_port(wait.host, wait.port, wait.timeout, backend_name)

    async def _systemctl(self, action: str, unit: str, backend_name: str) -> None:
        """Run systemctl start/stop asynchronously."""
        logger.info(f"  [{backend_name}] systemctl {action} {unit}")
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl",
                action,
                unit,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            if proc.returncode == 0 or (action == "stop" and proc.returncode in (3, 4, 5)):
                if proc.returncode == 0:
                    logger.info(f"  ✓ {unit} {action} succeeded")
                else:
                    logger.info(f"  ✓ {unit} {action} succeeded (already stopped / not loaded)")
            else:
                err = stderr.decode(errors="replace").strip()
                logger.warning(f"  ⚠ {unit} {action} failed (rc={proc.returncode}): {err[:300]}")
        except asyncio.TimeoutError:
            logger.warning(f"  ⚠ Timeout while running systemctl {action} {unit}")
        except FileNotFoundError:
            logger.error("  ✗ 'systemctl' command not found (is systemd available?)")
        except Exception as e:
            logger.error(f"  ✗ Error running systemctl {action} {unit}: {e}")

    async def _run_shell(self, cmd: str, backend_name: str) -> None:
        """Run a shell command via /bin/sh -c."""
        logger.info(f"  [{backend_name}] shell: {cmd}")
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
            if proc.returncode == 0:
                logger.info(f"  ✓ shell command succeeded")
            else:
                err = stderr.decode(errors="replace").strip()[:300]
                logger.warning(f"  ⚠ shell command failed (rc={proc.returncode}): {err}")
        except asyncio.TimeoutError:
            logger.warning(f"  ⚠ Timeout running shell command: {cmd[:80]}")
        except Exception as e:
            logger.error(f"  ✗ Error running shell '{cmd}': {e}")

    async def _wait_for_port(
        self, host: str, port: int, timeout: int, backend_name: str
    ) -> None:
        """Wait until a TCP port accepts connections (async version of the hook logic)."""
        logger.info(f"  [{backend_name}] waiting for {host}:{port} (timeout={timeout}s)...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), timeout=2.0
                )
                writer.close()
                await writer.wait_closed()
                logger.info(f"  ✓ {host}:{port} is ready")
                return
            except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                await asyncio.sleep(1.0)
        logger.warning(f"  ⚠ Timeout waiting for {host}:{port} to become ready")
