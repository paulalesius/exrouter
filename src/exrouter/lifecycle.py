"""Declarative lifecycle actions executor (systemd, shell commands, Python scripts, wait conditions).

The single execution path for backend activation and deactivation. Per phase the
user declares any combination of systemd units, shell commands, and Python scripts
in the YAML `lifecycle:` block; they are executed in the fixed order
systemd -> shell -> python -> wait_for.

A Python script is a plain file that defines one callable named after the phase it
is used in: activate() for on_activate, deactivate() for on_deactivate. Sync and
async callables are both supported.
"""

import asyncio
import importlib.util
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

        # 3. Python script (first-class action type, same status as systemd/shell)
        if action_set.python:
            await self._run_python(action_set.python, phase, backend_name)

        # 4. Wait conditions (most useful after starting on activate)
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

    async def _run_python(self, script_path: str, phase: str, backend_name: str) -> None:
        """Import a user Python script and call its phase function.

        The script must define a callable named after the phase: activate() for
        the activate phase, deactivate() for the deactivate phase. Sync and async
        callables are both supported. Failures are logged, not raised, consistent
        with the shell and systemd actions: one bad script must not wedge the proxy.
        """
        logger.info(f"  [{backend_name}] python: {script_path} ({phase})")
        try:
            spec = importlib.util.spec_from_file_location(
                f"exrouter_lifecycle_{backend_name}_{phase}", script_path
            )
            if spec is None or spec.loader is None:
                logger.error(f"  ✗ Failed to load lifecycle script spec: {script_path}")
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, phase, None)
            if not callable(func):
                logger.error(
                    f"  ✗ Lifecycle script {script_path} must define a callable "
                    f"'{phase}()' (sync or async)"
                )
                return
            result = func()
            if asyncio.iscoroutine(result):
                await result
            logger.info(f"  ✓ python script succeeded")
        except Exception as e:
            logger.error(f"  ✗ Error running lifecycle script '{script_path}': {e}")

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
