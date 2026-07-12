"""WebSocket connection pooling to prevent file descriptor leaks.

Architecture:
- Pool caches backend WebSocket connections by (backend_name, ws_url)
- Connections are reused for concurrent requests to same backend
- Idle connections are closed after timeout
- Ensures proper cleanup on disconnect
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import websockets
from websockets.typing import Subprotocol
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger("exrouter")


@dataclass
class WebSocketConnection:
    """Cached WebSocket connection with metadata."""
    ws: ClientConnection
    created_at: float
    last_used: float
    active_users: int = 1  # ref count for concurrent usage
    subprotocol: Optional[Subprotocol] = None


class WebSocketPool:
    """Connection pool for WebSocket backends.
    
    Prevents file descriptor leaks by:
    1. Reusing connections for same backend URL
    2. Closing idle connections after timeout
    3. Proper cleanup on disconnect
    """
    
    def __init__(self, max_idle_seconds: int = 300, max_connections: int = 100):
        self.max_idle_seconds = max_idle_seconds
        self.max_connections = max_connections
        
        # Pool: (backend_name, ws_url) -> list[WebSocketConnection]
        self._pool: dict[tuple[str, str], list[WebSocketConnection]] = defaultdict(list)
        
        # Lock for pool operations
        self._lock = asyncio.Lock()
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = True
        
        # Metrics
        self._created_count = 0
        self._reused_count = 0
        self._closed_count = 0
    
    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"WebSocketPool started (max_idle={self.max_idle_seconds}s, max_conn={self.max_connections})")
    
    async def stop(self) -> None:
        """Stop cleanup and close all connections."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all pooled connections
        async with self._lock:
            for key, connections in list(self._pool.items()):
                for conn in connections:
                    await self._close_connection(conn)
            self._pool.clear()
        
        logger.info(f"WebSocketPool stopped (created={self._created_count}, reused={self._reused_count}, closed={self._closed_count})")
    
    async def get_connection(
        self,
        backend_name: str,
        ws_url: str,
        additional_headers: dict
    ) -> ClientConnection:
        """Get a WebSocket connection from pool or create new one.
        
        Returns either:
        - Reused connection from pool (if available)
        - New connection (if pool empty or full)
        """
        key = (backend_name, ws_url)
        
        async with self._lock:
            # Try to get existing connection
            connections = self._pool[key]
            for conn in connections:
                if conn.active_users > 0 and conn.ws.close_code is None:
                    conn.active_users += 1
                    conn.last_used = asyncio.get_event_loop().time()
                    self._reused_count += 1
                    logger.debug(f"Reused WebSocket connection for {backend_name} (active_users={conn.active_users})")
                    return conn.ws
            
            # Check if we can create new connection
            total_connections = sum(len(c) for c in self._pool.values())
            if total_connections >= self.max_connections:
                logger.warning(f"WebSocketPool max connections reached ({self.max_connections}), closing idle")
                await self._force_cleanup()
            
            # Create new connection
            try:
                ws = await websockets.connect(ws_url, additional_headers=additional_headers)
                conn = WebSocketConnection(
                    ws=ws,
                    created_at=asyncio.get_event_loop().time(),
                    last_used=asyncio.get_event_loop().time(),
                    active_users=1
                )
                self._pool[key].append(conn)
                self._created_count += 1
                logger.info(f"Created new WebSocket connection for {backend_name} (total={total_connections + 1})")
                return ws
                
            except Exception as e:
                logger.error(f"Failed to create WebSocket connection for {backend_name}: {e}")
                raise
    
    async def release_connection(
        self,
        backend_name: str,
        ws_url: str,
        ws: ClientConnection
    ) -> None:
        """Release a WebSocket connection back to pool.
        
        Decrements ref count. If 0, closes connection (no longer reuse).
        """
        key = (backend_name, ws_url)
        
        async with self._lock:
            connections = self._pool[key]
            for conn in connections:
                if conn.ws is ws:
                    conn.active_users -= 1
                    conn.last_used = asyncio.get_event_loop().time()
                    
                    if conn.active_users <= 0:
                        # Close the connection when ref count reaches 0
                        await self._close_connection(conn)
                        # Remove from pool
                        connections.pop(connections.index(conn))
                        if not connections:
                            self._pool.pop(key, None)
                        logger.debug(f"Closed WebSocket connection for {backend_name} (ref count=0)")
                    
                    return
    
    async def close_connection(
        self,
        backend_name: str,
        ws_url: str,
        ws: ClientConnection
    ) -> None:
        """Close and remove a WebSocket connection from pool."""
        key = (backend_name, ws_url)
        
        async with self._lock:
            connections = self._pool[key]
            for i, conn in enumerate(connections):
                if conn.ws is ws:
                    await self._close_connection(conn)
                    connections.pop(i)
                    if not connections:
                        self._pool.pop(key, None)
                    return
    
    async def _close_connection(self, conn: WebSocketConnection) -> None:
        """Close a single WebSocket connection."""
        try:
            if conn.ws.close_code is None:
                await conn.ws.close()
                self._closed_count += 1
                logger.debug(f"Closed WebSocket connection (created={self._created_count}, closed={self._closed_count})")
        except Exception as e:
            logger.debug(f"Error closing WebSocket connection: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background loop to close idle connections."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocketPool cleanup error: {e}")
    
    async def _cleanup_idle(self) -> None:
        """Close connections that have been idle too long."""
        now = asyncio.get_event_loop().time()
        closed_count = 0
        
        async with self._lock:
            for key in list(self._pool.keys()):
                connections = self._pool[key]
                still_active = []
                
                for conn in connections:
                    idle_time = now - conn.last_used
                    if idle_time > self.max_idle_seconds and conn.active_users == 0:
                        await self._close_connection(conn)
                        closed_count += 1
                    else:
                        still_active.append(conn)
                
                if still_active:
                    self._pool[key] = still_active
                else:
                    self._pool.pop(key, None)
        
        if closed_count > 0:
            logger.info(f"WebSocketPool cleanup: closed {closed_count} idle connections")
    
    async def _force_cleanup(self) -> None:
        """Force close idle connections when pool is full."""
        await self._cleanup_idle()
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        total_pooled = sum(len(c) for c in self._pool.values())
        active_pooled = sum(
            sum(1 for c in conns if c.active_users > 0)
            for conns in self._pool.values()
        )
        
        return {
            "created": self._created_count,
            "reused": self._reused_count,
            "closed": self._closed_count,
            "pooled": total_pooled,
            "active": active_pooled,
            "idle": total_pooled - active_pooled
        }
