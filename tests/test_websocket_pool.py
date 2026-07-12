"""Integration tests for WebSocketPool with websockets 15.x API.

Verifies:
- ClientConnection compatibility (no .closed attribute crash)
- Connection creation, reuse, and release
- Idle cleanup
- Pool stats tracking
"""

import asyncio
import pytest
from exrouter.websocket_pool import WebSocketPool, WebSocketConnection


@pytest.fixture
def pool():
    return WebSocketPool(max_idle_seconds=1, max_connections=10)


@pytest.mark.asyncio
async def test_pool_create_and_close(pool):
    """Pool starts and stops cleanly without errors."""
    await pool.start()
    await asyncio.sleep(0.1)  # Let cleanup loop start
    await pool.stop()
    
    stats = pool.get_stats()
    assert stats["created"] == 0
    assert stats["closed"] == 0


@pytest.mark.asyncio
async def test_pool_close_connection_checks_close_code():
    """Verify _close_connection uses close_code instead of .closed (websockets 15.x fix).
    
    Before fix: AttributeError: 'ClientConnection' object has no attribute 'closed'
    After fix: Uses conn.ws.close_code is None
    """
    pool = WebSocketPool()
    
    # Create a mock WebSocketConnection with a mock ws that has close_code but no .closed
    class MockWS:
        close_code = None  # Open connection (websockets 15.x style)
        closed = None  # No .closed attribute in 15.x
        
        def __getattribute__(self, name):
            if name == "closed":
                raise AttributeError("'MockWS' object has no attribute 'closed'")
            return object.__getattribute__(self, name)
        
        async def close(self):
            self.close_code = 1000
    
    mock_ws = MockWS()
    conn = WebSocketConnection(
        ws=mock_ws,
        created_at=0,
        last_used=0,
        active_users=0
    )
    
    # This should NOT raise AttributeError about .closed
    await pool._close_connection(conn)
    
    assert mock_ws.close_code == 1000
    stats = pool.get_stats()
    assert stats["closed"] == 1


@pytest.mark.asyncio
async def test_pool_get_connection_avoids_closed_attribute():
    """Verify get_connection checks close_code instead of .closed when scanning pool.
    
    Before fix: Line 102 crashes with 'ClientConnection' object has no attribute 'closed'
    After fix: Uses conn.ws.close_code is None
    """
    pool = WebSocketPool()
    
    # Create a mock ws that mimics websockets 15.x ClientConnection
    class MockWS15:
        close_code = None  # Connection is open
        closed = None
        
        def __getattribute__(self, name):
            if name == "closed":
                raise AttributeError("'ClientConnection' object has no attribute 'closed'")
            return object.__getattribute__(self, name)
        
        async def close(self):
            self.close_code = 1000
    
    # Manually seed the pool with a mock connection
    mock_ws = MockWS15()
    conn = WebSocketConnection(
        ws=mock_ws,
        created_at=0,
        last_used=0,
        active_users=1
    )
    pool._pool[("test-backend", "ws://test")].append(conn)
    
    # This should find and reuse the connection without crashing on .closed
    result = await pool.get_connection("test-backend", "ws://test", {})
    
    assert result is mock_ws
    assert conn.active_users == 2
    stats = pool.get_stats()
    assert stats["reused"] == 1


@pytest.mark.asyncio
async def test_pool_rejects_closed_connections():
    """Pool should skip connections where close_code is not None (already closed)."""
    pool = WebSocketPool()
    
    class MockWS15:
        close_code = 1000  # Already closed
        closed = None
        
        def __getattribute__(self, name):
            if name == "closed":
                raise AttributeError("'ClientConnection' object has no attribute 'closed'")
            return object.__getattribute__(self, name)
        
        async def close(self):
            pass
    
    mock_ws = MockWS15()
    conn = WebSocketConnection(
        ws=mock_ws,
        created_at=0,
        last_used=0,
        active_users=1
    )
    pool._pool[("test-backend", "ws://test")].append(conn)
    
    # Since the only pooled connection is closed, get_connection should try to create a new one
    # which will fail since ws://test isn't real — but it should NOT crash on .closed
    with pytest.raises(Exception):  # Connection error is expected, AttributeError is not
        try:
            await pool.get_connection("test-backend", "ws://test", {})
        except AttributeError as e:
            if "closed" in str(e):
                pytest.fail(f"Crashed on .closed attribute: {e}")
            raise


@pytest.mark.asyncio
async def test_pool_release_decrements_ref_count():
    """Release should decrement active_users and close at zero."""
    pool = WebSocketPool()
    
    class MockWS15:
        close_code = None
        closed = None
        
        def __getattribute__(self, name):
            if name == "closed":
                raise AttributeError("'ClientConnection' object has no attribute 'closed'")
            return object.__getattribute__(self, name)
        
        async def close(self):
            self.close_code = 1000
    
    mock_ws = MockWS15()
    conn = WebSocketConnection(
        ws=mock_ws,
        created_at=0,
        last_used=0,
        active_users=2
    )
    pool._pool[("test-backend", "ws://test")].append(conn)
    
    # First release: ref count 2 -> 1, should NOT close
    await pool.release_connection("test-backend", "ws://test", mock_ws)
    assert conn.active_users == 1
    assert mock_ws.close_code is None  # Still open
    
    # Second release: ref count 1 -> 0, should close
    await pool.release_connection("test-backend", "ws://test", mock_ws)
    assert conn.active_users == 0
    assert mock_ws.close_code == 1000  # Now closed


@pytest.mark.asyncio
async def test_websockets_version_compatibility():
    """Confirm websockets 15.x API is in use (no .closed, has .close_code)."""
    import websockets
    from websockets.asyncio.client import ClientConnection
    
    # websockets 15.x: ClientConnection has close_code, no closed
    assert hasattr(ClientConnection, "close_code")
    
    # Version check (15.0+ is the new API)
    version = websockets.__version__
    major = int(version.split(".")[0])
    assert major >= 15, f"Expected websockets >= 15, got {version}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
