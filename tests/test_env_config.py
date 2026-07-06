#!/usr/bin/env python3
"""Test ENV variable configuration for EXRouter."""

import os
import sys
sys.path.insert(0, '/src/exrouter/src')

from exrouter.config import Config, ServerConfig


def test_server_config_from_env():
    """Test ServerConfig.from_env() reads EXROUTER_HOST and EXROUTER_PORT."""
    os.environ["EXROUTER_HOST"] = "127.0.0.1"
    os.environ["EXROUTER_PORT"] = "9999"
    
    config = ServerConfig.from_env()
    assert config.host == "127.0.0.1"
    assert config.port == 9999
    
    del os.environ["EXROUTER_HOST"]
    del os.environ["EXROUTER_PORT"]


def test_server_config_from_env_defaults():
    """Test ServerConfig.from_env() uses defaults when ENV not set."""
    if "EXROUTER_HOST" in os.environ:
        del os.environ["EXROUTER_HOST"]
    if "EXROUTER_PORT" in os.environ:
        del os.environ["EXROUTER_PORT"]
    
    config = ServerConfig.from_env()
    assert config.host == "0.0.0.0"
    assert config.port == 4001


def test_invalid_port_falls_back_to_default():
    """Test invalid EXROUTER_PORT falls back to default 4001."""
    os.environ["EXROUTER_HOST"] = "127.0.0.1"
    os.environ["EXROUTER_PORT"] = "not_a_number"
    
    config = ServerConfig.from_env()
    assert config.host == "127.0.0.1"
    assert config.port == 4001  # Default
    
    del os.environ["EXROUTER_HOST"]
    del os.environ["EXROUTER_PORT"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
