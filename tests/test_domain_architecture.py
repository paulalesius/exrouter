"""Tests for the new hierarchical domain architecture.

Tests:
- LockDomain parsing from YAML
- Backend domain assignment
- Cross-domain lock validation
- LockManager isolation per domain
"""

import pytest
from exrouter.config import Config, LockDomain
from exrouter.proxy import LockProxy


def test_domain_parsing():
    """Test that domains are correctly parsed from YAML config."""
    config_yaml = """
backends:
  compute:
    llm:
      url: http://localhost:8080
      paths: [/v1/chat]
      locks: [stt]
    stt:
      url: http://localhost:7301
      paths: [/transcribe]
      locks: []
  
  frontend:
    webui:
      url: http://localhost:9090
      paths: [/]
      locks: []

global_lock:
  enabled: true
"""
    config = Config.from_dict(__import__('yaml').safe_load(config_yaml))
    
    # Check domains exist
    assert 'compute' in config.domains
    assert 'frontend' in config.domains
    
    # Check backends are in correct domains
    assert 'llm' in config.domains['compute'].backends
    assert 'stt' in config.domains['compute'].backends
    assert 'webui' in config.domains['frontend'].backends


def test_backend_domain_assignment():
    """Test that Backends get correct domain_name from LockProxy."""
    config_yaml = """
backends:
  compute:
    llm:
      url: http://localhost:8080
      paths: [/v1/chat]
      locks: [stt]
    stt:
      url: http://localhost:7301
      paths: [/transcribe]
      locks: []

global_lock:
  enabled: true
"""
    config = Config.from_dict(__import__('yaml').safe_load(config_yaml))
    proxy = LockProxy(config)
    
    # Check backends have correct domain_name
    assert proxy.backends['llm'].domain_name == 'compute'
    assert proxy.backends['stt'].domain_name == 'compute'
    
    # Check domains dict
    assert 'compute' in proxy.domains
    assert 'llm' in proxy.domains['compute'].backends
    assert 'stt' in proxy.domains['compute'].backends


def test_lock_targets_within_domain():
    """Test that get_lock_targets only returns backends in same domain."""
    config_yaml = """
backends:
  compute:
    llm:
      url: http://localhost:8080
      paths: [/v1/chat]
      locks: [stt]
    stt:
      url: http://localhost:7301
      paths: [/transcribe]
      locks: []
  
  frontend:
    webui:
      url: http://localhost:9090
      paths: [/]
      locks: []

global_lock:
  enabled: true
"""
    config = Config.from_dict(__import__('yaml').safe_load(config_yaml))
    proxy = LockProxy(config)
    
    # llm is in 'compute' domain, stt is also in 'compute'
    compute_domain = proxy.domains['compute']
    llm_backend = compute_domain.backends['llm']
    
    # get_lock_targets should return stt (same domain)
    targets = llm_backend.get_lock_targets(compute_domain.backends)
    assert 'stt' in targets
    
    # But if we pass backends from another domain, they shouldn't be returned
    frontend_domain = proxy.domains['frontend']
    targets = llm_backend.get_lock_targets(frontend_domain.backends)
    assert 'stt' not in targets


def test_cross_domain_lock_validation():
    """Test that cross-domain locks are caught during validation."""
    config_yaml = """
backends:
  compute:
    llm:
      url: http://localhost:8080
      paths: [/v1/chat]
      locks: [webui]  # webui is in 'frontend', not 'compute'
  
  frontend:
    webui:
      url: http://localhost:9090
      paths: [/]
      locks: []

global_lock:
  enabled: true
"""
    with pytest.raises(ValueError) as exc_info:
        Config.from_dict(__import__('yaml').safe_load(config_yaml))
    
    assert 'frontend' in str(exc_info.value)
    assert 'compute' in str(exc_info.value)


def test_missing_lock_target_validation():
    """Test that missing lock targets are caught during validation."""
    config_yaml = """
backends:
  compute:
    llm:
      url: http://localhost:8080
      paths: [/v1/chat]
      locks: [nonexistent]

global_lock:
  enabled: true
"""
    with pytest.raises(ValueError) as exc_info:
        Config.from_dict(__import__('yaml').safe_load(config_yaml))
    
    assert 'nonexistent' in str(exc_info.value)


def test_multiple_domains_isolation():
    """Test that multiple domains have separate LockManagers."""
    config_yaml = """
backends:
  compute:
    llm:
      url: http://localhost:8080
      paths: [/v1/chat]
      locks: [stt]
    stt:
      url: http://localhost:7301
      paths: [/transcribe]
      locks: []
  
  frontend:
    webui:
      url: http://localhost:9090
      paths: [/]
      locks: []
  
  audio:
    tts:
      url: http://localhost:7302
      paths: [/tts]
      locks: []

global_lock:
  enabled: true
"""
    config = Config.from_dict(__import__('yaml').safe_load(config_yaml))
    proxy = LockProxy(config)
    
    # Check each domain has its own LockManager
    assert proxy.domains['compute'].lock_manager is not None
    assert proxy.domains['frontend'].lock_manager is not None
    assert proxy.domains['audio'].lock_manager is not None
    
    # Check LockManagers are different objects
    assert proxy.domains['compute'].lock_manager != proxy.domains['frontend'].lock_manager
    assert proxy.domains['frontend'].lock_manager != proxy.domains['audio'].lock_manager


def test_config_endpoint():
    """Test that /config endpoint returns full config as JSON."""
    config_yaml = """
backends:
  compute:
    llm:
      url: http://localhost:8080
      paths: [/v1/chat]
      locks: []

global_lock:
  enabled: true
"""
    config = Config.from_dict(__import__('yaml').safe_load(config_yaml))
    proxy = LockProxy(config)
    
    # Test model_dump returns dict
    config_dict = proxy.config.model_dump()
    assert isinstance(config_dict, dict)
    assert 'server' in config_dict
    assert 'domains' in config_dict
    assert 'compute' in config_dict['domains']
    assert 'backends' in config_dict['domains']['compute']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
