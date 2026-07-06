"""Test domain-aware routing in LockProxy._find_backend().

Test cases:
1. Request to openwebui.unnsvc.org/path → matches open-webui only
2. Request to 127.0.0.1:4001/v1/tokenize → matches llm only
3. Request to 127.0.0.1:4001/ → matches backends with empty domain list
4. Request to dashboard.unnsvc.org/ → matches hermes-dashboard only
"""

import pytest
from exrouter.config import Config
from exrouter.proxy import LockProxy


@pytest.fixture
def domain_aware_config():
    """Create a test config with domain-aware routing.
    
    Simulates the real-world scenario:
    - open-webui: domain: ["openwebui.unnsvc.org"], paths: ["/"]
    - llm: domain: [], paths: ["/v1/tokenize", ...]
    - hermes-dashboard: domain: ["dashboard.unnsvc.org"], paths: ["/"]
    """
    return Config.from_dict({
        "server": {"host": "127.0.0.1", "port": 9999},
        "backends": {
            "frontend": {
                "open-webui": {
                    "url": "http://localhost:8080",
                    "paths": ["/"],
                    "domain": ["openwebui.unnsvc.org"],
                    "locks": []
                },
                "hermes-dashboard": {
                    "url": "http://localhost:8081",
                    "paths": ["/"],
                    "domain": ["dashboard.unnsvc.org"],
                    "locks": []
                }
            },
            "compute": {
                "llm": {
                    "url": "http://localhost:8082",
                    "paths": ["/v1/tokenize", "/v1/chat/completions"],
                    "domain": [],
                    "locks": []
                },
                "vision": {
                    "url": "http://localhost:8083",
                    "paths": ["/v1/vision/*"],
                    "domain": [],
                    "locks": []
                }
            }
        },
        "global_lock": {"enabled": False}
    })


def test_domain_specific_host_routes_to_named_domain(domain_aware_config):
    """Request to openwebui.unnsvc.org → only open-webui matches.
    
    The /v1/tokenize path should NOT match open-webui even though
    open-webui has paths: ["/"] which would match anything.
    """
    proxy = LockProxy(domain_aware_config)
    
    # Request to openwebui.unnsvc.org with /v1/tokenize path
    # Should match open-webui (domain matches), NOT llm (domain is empty)
    backend = proxy._find_backend("openwebui.unnsvc.org", "/v1/tokenize")
    
    assert backend is not None
    assert backend.name == "open-webui"


def test_ip_host_routes_to_domain_agnostic_backend(domain_aware_config):
    """Request to 127.0.0.1:4001/v1/tokenize → only llm matches.
    
    The Host header doesn't match any domain pattern, so only backends
    with empty domain list should be considered.
    """
    proxy = LockProxy(domain_aware_config)
    
    # Request to 127.0.0.1:4001 with /v1/tokenize path
    # Should match llm (domain is empty), NOT open-webui (domain doesn't match)
    backend = proxy._find_backend("127.0.0.1:4001", "/v1/tokenize")
    
    assert backend is not None
    assert backend.name == "llm"


def test_ip_host_root_path_routes_to_domain_agnostic(domain_aware_config):
    """Request to 127.0.0.1:4001/ → matches backends with empty domain list.
    
    Root path / should match llm (which has paths including /v1/* but no /)
    Actually, let's check what backends have / in their paths...
    """
    proxy = LockProxy(domain_aware_config)
    
    # Request to 127.0.0.1:4001 with / path
    # Should match llm or vision (domain is empty), NOT open-webui (domain doesn't match)
    # llm has paths: ["/v1/tokenize", "/v1/chat/completions"] - no "/"
    # vision has paths: ["/v1/vision/*"] - no "/"
    # So this should return None for root path on IP host
    backend = proxy._find_backend("127.0.0.1:4001", "/")
    
    # None because llm/vision don't have "/" in their paths
    assert backend is None


def test_dashboard_domain_routes_to_dashboard(domain_aware_config):
    """Request to dashboard.unnsvc.org/ → matches hermes-dashboard only."""
    proxy = LockProxy(domain_aware_config)
    
    # Request to dashboard.unnsvc.org with / path
    # Should match hermes-dashboard (domain matches and paths: ["/"])
    backend = proxy._find_backend("dashboard.unnsvc.org", "/")
    
    assert backend is not None
    assert backend.name == "hermes-dashboard"


def test_wildcard_domain_pattern(domain_aware_config):
    """Test wildcard domain patterns like '*.unnsvc.org'."""
    config = Config.from_dict({
        "server": {"host": "127.0.0.1", "port": 9999},
        "backends": {
            "frontend": {
                "wildcard-backend": {
                    "url": "http://localhost:8080",
                    "paths": ["/"],
                    "domain": ["*.unnsvc.org"],
                    "locks": []
                },
                "ip-backend": {
                    "url": "http://localhost:8081",
                    "paths": ["/"],
                    "domain": [],
                    "locks": []
                }
            }
        },
        "global_lock": {"enabled": False}
    })
    
    proxy = LockProxy(config)
    
    # Should match wildcard-backend
    backend = proxy._find_backend("anything.unnsvc.org", "/")
    assert backend is not None
    assert backend.name == "wildcard-backend"
    
    # Should match ip-backend
    backend = proxy._find_backend("127.0.0.1:8081", "/")
    assert backend is not None
    assert backend.name == "ip-backend"


def test_path_matching_after_domain_filter(domain_aware_config):
    """Path matching happens AFTER domain filtering."""
    proxy = LockProxy(domain_aware_config)
    
    # Request to openwebui.unnsvc.org with /v1/vision/query path
    # Should match open-webui (domain matches, paths: ["/"] matches everything)
    backend = proxy._find_backend("openwebui.unnsvc.org", "/v1/vision/query")
    assert backend is not None
    assert backend.name == "open-webui"
    
    # Request to 127.0.0.1:4001 with /v1/vision/query path
    # Should match vision (domain is empty, paths: ["/v1/vision/*"] matches)
    backend = proxy._find_backend("127.0.0.1:4001", "/v1/vision/query")
    assert backend is not None
    assert backend.name == "vision"
