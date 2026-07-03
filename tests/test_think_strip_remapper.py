"""Tests for the thinking tag strip remapper."""

import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.exrouter.config import Config
from src.exrouter.proxy import LockProxy


@pytest.fixture
def think_strip_proxy():
    """Create a proxy with think-strip remapper configured."""
    config_dict = {
        "server": {"host": "127.0.0.1", "port": 4001},
        "global_lock": {"enabled": False},
        "backends": {
            "llm": {
                "url": "http://127.0.0.1:8080",
                "paths": ["/v1/chat/completions", "/v1/completions", "/v1/responses"],
                "remapper": "/src/exrouter/samples/llm-think-strip-remapper.py"
            }
        }
    }
    config = Config(**config_dict)
    proxy = LockProxy(config)
    return proxy


def test_strip_thinking_tags_at_start_chat_completions(think_strip_proxy):
    """Test that thinking tags are stripped from chat completions response."""
    client = TestClient(think_strip_proxy.app)
    
    # Mock the backend response
    mock_response = {
        "choices": [{
            "message": {
                "content": "</think>\nThis is thinking content\n</think>\nThis is the actual response"
            }
        }]
    }
    
    # Patch the httpx client to return our mock response
    original_send = think_strip_proxy.httpx_client.send
    async def mock_send(request, stream=False):
        from httpx import Response
        import asyncio
        
        async def read_bytes():
            yield json.dumps(mock_response).encode()
        
        return Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(mock_response).encode()
        )
    
    think_strip_proxy.httpx_client.send = mock_send
    
    # Make a request
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify thinking tags are stripped
    content = data["choices"][0]["message"]["content"]
    assert "</think>" not in content
    assert "This is the actual response" in content
    assert "This is thinking content" not in content


def test_strip_thinking_tags_at_start_responses_api(think_strip_proxy):
    """Test that thinking tags are stripped from Responses API format."""
    client = TestClient(think_strip_proxy.app)
    
    # Properly formatted thinking tag at start
    mock_response = {
        "output": [{
            "content": [{
                "type": "output_text",
                "text": "</think>\nThinking here\n</think>\nActual response text"
            }]
        }],
        "status": "completed"
    }
    
    async def mock_send(request, stream=False):
        from httpx import Response
        return Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(mock_response).encode()
        )
    
    think_strip_proxy.httpx_client.send = mock_send
    
    response = client.post(
        "/v1/responses",
        json={
            "model": "test",
            "input": "Hello",
            "text": {"format": {"type": "text"}}
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify thinking tags are stripped
    text = data["output"][0]["content"][0]["text"]
    assert "Actual response text" in text
    assert "Thinking here" not in text


def test_preserve_thinking_tags_in_middle(think_strip_proxy):
    """Test that thinking tags in the middle of content are preserved."""
    client = TestClient(think_strip_proxy.app)
    
    mock_response = {
        "choices": [{
            "message": {
                "content": "Start of response\n</think>\nMiddle thinking\n</think>\nEnd of response"
            }
        }]
    }
    
    async def mock_send(request, stream=False):
        from httpx import Response
        return Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(mock_response).encode()
        )
    
    think_strip_proxy.httpx_client.send = mock_send
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Content should be unchanged since tag is not at start
    content = data["choices"][0]["message"]["content"]
    assert "Start of response" in content
    assert "Middle thinking" in content  # Preserved
    assert "End of response" in content


def test_strip_thinking_tags_with_different_formats(think_strip_proxy):
    """Test stripping with different thinking tag formats."""
    client = TestClient(think_strip_proxy.app)
    
    # Test  format
    mock_response_think = {
        "choices": [{
            "message": {
                "content": "</think>\nReasoning here\n</think>\nActual response"
            }
        }]
    }
    
    async def mock_send_think(request, stream=False):
        from httpx import Response
        return Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(mock_response_think).encode()
        )
    
    think_strip_proxy.httpx_client.send = mock_send_think
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]}
    )
    
    assert response.status_code == 200
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    assert "</think>" not in content
    assert "Actual response" in content


def test_streaming_think_strip():
    """Test streaming response with thinking tag stripping."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "llm_think_strip_remapper",
        "/src/exrouter/samples/llm-think-strip-remapper.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    StreamingThinkStripper = module.StreamingThinkStripper
    
    stripper = StreamingThinkStripper()
    
    # Simulate streaming chunks with thinking tags at start
    # First chunk starts with opening tag
    chunks = [
        "</think>\n",  # Opening tag
        "Thinking content\n",  # Thinking content
        "</think>\n",  # Closing tag
        "Actual response part 1\n",  # After closing tag
        "Actual response part 2\n"
    ]
    
    output_chunks = []
    for chunk in chunks:
        result = stripper.process_chunk(chunk)
        if result is not None:
            output_chunks.append(result)
    
    # After processing, thinking content should be stripped
    combined_output = "".join(output_chunks)
    assert "Actual response part 1" in combined_output
    assert "Actual response part 2" in combined_output
    assert "Thinking content" not in combined_output
