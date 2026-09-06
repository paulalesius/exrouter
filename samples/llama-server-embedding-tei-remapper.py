"""TEI-compatible remapper for llama-server (fixed - uses httpx only)"""

import json
import httpx
from exrouter.remapper import RequestRemapper, RemapResult
from exrouter.hooks import HookContext

# Reuse httpx client (already available as dependency)
_client = httpx.AsyncClient(timeout=600.0)


class RequestRemapper:
    async def remap(self, context: HookContext) -> RemapResult | None:
        path = context.request_path.lower()

        # === /v1/info ===
        if path == "/v1/info":
            info = {
                "model_id": "bge-m3",
                "model_type": "text-embeddings",
                "max_input_length": 8192,
                "embedding_dim": 1024,
            }
            return RemapResult(
                status_code=200,
                content=json.dumps(info).encode(),
                response_headers={"content-type": "application/json"}
            )

        # === /v1/models ===
        if path in ("/v1/models", "/models"):
            return RemapResult(
                status_code=200,
                content=json.dumps({
                    "object": "list",
                    "data": [{"id": "bge-m3", "object": "model"}]
                }).encode(),
                response_headers={"content-type": "application/json"}
            )

        # === Handle embedding requests (TEI style) ===
        if path in ("/v1/embed", "/embed", "/v1/embeddings", "/embeddings"):
            if not context.request_body:
                return RemapResult(status_code=400, content=b"Empty body")

            try:
                data = json.loads(context.request_body)

                # Convert TEI-style "inputs" to OpenAI-style "input"
                if "inputs" in data and "input" not in data:
                    data["input"] = data.pop("inputs")

                print(f"[REMAPPER] Handling embedding request on {path}")

                # Call llama-server using httpx (already installed)
                resp = await _client.post(
                    "http://127.0.0.1:8081/v1/embeddings",
                    json=data
                )
                resp.raise_for_status()
                openai_resp = resp.json()

                # Detect if client wants raw TEI format or OpenAI format
                # Raw TEI format: /v1/embed or /embed paths typically expect list response
                # OpenAI format: /v1/embeddings or /embeddings paths expect {"data": [...]}
                # Also check if request used "inputs" (TEI) vs "input" (OpenAI)
                # Note: data dict already has "inputs" converted to "input" if present
                # Check original body for "inputs" key before conversion
                original_body = json.loads(context.request_body) if isinstance(context.request_body, str) else json.loads(context.request_body.decode('utf-8'))
                wants_raw_tei = (
                    path in ("/v1/embed", "/embed") or
                    "inputs" in original_body
                )

                if wants_raw_tei:
                    # Return raw TEI format: list of embedding vectors
                    embeddings = [item["embedding"] for item in openai_resp.get("data", [])]
                    return RemapResult(
                        status_code=200,
                        content=json.dumps(embeddings).encode(),
                        response_headers={"content-type": "application/json"}
                    )
                else:
                    # Return OpenAI-compatible format: {"data": [{"embedding": [...], "index": 0, ...}]}
                    # Used by Open WebUI, LangChain, and other OpenAI-compatible clients
                    return RemapResult(
                        status_code=200,
                        content=json.dumps(openai_resp).encode(),
                        response_headers={"content-type": "application/json"}
                    )

            except Exception as e:
                print(f"[REMAPPER] Embedding error: {e}")
                return RemapResult(status_code=502, content=str(e).encode())

        return None
