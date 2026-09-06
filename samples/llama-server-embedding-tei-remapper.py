"""TEI-compatible remapper for llama-server.

llama-server speaks OpenAI's /v1/embeddings API, but TEI clients expect
/v1/info, /v1/models, and a bare list of vectors from /v1/embed.

This remapper never calls the GPU itself. remap() runs before the proxy
acquires locks, so a direct HTTP call from here would bypass the lock
system entirely: an in-flight embedding would hold no locks and the
other GPU backends could run at the same time. Instead:

- /v1/info and /v1/models are answered statically (no GPU involved).
- Embedding requests are rewritten (TEI "inputs" -> OpenAI "input",
  path -> the backend's /v1/embeddings) and forwarded by the proxy
  under the backend's locks, like any other request.
- In the response phase the OpenAI answer is unwrapped back into a bare
  vector list for clients that asked for TEI format. That is marked
  with the x-exrouter-embed-format request header, which the proxy
  keeps in HookContext.request_headers across both phases (and forwards
  to the backend harmlessly).
"""

import json

from exrouter.hooks import HookContext
from exrouter.remapper import RemapResult, RequestRemapper

EMBED_PATHS = ("/v1/embed", "/embed", "/v1/embeddings", "/embeddings")
TEI_FORMAT_HEADER = "x-exrouter-embed-format"


class RequestRemapper:
    async def remap(self, context: HookContext) -> RemapResult | None:
        # Response phase: the backend answer is available. Unwrap it into
        # a bare TEI vector list when the client asked for TEI format.
        if context.response_body is not None:
            if (
                context.response_status != 200
                or context.request_headers.get(TEI_FORMAT_HEADER) != "tei"
            ):
                # OpenAI-format client or backend error: pass through.
                return None
            try:
                payload = json.loads(context.response_body)
                embeddings = [item["embedding"] for item in payload.get("data", [])]
            except (AttributeError, KeyError, TypeError, ValueError):
                # Unexpected payload: pass through untouched.
                return None
            return RemapResult(
                status_code=200,
                content=json.dumps(embeddings).encode(),
                response_headers={"content-type": "application/json"},
            )

        path = context.request_path.lower()

        # /v1/info (static, no GPU)
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
                response_headers={"content-type": "application/json"},
            )

        # /v1/models (static, no GPU)
        if path in ("/v1/models", "/models"):
            return RemapResult(
                status_code=200,
                content=json.dumps({
                    "object": "list",
                    "data": [{"id": "bge-m3", "object": "model"}],
                }).encode(),
                response_headers={"content-type": "application/json"},
            )

        # Embedding requests: translate and let the proxy forward under lock.
        if path in EMBED_PATHS:
            if not context.request_body:
                return RemapResult(status_code=400, content=b"Empty body")
            try:
                data = json.loads(context.request_body)
            except ValueError:
                return RemapResult(status_code=400, content=b"Invalid JSON body")

            wants_tei = path in ("/v1/embed", "/embed") or "inputs" in data
            if "inputs" in data and "input" not in data:
                data["input"] = data.pop("inputs")

            return RemapResult(
                path="/v1/embeddings",
                body=json.dumps(data).encode(),
                headers={TEI_FORMAT_HEADER: "tei"} if wants_tei else None,
            )

        return None
