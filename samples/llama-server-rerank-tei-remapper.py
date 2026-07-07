"""Remapper for llama-server --reranker"""

import json
from exrouter.remapper import RequestRemapper, RemapResult
from exrouter.hooks import HookContext


class RequestRemapper:
    async def remap(self, context: HookContext) -> RemapResult | None:
        path = context.request_path.lower()
        
        # Handle rerank requests
        if path in ("/v1/rerank", "/v1/reranking", "/reranking", "/rerank"):
            print(f"[REMAPPER] Handling rerank request: {path}")
            # Transform request body
            if context.response_status is None:
                print(f"[REMAPPER] Request branch: response_status is None")
                if not context.request_body:
                    return RemapResult(status_code=400, content=b"Empty body")
                
                try:
                    data = json.loads(context.request_body)
                    
                    # Some clients use "texts" or "passages" instead of "documents"
                    if "texts" in data and "documents" not in data:
                        data["documents"] = data.pop("texts")
                    if "passages" in data and "documents" not in data:
                        data["documents"] = data.pop("passages")
                    
                    # llama-server expects documents as string array, not objects
                    # Transform [{"text": "x"}] -> ["x"]
                    if "documents" in data:
                        docs = data["documents"]
                        if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], dict):
                            # Extract text from objects
                            data["documents"] = [
                                doc["text"] if isinstance(doc, dict) and "text" in doc else str(doc)
                                for doc in docs
                            ]
                    
                    print(f"[REMAPPER] Request: {path}, {len(data.get('documents', []))} documents")
                    
                    # Return remapped request - exrouter will forward to backend
                    return RemapResult(
                        body=json.dumps(data).encode()
                    )
                    
                except Exception as e:
                    print(f"[REMAPPER] Rerank request error: {e}")
                    error_response = {"error": str(e), "status": 502}
                    return RemapResult(
                        status_code=502,
                        content=json.dumps(error_response).encode()
                    )
            
            # Transform response body
            elif context.response_status is not None:
                print(f"[REMAPPER] Response branch entered: status={context.response_status}")
                if context.response_status != 200 or not context.response_body:
                    print(f"[REMAPPER] Skipping: status={context.response_status}, has_body={bool(context.response_body)}")
                    return None
                    
                try:
                    print(f"[REMAPPER] Parsing response body: {context.response_body[:200]}...")
                    data = json.loads(context.response_body)
                    print(f"[REMAPPER] Parsed type: {type(data)}")
                    
                    # llama-server returns results with 'relevance_score'
                    # TEI format expects 'score' - transform this
                    # Also: TEI returns a list directly, but llama-server wraps in object
                    if 'results' in data:
                        print(f"[REMAPPER] Found 'results' key, transforming...")
                        for result in data['results']:
                            if 'relevance_score' in result:
                                result['score'] = result.pop('relevance_score')
                                print(f"[REMAPPER] Transformed relevance_score -> score")
                        # Extract results array as root (TEI expects list, not object)
                        data = data['results']
                        print(f"[REMAPPER] Extracted results array, new type: {type(data)}")
                    else:
                        print(f"[REMAPPER] No 'results' key found in data")
                    
                    print(f"[REMAPPER] Response: {path}, {len(data)} results")
                    print(f"[REMAPPER] Returning list: {data[:2]}...")
                    
                    # Return the transformed response as JSON string (not bytes, not parsed object)
                    # RemapResult.content expects bytes | str, and Hindsight parses this as JSON
                    return RemapResult(
                        content=json.dumps(data)  # JSON string, not bytes
                    )
                    
                except Exception as e:
                    print(f"[REMAPPER] Rerank response error: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            else:
                print(f"[REMAPPER] Neither request nor response branch matched")
            
            return None
        
        return None
