"""LLM remapper with detailed logging + improved schema validation.

Logs the exact final content sent to the client so we can diagnose why
the benchmark still rejects it even when our validator says PASSED.
"""

import json
import os
import re
from typing import Optional, Any
from exrouter.remapper import RequestRemapper, RemapResult
from exrouter.hooks import HookContext

DEBUG_LOG_RESPONSES = os.environ.get("EXROUTER_DEBUG_LOG_RESPONSES", "true").lower() == "true"


def strip_leading_think(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'^\s*<(?:think|reasoning)\b[^>]*>.*?</(?:think|reasoning)>\s*', '', text,
                  count=1, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'^\s*</?(?:think|reasoning)\b[^>]*>\s*', '', text,
                  count=1, flags=re.IGNORECASE)
    return text.strip()


def extract_json_from_text(text: str) -> Optional[str]:
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        candidate = match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return None


def clean_chat_response_content(content: str) -> str:
    cleaned = strip_leading_think(content)
    extracted = extract_json_from_text(cleaned)
    return extracted or cleaned


def validate_json_against_schema(data: Any, schema: dict, path: str = "$") -> list[str]:
    """Improved basic JSON Schema validator."""
    errors = []
    if not schema:
        return errors

    # Type
    expected_type = schema.get("type")
    if expected_type:
        type_map = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "object": dict, "array": list
        }
        python_type = type_map.get(expected_type)
        if python_type and not isinstance(data, python_type):
            errors.append(f"{path}: expected type '{expected_type}', got {type(data).__name__}")

    # Enum
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: value '{data}' not in enum {schema['enum']}")

    # String constraints
    if isinstance(data, str):
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(f"{path}: string too short (minLength={schema['minLength']})")
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(f"{path}: string too long (maxLength={schema['maxLength']})")

    # Number constraints
    if isinstance(data, (int, float)):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: value {data} < minimum ({schema['minimum']})")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path}: value {data} > maximum ({schema['maximum']})")

    # Object
    if isinstance(data, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", True)

        for req in required:
            if req not in data:
                errors.append(f"{path}: missing required property '{req}'")

        for key, value in data.items():
            if key in props:
                errors.extend(validate_json_against_schema(value, props[key], f"{path}.{key}"))
            elif additional is False:
                errors.append(f"{path}: unexpected property '{key}' (additionalProperties=false)")

    # Array
    if isinstance(data, list) and "items" in schema:
        for i, item in enumerate(data):
            errors.extend(validate_json_against_schema(item, schema["items"], f"{path}[{i}]"))

    return errors


class StreamingThinkStripper:
    def __init__(self, schema: Optional[dict] = None):
        self.buffer = ""
        self.stripped = False
        self.full_content = "" if DEBUG_LOG_RESPONSES else None
        self.schema = schema

    def process_chunk(self, content: str) -> Optional[str]:
        if self.full_content is not None:
            self.full_content += content or ""

        if self.stripped:
            return content

        self.buffer += content or ""
        pattern = r'^\s*<(?:think|reasoning)\b[^>]*>.*?</(?:think|reasoning)>\s*'
        match = re.match(pattern, self.buffer, re.DOTALL | re.IGNORECASE)

        if match:
            rest = self.buffer[match.end():]
            self.stripped = True
            self.buffer = ""
            return rest.strip() if rest else None

        if re.search(r'<(?:think|reasoning)\b[^>]*>', self.buffer, re.IGNORECASE):
            return None

        self.stripped = True
        result = self.buffer
        self.buffer = ""
        return result

    def __del__(self):
        if not self.full_content:
            return

        final = clean_chat_response_content(self.full_content)

        if DEBUG_LOG_RESPONSES:
            print("\n" + "="*95)
            print("[LLM-REMAPPER][DEBUG] Streaming - Raw model output:")
            print(repr(self.full_content[:500]))
            print("\n[LLM-REMAPPER][DEBUG] Streaming - Final cleaned content (sent to client):")
            print(repr(final[:500]))
            print(f"\nLength: {len(final)} characters")
            print("="*95 + "\n")

        if self.schema:
            try:
                parsed = json.loads(final)
                errors = validate_json_against_schema(parsed, self.schema)
                if errors:
                    print("\n" + "="*95)
                    print("[LLM-REMAPPER][JSON_SCHEMA] VALIDATION FAILED")
                    for err in errors:
                        print(f"   - {err}")
                    print("\nFinal content sent to client:")
                    print(repr(final))
                    print("="*95 + "\n")
                else:
                    print("\n" + "="*95)
                    print("[LLM-REMAPPER][JSON_SCHEMA] VALIDATION PASSED ✓")
                    print("Final content sent to client:")
                    print(repr(final))
                    print("="*95 + "\n")
            except Exception as e:
                print(f"[LLM-REMAPPER][JSON_SCHEMA] Could not parse/validate: {e}")


class RequestRemapper:
    def __init__(self):
        self._pending_schema: Optional[dict] = None

    def remap(self, context: HookContext) -> Optional[RemapResult]:
        if context.response_body:
            return self._process_response(context)

        path = context.request_path.lower()
        if not any(x in path for x in ["/chat/completions", "/completions", "/responses"]):
            return None
        if not context.request_body:
            return None

        try:
            data = json.loads(context.request_body)
            changed = False

            for key in ["reasoning_format", "reasoning-format", "reasoning", "enable_reasoning", "thinking"]:
                if key in data:
                    if data[key] not in (None, "none", False, "off"):
                        data[key] = "none"
                        changed = True
                else:
                    data[key] = "none"
                    changed = True

            if "response_format" in data:
                rf = data.get("response_format") or {}
                if isinstance(rf, dict) and rf.get("type") == "json_schema":
                    schema = rf.get("json_schema", {}).get("schema")
                    self._pending_schema = schema

                    print("\n" + "="*95)
                    print("[LLM-REMAPPER][JSON_SCHEMA] Client requested full json_schema")
                    print("   → Downgrading to json_object + client-side validation")
                    if schema:
                        print(f"   Schema top-level keys: {list(schema.keys()) if isinstance(schema, dict) else 'N/A'}")
                    print("="*95 + "\n")

                    data["response_format"] = {"type": "json_object"}
                    changed = True

            if changed:
                new_body = json.dumps(data).encode("utf-8")
                return RemapResult(body=new_body, headers={"content-length": str(len(new_body))})

        except Exception as e:
            print(f"[LLM-REMAPPER] Request error: {e}")

        return None

    def _process_response(self, context: HookContext) -> Optional[RemapResult]:
        try:
            resp = json.loads(context.response_body)
        except Exception:
            return None

        modified = False
        schema = self._pending_schema
        self._pending_schema = None

        if "choices" in resp:
            for choice in resp.get("choices", []):
                if "message" in choice and "content" in choice["message"]:
                    original = choice["message"]["content"] or ""
                    cleaned = clean_chat_response_content(original)
                    if cleaned != original:
                        choice["message"]["content"] = cleaned
                        modified = True

                    if schema:
                        try:
                            parsed = json.loads(cleaned)
                            errors = validate_json_against_schema(parsed, schema)
                            if errors:
                                print("\n" + "="*95)
                                print("[LLM-REMAPPER][JSON_SCHEMA] VALIDATION FAILED (non-streaming)")
                                for err in errors:
                                    print(f"   - {err}")
                                print("\nFinal content sent to client:")
                                print(repr(cleaned))
                                print("="*95 + "\n")
                            else:
                                print("\n" + "="*95)
                                print("[LLM-REMAPPER][JSON_SCHEMA] VALIDATION PASSED ✓ (non-streaming)")
                                print("Final content sent to client:")
                                print(repr(cleaned))
                                print("="*95 + "\n")
                        except Exception as e:
                            print(f"[LLM-REMAPPER][JSON_SCHEMA] Validation error: {e}")

        if modified:
            if DEBUG_LOG_RESPONSES:
                print("\n" + "="*95)
                print("[LLM-REMAPPER][DEBUG] Non-streaming - Response was cleaned")
                print("="*95 + "\n")

            new_body = json.dumps(resp).encode("utf-8")
            return RemapResult(content=new_body, response_headers={"content-type": "application/json"})

        return None

    def get_streaming_stripper(self):
        schema = self._pending_schema
        self._pending_schema = None
        return StreamingThinkStripper(schema=schema)
