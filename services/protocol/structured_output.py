from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

JSON_MODE_INSTRUCTIONS = (
    "Return only valid JSON. Do not wrap the JSON in Markdown fences, do not add prose, "
    "and do not include comments."
)


def response_format_from_body(body: dict[str, Any]) -> dict[str, Any] | None:
    value = body.get("response_format")
    return value if isinstance(value, dict) else None


def has_structured_response_format(body: dict[str, Any]) -> bool:
    response_format = response_format_from_body(body)
    return bool(response_format and str(response_format.get("type") or "").strip())


def reject_streaming_structured_output(body: dict[str, Any]) -> None:
    if body.get("stream") and has_structured_response_format(body):
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "response_format is only supported for non-streaming text requests "
                    "by this ChatGPT web compatibility backend"
                )
            },
        )


def structured_output_system_message(response_format: dict[str, Any] | None) -> dict[str, str] | None:
    if not response_format:
        return None
    format_type = str(response_format.get("type") or "").strip()
    if format_type == "json_object":
        return {"role": "system", "content": JSON_MODE_INSTRUCTIONS}
    if format_type == "json_schema":
        schema = _json_schema(response_format)
        if not schema:
            return {"role": "system", "content": JSON_MODE_INSTRUCTIONS}
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        return {
            "role": "system",
            "content": (
                f"{JSON_MODE_INSTRUCTIONS} The JSON must conform to this JSON Schema: "
                f"{schema_text}"
            ),
        }
    return None


def _json_schema(response_format: dict[str, Any]) -> dict[str, Any]:
    schema_container = response_format.get("json_schema")
    if not isinstance(schema_container, dict):
        return {}
    schema = schema_container.get("schema")
    return schema if isinstance(schema, dict) else {}


def validate_response_format(response_format: dict[str, Any] | None) -> None:
    if not response_format:
        return
    format_type = str(response_format.get("type") or "").strip()
    if format_type not in {"json_object", "json_schema"}:
        raise HTTPException(
            status_code=400,
            detail={"error": f"unsupported response_format type: {format_type or '<empty>'}"},
        )
    if format_type == "json_schema":
        schema_container = response_format.get("json_schema")
        if not isinstance(schema_container, dict):
            raise HTTPException(status_code=400, detail={"error": "response_format.json_schema is required"})
        if not isinstance(schema_container.get("schema"), dict):
            raise HTTPException(status_code=400, detail={"error": "response_format.json_schema.schema is required"})


def normalize_structured_output(content: str, response_format: dict[str, Any] | None) -> str:
    if not response_format:
        return content
    validate_response_format(response_format)
    data = _loads_json_object(content)
    if str(response_format.get("type") or "").strip() == "json_schema":
        _validate_minimal_schema(data, _json_schema(response_format))
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _loads_json_object(content: str) -> Any:
    text = str(content or "").strip()
    if not text:
        raise ValueError("structured output was empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    extracted = _extract_json_value(text)
    if extracted is None:
        raise ValueError("structured output was not valid JSON")
    return extracted


def _extract_json_value(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    start_chars = "{["
    for index, char in enumerate(text):
        if char not in start_chars:
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not str(text[index + end :]).strip().strip("`"):
            return value
        return value
    return None


def _validate_minimal_schema(data: Any, schema: dict[str, Any]) -> None:
    schema_type = str(schema.get("type") or "").strip()
    if schema_type == "object" and not isinstance(data, dict):
        raise ValueError("structured output must be a JSON object")
    if schema_type == "array" and not isinstance(data, list):
        raise ValueError("structured output must be a JSON array")
    if isinstance(data, dict):
        required = schema.get("required")
        if isinstance(required, list):
            missing = [str(key) for key in required if str(key) not in data]
            if missing:
                raise ValueError(f"structured output is missing required fields: {', '.join(missing)}")
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, property_schema in properties.items():
                if key not in data or not isinstance(property_schema, dict):
                    continue
                _validate_simple_type(data[key], property_schema, str(key))


def _validate_simple_type(value: Any, schema: dict[str, Any], path: str) -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if "null" in schema_type and value is None:
            return
        schema_type = next((item for item in schema_type if item != "null"), "")
    expected = str(schema_type or "").strip()
    if not expected:
        return
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: (isinstance(item, int | float) and not isinstance(item, bool)),
        "boolean": lambda item: isinstance(item, bool),
    }
    check = checks.get(expected)
    if check and not check(value):
        raise ValueError(f"structured output field {path} must be {expected}")
