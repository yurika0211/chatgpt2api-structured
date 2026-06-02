from __future__ import annotations

import unittest

from fastapi import HTTPException

from services.protocol import openai_v1_chat_complete, openai_v1_response
from services.protocol.structured_output import normalize_structured_output


class StructuredOutputTests(unittest.TestCase):
    def test_json_object_extracts_object_from_text(self):
        result = normalize_structured_output(
            'Here is the JSON:\n{"title": "Demo", "tags": ["api"]}',
            {"type": "json_object"},
        )

        self.assertEqual(result, '{"title":"Demo","tags":["api"]}')

    def test_json_schema_validates_required_fields(self):
        result = normalize_structured_output(
            '{"title": "Demo", "year": 2026}',
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "article",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "required": ["title", "year"],
                        "properties": {
                            "title": {"type": "string"},
                            "year": {"type": "integer"},
                        },
                    },
                },
            },
        )

        self.assertEqual(result, '{"title":"Demo","year":2026}')

    def test_json_schema_rejects_missing_required_field(self):
        with self.assertRaises(ValueError):
            normalize_structured_output(
                '{"title": "Demo"}',
                {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
                            "type": "object",
                            "required": ["title", "summary"],
                        },
                    },
                },
            )

    def test_chat_completions_rejects_streaming_response_format(self):
        with self.assertRaises(HTTPException) as ctx:
            openai_v1_chat_complete.handle(
                {
                    "model": "auto",
                    "stream": True,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": "Return JSON"}],
                }
            )

        self.assertEqual(ctx.exception.status_code, 400)

    def test_responses_rejects_streaming_response_format(self):
        with self.assertRaises(HTTPException) as ctx:
            openai_v1_response.handle(
                {
                    "model": "auto",
                    "stream": True,
                    "response_format": {"type": "json_object"},
                    "input": "Return JSON",
                }
            )

        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
