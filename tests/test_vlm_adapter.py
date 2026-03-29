from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from far_ai_brain.ai.vlm_adapter import ExtractionError, VLMAdapter


class TestExtractionError:
    def test_message(self):
        err = ExtractionError("bad json")
        assert str(err) == "bad json"
        assert err.raw_response == ""

    def test_raw_response_stored(self):
        err = ExtractionError("parse fail", raw_response="<html>gibberish</html>")
        assert err.raw_response == "<html>gibberish</html>"


class TestParseJsonResponse:
    def _adapter(self) -> VLMAdapter:
        with patch("far_ai_brain.ai.vlm_adapter.model_config") as mock_cfg:
            mock_cfg.get_provider_and_model.return_value = ("google", "test-model", None)
            mock_cfg.get_api_key.return_value = "fake-key"
            return VLMAdapter(role="primary")

    def test_clean_json(self):
        adapter = self._adapter()
        result = adapter._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_strips_markdown_fences(self):
        adapter = self._adapter()
        raw = '```json\n{"key": "value"}\n```'
        result = adapter._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_strips_bare_fences(self):
        adapter = self._adapter()
        raw = '```\n{"key": 123}\n```'
        result = adapter._parse_json_response(raw)
        assert result == {"key": 123}

    def test_invalid_json_raises(self):
        adapter = self._adapter()
        with pytest.raises(ExtractionError, match="invalid JSON"):
            adapter._parse_json_response("not json at all")

    def test_raw_response_truncated(self):
        adapter = self._adapter()
        long_garbage = "x" * 5000
        with pytest.raises(ExtractionError) as exc_info:
            adapter._parse_json_response(long_garbage)
        assert len(exc_info.value.raw_response) <= 2000


class TestVLMAdapterInit:
    def test_resolves_role(self):
        with patch("far_ai_brain.ai.vlm_adapter.model_config") as mock_cfg:
            mock_cfg.get_provider_and_model.return_value = ("google", "gemini-flash", None)
            mock_cfg.get_api_key.return_value = "key123"
            adapter = VLMAdapter(role="cheap")
            assert adapter.role == "cheap"
            assert adapter.provider == "google"
            assert adapter.model == "gemini-flash"
            assert adapter.api_key == "key123"


class TestExtract:
    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with patch("far_ai_brain.ai.vlm_adapter.model_config") as mock_cfg:
            mock_cfg.get_provider_and_model.return_value = ("azure", "model", None)
            mock_cfg.get_api_key.return_value = "key"
            adapter = VLMAdapter(role="primary")

            with pytest.raises(ValueError, match="Unknown provider"):
                await adapter.extract(
                    images=[b"fake"],
                    system_prompt="sys",
                    user_prompt="usr",
                )


class TestSimpleQuery:
    @pytest.mark.asyncio
    async def test_google_provider(self):
        with patch("far_ai_brain.ai.vlm_adapter.model_config") as mock_cfg:
            mock_cfg.get_provider_and_model.return_value = ("google", "test-model", None)
            mock_cfg.get_api_key.return_value = "key"
            adapter = VLMAdapter(role="primary")

            mock_response = MagicMock()
            mock_response.text = "42"

            mock_aio = MagicMock()
            mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_aio.aclose = AsyncMock()

            mock_client_inst = MagicMock()
            mock_client_inst.aio = mock_aio

            with patch("far_ai_brain.ai.vlm_adapter.genai.Client", return_value=mock_client_inst):
                result = await adapter.simple_query("what is the number?")
                assert result == "42"
                mock_aio.aclose.assert_awaited()


class TestExtractBatch:
    @pytest.mark.asyncio
    async def test_runs_concurrently(self):
        with patch("far_ai_brain.ai.vlm_adapter.model_config") as mock_cfg:
            mock_cfg.get_provider_and_model.return_value = ("google", "test-model", None)
            mock_cfg.get_api_key.return_value = "key"
            adapter = VLMAdapter(role="primary")

            mock_response = MagicMock()
            mock_response.text = '{"result": "ok"}'

            mock_aio = MagicMock()
            mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_aio.aclose = AsyncMock()

            mock_client_inst = MagicMock()
            mock_client_inst.aio = mock_aio

            with patch("far_ai_brain.ai.vlm_adapter.genai.Client", return_value=mock_client_inst):
                calls = [
                    {"images": [b"img1"], "system_prompt": "sys", "user_prompt": "q1"},
                    {"images": [b"img2"], "system_prompt": "sys", "user_prompt": "q2"},
                ]
                results = await adapter.extract_batch(calls)
                assert len(results) == 2
                assert all(r["result"] == "ok" for r in results)
