from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from far_ai_brain.nodes.classify import classify_node


def _mock_vlm(mocker, extract_response: dict[str, Any]) -> MagicMock:
    instance = AsyncMock()
    instance.extract = AsyncMock(return_value=extract_response)
    instance.simple_query = AsyncMock(return_value="test")
    instance.model = "test-model"
    instance.provider = "test"
    instance.role = "cheap"
    mock_cls = MagicMock(return_value=instance)
    mocker.patch("far_ai_brain.nodes.classify.VLMAdapter", mock_cls)
    return mock_cls


def _base_state(png_bytes: bytes, page_count: int = 1) -> dict[str, Any]:
    return {
        "page_images": [png_bytes] * page_count,
        "page_count": page_count,
        "quality_scores": [0.85] * page_count,
        "is_handwritten": False,
    }


class TestSinglePage:
    @pytest.mark.asyncio
    async def test_classifies_document(self, sample_png_bytes, mocker):
        _mock_vlm(mocker, {
            "document_type": "printed_invoice",
            "complexity": "simple",
            "extraction_plan": {"focus": "table"},
        })
        state = _base_state(sample_png_bytes)
        result = await classify_node(state)

        assert result["document_type"] == "printed_invoice"
        assert result["page_pattern"] == "single_invoice"
        assert len(result["page_groups"]) == 1
        assert result["page_groups"][0]["page_indices"] == [0]

    @pytest.mark.asyncio
    async def test_thinking_level_low_for_simple(self, sample_png_bytes, mocker):
        _mock_vlm(mocker, {
            "document_type": "printed_invoice",
            "complexity": "simple",
        })
        state = _base_state(sample_png_bytes)
        result = await classify_node(state)
        assert result["thinking_level"] == "low"

    @pytest.mark.asyncio
    async def test_thinking_level_high_for_handwritten(self, sample_png_bytes, mocker):
        _mock_vlm(mocker, {
            "document_type": "handwritten_bill",
            "complexity": "simple",
        })
        state = _base_state(sample_png_bytes)
        state["is_handwritten"] = True
        result = await classify_node(state)
        assert result["thinking_level"] == "high"

    @pytest.mark.asyncio
    async def test_thinking_level_high_for_low_quality(self, sample_png_bytes, mocker):
        _mock_vlm(mocker, {
            "document_type": "printed_invoice",
            "complexity": "simple",
        })
        state = _base_state(sample_png_bytes)
        state["quality_scores"] = [0.3]
        result = await classify_node(state)
        assert result["thinking_level"] == "high"


class TestVLMFailure:
    @pytest.mark.asyncio
    async def test_fallback_on_exception(self, sample_png_bytes, mocker):
        instance = AsyncMock()
        instance.extract = AsyncMock(side_effect=RuntimeError("API down"))
        mock_cls = MagicMock(return_value=instance)
        mocker.patch("far_ai_brain.nodes.classify.VLMAdapter", mock_cls)

        state = _base_state(sample_png_bytes)
        result = await classify_node(state)

        assert result["document_type"] == "printed_invoice"
        assert result["thinking_level"] == "low"  # single-page fastpath
        assert len(result["page_groups"]) == 1


class TestMultiPage:
    @pytest.mark.asyncio
    async def test_single_invoice_pattern(self, sample_png_bytes, mocker):
        call_count = 0

        async def _extract_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"document_type": "printed_invoice", "complexity": "complex"}
            return {"pattern": "single_invoice", "page_groups": []}

        instance = AsyncMock()
        instance.extract = AsyncMock(side_effect=_extract_side_effect)
        mock_cls = MagicMock(return_value=instance)
        mocker.patch("far_ai_brain.nodes.classify.VLMAdapter", mock_cls)

        state = _base_state(sample_png_bytes, page_count=3)
        result = await classify_node(state)

        assert result["page_pattern"] == "single_invoice"
        assert len(result["page_groups"]) == 1
        assert result["page_groups"][0]["page_indices"] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_multiple_invoices_pattern(self, sample_png_bytes, mocker):
        call_count = 0

        async def _extract_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"document_type": "printed_invoice", "complexity": "complex"}
            return {
                "pattern": "multiple_invoices",
                "page_groups": [
                    {"group_index": 0, "page_indices": [0]},
                    {"group_index": 1, "page_indices": [1, 2]},
                ],
            }

        instance = AsyncMock()
        instance.extract = AsyncMock(side_effect=_extract_side_effect)
        mock_cls = MagicMock(return_value=instance)
        mocker.patch("far_ai_brain.nodes.classify.VLMAdapter", mock_cls)

        state = _base_state(sample_png_bytes, page_count=3)
        result = await classify_node(state)

        assert result["page_pattern"] == "multiple_invoices"
        assert len(result["page_groups"]) == 2
        assert result["page_groups"][0]["page_indices"] == [0]
        assert result["page_groups"][1]["page_indices"] == [1, 2]

    @pytest.mark.asyncio
    async def test_multipage_classification_failure_fallback(self, sample_png_bytes, mocker):
        call_count = 0

        async def _extract_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"document_type": "printed_invoice", "complexity": "simple"}
            raise RuntimeError("multipage API error")

        instance = AsyncMock()
        instance.extract = AsyncMock(side_effect=_extract_side_effect)
        mock_cls = MagicMock(return_value=instance)
        mocker.patch("far_ai_brain.nodes.classify.VLMAdapter", mock_cls)

        state = _base_state(sample_png_bytes, page_count=2)
        result = await classify_node(state)

        assert result["page_pattern"] == "single_invoice"
        assert result["page_groups"][0]["page_indices"] == [0, 1]
