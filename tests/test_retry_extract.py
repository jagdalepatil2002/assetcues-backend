from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from far_ai_brain.nodes.retry_extract import (
    _is_amount_field,
    _parse_amount,
    _write_back,
    retry_extract_node,
)


class TestIsAmountField:
    def test_direct_amount(self):
        assert _is_amount_field("grand_total") is True
        assert _is_amount_field("subtotal_before_tax") is True
        assert _is_amount_field("line_total") is True

    def test_nested_amount(self):
        assert _is_amount_field("line_0.unit_price") is True
        assert _is_amount_field("line_0.taxable_amount") is True

    def test_text_field(self):
        assert _is_amount_field("vendor_name") is False
        assert _is_amount_field("invoice_number") is False

    def test_nested_text_field(self):
        assert _is_amount_field("line_0.description") is False


class TestParseAmount:
    def test_simple_number(self):
        assert _parse_amount("45000.00") == 45000.0

    def test_with_commas(self):
        assert _parse_amount("1,00,000.50") == 100000.5

    def test_with_rupee_symbol(self):
        assert _parse_amount("₹ 5,000") == 5000.0

    def test_with_rs_prefix(self):
        assert _parse_amount("Rs 12345") == 12345.0

    def test_trailing_dot(self):
        assert _parse_amount("500.") == 500.0

    def test_invalid_text(self):
        assert _parse_amount("not a number") is None


class TestWriteBack:
    def test_top_level_field(self):
        extractions = [{"vendor_name": "old"}]
        _write_back(extractions, 0, "vendor_name", "new")
        assert extractions[0]["vendor_name"] == "new"

    def test_line_item_field(self):
        extractions = [{
            "line_items": [
                {"line_index": 0, "description": "old"},
            ],
        }]
        _write_back(extractions, 0, "line_0.description", "new")
        assert extractions[0]["line_items"][0]["description"] == "new"

    def test_invalid_ext_idx(self):
        extractions = [{"field": "val"}]
        _write_back(extractions, 5, "field", "new")
        assert extractions[0]["field"] == "val"

    def test_invalid_line_prefix_fallback(self):
        extractions = [{}]
        _write_back(extractions, 0, "badprefix.field", "value")
        assert extractions[0].get("badprefix.field") == "value"


class TestRetryExtractNode:
    @pytest.mark.asyncio
    async def test_no_failing_fields_skips(self, mocker):
        instance = AsyncMock()
        mock_cls = MagicMock(return_value=instance)
        mocker.patch("far_ai_brain.nodes.retry_extract.VLMAdapter", mock_cls)

        state = {
            "fields_failing": [],
            "extractions": [{"vendor_name": "Test"}],
            "source_maps": [],
            "page_images": [],
            "retry_count": 0,
        }
        result = await retry_extract_node(state)
        assert result["retry_count"] == 1
        instance.simple_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_reextraction(self, sample_png_bytes, mocker):
        instance = AsyncMock()
        instance.simple_query = AsyncMock(return_value="99999.00")
        instance.model = "test-model"
        instance.provider = "test"
        instance.role = "primary"
        mock_cls = MagicMock(return_value=instance)
        mocker.patch("far_ai_brain.nodes.retry_extract.VLMAdapter", mock_cls)
        mocker.patch(
            "far_ai_brain.nodes.retry_extract.crop_field_region",
            return_value=sample_png_bytes,
        )

        state = {
            "fields_failing": [
                {"extraction_index": 0, "field": "grand_total", "reason": "math"},
            ],
            "extractions": [{"grand_total": 50000}],
            "source_maps": [{"grand_total": {"bbox": {"x": 0, "y": 0, "w": 50, "h": 20}, "page_index": 0}}],
            "page_images": [sample_png_bytes],
            "retry_count": 0,
        }
        result = await retry_extract_node(state)
        assert result["retry_count"] == 1
        assert result["extractions"][0]["grand_total"] == 99999.0
        assert result["fields_failing"] == []

    @pytest.mark.asyncio
    async def test_crop_failure_skips_field(self, sample_png_bytes, mocker):
        instance = AsyncMock()
        mock_cls = MagicMock(return_value=instance)
        mocker.patch("far_ai_brain.nodes.retry_extract.VLMAdapter", mock_cls)
        mocker.patch(
            "far_ai_brain.nodes.retry_extract.crop_field_region",
            return_value=None,
        )

        state = {
            "fields_failing": [
                {"extraction_index": 0, "field": "grand_total", "reason": "math"},
            ],
            "extractions": [{"grand_total": 50000}],
            "source_maps": [{"grand_total": {"bbox": {"x": 0, "y": 0, "w": 1, "h": 1}, "page_index": 0}}],
            "page_images": [sample_png_bytes],
            "retry_count": 0,
        }
        result = await retry_extract_node(state)
        assert result["extractions"][0]["grand_total"] == 50000
