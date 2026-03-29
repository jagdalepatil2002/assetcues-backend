from __future__ import annotations

import copy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from far_ai_brain.nodes.extract import extract_node


def _make_vlm_mock(mocker, extraction_json: dict[str, Any]) -> MagicMock:
    """Create and patch a VLMAdapter mock returning *extraction_json*."""
    instance = AsyncMock()
    instance.extract = AsyncMock(return_value=extraction_json)
    instance.simple_query = AsyncMock(return_value="test response")
    instance.model = "test-model"
    instance.provider = "test"
    instance.role = "primary"

    mock_cls = MagicMock(return_value=instance)
    mocker.patch("far_ai_brain.nodes.extract.VLMAdapter", mock_cls)
    return mock_cls


def _build_extract_state(png_bytes: bytes) -> dict[str, Any]:
    return {
        "page_images": [png_bytes],
        "page_count": 1,
        "page_groups": [
            {
                "group_index": 0,
                "page_indices": [0],
                "page_images": [png_bytes],
            },
        ],
        "region_crops": [],
        "multi_res_images": [],
        "thinking_level": "low",
    }


@pytest.mark.asyncio
async def test_extract_returns_extractions(
    sample_png_bytes: bytes,
    mock_extraction_json: dict[str, Any],
    mocker,
):
    _make_vlm_mock(mocker, mock_extraction_json)
    state = _build_extract_state(sample_png_bytes)

    result = await extract_node(state)

    assert "extractions" in result
    assert isinstance(result["extractions"], list)
    assert len(result["extractions"]) == 1

    extraction = result["extractions"][0]
    assert "assets_to_create" in extraction
    assert isinstance(extraction["assets_to_create"], list)
    assert len(extraction["assets_to_create"]) >= 1
    assert "extraction_meta" in extraction
    assert extraction["extraction_meta"]["model_used"] == "test-model"


@pytest.mark.asyncio
async def test_asset_expansion_quantity(
    sample_png_bytes: bytes,
    mock_extraction_json: dict[str, Any],
    mocker,
):
    extraction_qty3 = copy.deepcopy(mock_extraction_json)
    item = extraction_qty3["line_items"][0]
    item["quantity"] = {"value": 3, "confidence": 0.98}
    item["unit_price"] = {"value": 10000.0, "confidence": 0.94}
    item["taxable_amount"] = {"value": 30000.0, "confidence": 0.93}
    item["cgst_amount"] = {"value": 2700.0, "confidence": 0.91}
    item["sgst_amount"] = {"value": 2700.0, "confidence": 0.91}
    item["igst_amount"] = {"value": None, "confidence": None}
    item["line_total"] = {"value": 35400.0, "confidence": 0.92}

    extraction_qty3["totals"]["subtotal_before_tax"] = {"value": 30000.0, "confidence": 0.93}
    extraction_qty3["totals"]["total_cgst"] = {"value": 2700.0, "confidence": 0.91}
    extraction_qty3["totals"]["total_sgst"] = {"value": 2700.0, "confidence": 0.91}
    extraction_qty3["totals"]["total_tax"] = {"value": 5400.0, "confidence": 0.92}
    extraction_qty3["totals"]["grand_total"] = {"value": 35400.0, "confidence": 0.95}

    _make_vlm_mock(mocker, extraction_qty3)
    state = _build_extract_state(sample_png_bytes)

    result = await extract_node(state)

    extraction = result["extractions"][0]
    assets = extraction["assets_to_create"]
    assert len(assets) == 3
    assert [a["quantity_index"] for a in assets] == [1, 2, 3]
    assert all(a["quantity_total"] == 3 for a in assets)
    assert all(a["individual_cost_before_tax"] == 10000.0 for a in assets)
