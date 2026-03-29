from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from far_ai_brain.pipeline.graph import pipeline


def _patch_all_vlm(mocker, extraction_json: dict[str, Any]) -> MagicMock:
    """Mock VLMAdapter across every node module that instantiates it."""
    instance = AsyncMock()
    instance.extract = AsyncMock(return_value=extraction_json)
    instance.simple_query = AsyncMock(return_value="test response")
    instance.extract_batch = AsyncMock(return_value=[extraction_json])
    instance.model = "test-model"
    instance.provider = "test"
    instance.role = "primary"

    mock_cls = MagicMock(return_value=instance)

    for mod in [
        "far_ai_brain.nodes.classify",
        "far_ai_brain.nodes.extract",
    ]:
        mocker.patch(f"{mod}.VLMAdapter", mock_cls)

    return mock_cls


@pytest.mark.asyncio
async def test_full_pipeline_success(
    sample_png_bytes: bytes,
    mock_extraction_json: dict[str, Any],
    mocker,
):
    _patch_all_vlm(mocker, mock_extraction_json)

    initial_state = {
        "tenant_id": "test-tenant",
        "upload_id": "test-upload-001",
        "raw_file_bytes": sample_png_bytes,
        "file_type": "png",
        "file_name": "test_invoice.png",
        "retry_count": 0,
    }

    result = await pipeline.ainvoke(initial_state)

    assert "extractions" in result
    assert isinstance(result["extractions"], list)
    assert len(result["extractions"]) > 0
    assert "final_confidence" in result
    assert isinstance(result["final_confidence"], (int, float))

    first = result["extractions"][0]
    assert "extraction_meta" in first
    assert "assets_to_create" in first
    assert isinstance(first["assets_to_create"], list)


@pytest.mark.asyncio
async def test_pipeline_training_disabled(
    sample_png_bytes: bytes,
    mock_extraction_json: dict[str, Any],
    mocker,
):
    _patch_all_vlm(mocker, mock_extraction_json)

    mocker.patch("far_ai_brain.services.training_collector.settings")
    from far_ai_brain.services.training_collector import settings as tc_settings
    mocker.patch.object(tc_settings, "save_training_data", False)

    initial_state = {
        "tenant_id": "test-tenant",
        "upload_id": "test-upload-002",
        "raw_file_bytes": sample_png_bytes,
        "file_type": "png",
        "file_name": "test_invoice.png",
        "retry_count": 0,
    }

    result = await pipeline.ainvoke(initial_state)
    assert "extractions" in result
