from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from far_ai_brain.services.training_collector import (
    _confidence_float,
    _document_type_str,
    _model_used_str,
    _resolve_extraction_id,
    _safe_file_stem,
    collect_training_data,
    collect_training_node,
)


class TestSafeFileStem:
    def test_clean_string(self):
        assert _safe_file_stem("ext-001") == "ext-001"

    def test_special_characters(self):
        result = _safe_file_stem("ext/001\\bad:chars")
        assert "/" not in result
        assert "\\" not in result
        assert ":" not in result

    def test_empty_string(self):
        assert _safe_file_stem("") == "extraction"

    def test_truncation(self):
        long = "a" * 300
        result = _safe_file_stem(long)
        assert len(result) <= 200


class TestResolveExtractionId:
    def test_from_meta(self):
        ext = {"extraction_meta": {"extraction_id": "ext-123"}}
        assert _resolve_extraction_id(ext, "upload-1", 0) == "ext-123"

    def test_fallback_to_upload_id(self):
        ext = {"extraction_meta": {}}
        assert _resolve_extraction_id(ext, "upload-1", 2) == "upload-1_2"

    def test_fallback_unknown(self):
        ext = {}
        result = _resolve_extraction_id(ext, "", 0)
        assert result == "unknown_0"


class TestDocumentTypeStr:
    def test_dict_with_value(self):
        ext = {"document_type": {"value": "printed_invoice"}}
        assert _document_type_str(ext, "fallback") == "printed_invoice"

    def test_plain_string(self):
        ext = {"document_type": "thermal_receipt"}
        assert _document_type_str(ext, "fallback") == "thermal_receipt"

    def test_missing(self):
        assert _document_type_str({}, "fallback") == "fallback"


class TestConfidenceFloat:
    def test_from_meta(self):
        ext = {"extraction_meta": {"overall_confidence": 0.92}}
        assert _confidence_float(ext) == 0.92

    def test_missing(self):
        assert _confidence_float({}) == 0.0


class TestModelUsedStr:
    def test_from_meta(self):
        ext = {"extraction_meta": {"model_used": "gemini-flash"}}
        assert _model_used_str(ext) == "gemini-flash"

    def test_missing(self):
        assert _model_used_str({}) == ""


class TestCollectTrainingData:
    @pytest.mark.asyncio
    async def test_disabled_does_nothing(self, mocker):
        mocker.patch("far_ai_brain.services.training_collector.settings")
        from far_ai_brain.services.training_collector import settings
        mocker.patch.object(settings, "save_training_data", False)

        state: dict[str, Any] = {"extractions": [{"test": True}]}
        await collect_training_data(state)

    @pytest.mark.asyncio
    async def test_enabled_writes_files(self, sample_png_bytes, mocker):
        with tempfile.TemporaryDirectory() as tmpdir:
            mocker.patch("far_ai_brain.services.training_collector.settings")
            from far_ai_brain.services.training_collector import settings
            mocker.patch.object(settings, "save_training_data", True)
            mocker.patch.object(settings, "training_data_dir", tmpdir)

            state: dict[str, Any] = {
                "extractions": [{
                    "extraction_meta": {
                        "extraction_id": "test-ext",
                        "overall_confidence": 0.9,
                        "model_used": "test-model",
                    },
                    "document_type": {"value": "printed_invoice"},
                }],
                "page_images": [sample_png_bytes],
                "page_groups": [{"group_index": 0, "page_indices": [0], "page_images": [sample_png_bytes]}],
                "upload_id": "upload-1",
                "tenant_id": "tenant-1",
                "file_name": "test.png",
                "document_type": "printed_invoice",
                "is_handwritten": False,
                "quality_scores": [0.85],
                "region_crops": [],
            }
            await collect_training_data(state)

            jsonl_path = os.path.join(tmpdir, "extractions.jsonl")
            assert os.path.exists(jsonl_path)

            with open(jsonl_path) as f:
                line = f.readline()
                record = json.loads(line)
                assert record["extraction_id"] == "test-ext"
                assert record["tenant_id"] == "tenant-1"
                assert len(record["image_paths"]) >= 1

            images_dir = os.path.join(tmpdir, "images")
            assert os.path.isdir(images_dir)
            assert len(os.listdir(images_dir)) >= 1

    @pytest.mark.asyncio
    async def test_no_extractions_skips(self, mocker):
        with tempfile.TemporaryDirectory() as tmpdir:
            mocker.patch("far_ai_brain.services.training_collector.settings")
            from far_ai_brain.services.training_collector import settings
            mocker.patch.object(settings, "save_training_data", True)
            mocker.patch.object(settings, "training_data_dir", tmpdir)

            state: dict[str, Any] = {"extractions": []}
            await collect_training_data(state)

            jsonl_path = os.path.join(tmpdir, "extractions.jsonl")
            assert not os.path.exists(jsonl_path)

    @pytest.mark.asyncio
    async def test_failure_does_not_propagate(self, mocker):
        mocker.patch("far_ai_brain.services.training_collector.settings")
        from far_ai_brain.services.training_collector import settings
        mocker.patch.object(settings, "save_training_data", True)
        mocker.patch.object(settings, "training_data_dir", "/nonexistent/path/that/will/fail")

        mocker.patch("os.makedirs", side_effect=PermissionError("denied"))

        state: dict[str, Any] = {"extractions": [{"test": True}]}
        await collect_training_data(state)


class TestCollectTrainingNode:
    @pytest.mark.asyncio
    async def test_returns_empty_dict(self, mocker):
        mocker.patch("far_ai_brain.services.training_collector.settings")
        from far_ai_brain.services.training_collector import settings
        mocker.patch.object(settings, "save_training_data", False)

        result = await collect_training_node({})
        assert result == {}
