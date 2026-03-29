from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from far_ai_brain.api.main import app
from far_ai_brain.config.settings import settings as app_settings
from tests.conftest import MINIMAL_PDF


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_pipeline_result() -> dict[str, Any]:
    """Pre-built pipeline result that _run_extraction can process."""
    return {
        "extractions": [
            {
                "extraction_meta": {
                    "extraction_id": "test-ext-001",
                    "overall_confidence": 0.92,
                },
                "assets_to_create": [
                    {"temp_asset_id": "tmp_001", "asset_name": "Test Laptop"},
                ],
            },
        ],
        "page_groups": [
            {"group_index": 0, "page_indices": [0], "page_images": [b"fake"]},
        ],
        "final_confidence": 0.92,
        "validation_results": [],
        "fields_for_review": [],
        "split_suggestions": [],
        "group_suggestions": [],
    }


def test_extract_multipart_pdf_sniffs_type_when_no_extension(
    client: TestClient,
    mock_pipeline_result: dict[str, Any],
    mocker,
):
    """file_name without '.' still works for PDF bytes (magic-byte sniff)."""
    mock_pipeline = mocker.patch("far_ai_brain.api.main.pipeline")
    mock_pipeline.ainvoke = AsyncMock(return_value=mock_pipeline_result)

    response = client.post(
        "/api/v1/extract/upload",
        data={"tenant_id": "test-tenant"},
        files={"file": ("upload", MINIMAL_PDF, "application/pdf")},
    )
    assert response.status_code == 200


def test_extract_multipart_success_png(
    client: TestClient,
    sample_png_bytes: bytes,
    mock_pipeline_result: dict[str, Any],
    mocker,
):
    mock_pipeline = mocker.patch("far_ai_brain.api.main.pipeline")
    mock_pipeline.ainvoke = AsyncMock(return_value=mock_pipeline_result)

    response = client.post(
        "/api/v1/extract/upload",
        data={"tenant_id": "test-tenant"},
        files={"file": ("invoice.png", sample_png_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["extractions"], list)
    assert len(data["extractions"]) > 0


def test_extract_json_endpoint_disabled(client: TestClient):
    response = client.post("/api/v1/extract", json={})
    assert response.status_code == 410
    assert "disabled" in response.json()["detail"].lower()


def test_extract_unsupported_format(
    client: TestClient,
    sample_png_bytes: bytes,
):
    # A PNG with .xyz extension is now correctly sniffed as PNG via magic bytes
    # Use truly unrecognizable bytes to trigger the unsupported format error
    response = client.post(
        "/api/v1/extract/upload",
        data={"tenant_id": "test-tenant"},
        files={"file": ("test.xyz", b"not-a-real-image-format-at-all", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_extract_oversized(
    client: TestClient,
    mocker,
):
    mocker.patch.object(app_settings, "max_file_size_mb", 0.001)
    large_bytes = b"\x00" * 2000
    response = client.post(
        "/api/v1/extract/upload",
        data={"tenant_id": "test-tenant"},
        files={"file": ("invoice.png", large_bytes, "image/png")},
    )
    assert response.status_code == 400
    assert "too large" in response.json()["detail"]


def test_extract_missing_tenant(client: TestClient, sample_png_bytes: bytes):
    response = client.post(
        "/api/v1/extract/upload",
        files={"file": ("invoice.png", sample_png_bytes, "image/png")},
    )
    assert response.status_code == 422


def test_extract_multipart_success(
    client: TestClient,
    sample_png_bytes: bytes,
    mock_pipeline_result: dict[str, Any],
    mocker,
):
    mock_pipeline = mocker.patch("far_ai_brain.api.main.pipeline")
    mock_pipeline.ainvoke = AsyncMock(return_value=mock_pipeline_result)

    response = client.post(
        "/api/v1/extract/upload",
        data={"tenant_id": "test-tenant"},
        files={"file": ("invoice.png", sample_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["extractions"], list)
    assert len(data["extractions"]) > 0


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert isinstance(data["models"], dict)
    assert len(data["models"]) > 0


def test_pipeline_error_returns_500(
    client: TestClient,
    sample_png_bytes: bytes,
    mocker,
):
    mock_pipeline = mocker.patch("far_ai_brain.api.main.pipeline")
    mock_pipeline.ainvoke = AsyncMock(side_effect=RuntimeError("Pipeline crashed"))

    response = client.post(
        "/api/v1/extract/upload",
        data={"tenant_id": "test-tenant"},
        files={"file": ("invoice.png", sample_png_bytes, "image/png")},
    )
    assert response.status_code == 500
    assert "failed" in response.json()["detail"].lower()


def test_pipeline_result_with_error_key(
    client: TestClient,
    sample_png_bytes: bytes,
    mocker,
):
    mock_pipeline = mocker.patch("far_ai_brain.api.main.pipeline")
    mock_pipeline.ainvoke = AsyncMock(return_value={"error": "VLM timeout"})

    response = client.post(
        "/api/v1/extract/upload",
        data={"tenant_id": "test-tenant"},
        files={"file": ("invoice.png", sample_png_bytes, "image/png")},
    )
    assert response.status_code == 500


def test_request_id_header_propagated(
    client: TestClient,
    sample_png_bytes: bytes,
    mock_pipeline_result: dict[str, Any],
    mocker,
):
    mock_pipeline = mocker.patch("far_ai_brain.api.main.pipeline")
    mock_pipeline.ainvoke = AsyncMock(return_value=mock_pipeline_result)

    custom_id = "my-custom-request-id-123"
    response = client.post(
        "/api/v1/extract/upload",
        data={"tenant_id": "test-tenant"},
        files={"file": ("invoice.png", sample_png_bytes, "image/png")},
        headers={"X-Request-ID": custom_id},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_request_id_generated_when_missing(
    client: TestClient,
    sample_png_bytes: bytes,
    mock_pipeline_result: dict[str, Any],
    mocker,
):
    mock_pipeline = mocker.patch("far_ai_brain.api.main.pipeline")
    mock_pipeline.ainvoke = AsyncMock(return_value=mock_pipeline_result)

    response = client.post(
        "/api/v1/extract/upload",
        data={"tenant_id": "test-tenant"},
        files={"file": ("invoice.png", sample_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0
