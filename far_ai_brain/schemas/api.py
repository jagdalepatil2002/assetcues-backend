"""
API response models for Assetcues Invoice Agentic AI.
Multipart upload is the active request path.
"""
from __future__ import annotations

import base64
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ExtractionRequest(BaseModel):
    """
    Deprecated request model kept for backward compatibility notes only.
    JSON/base64 endpoint is disabled; use multipart upload endpoint.
    """
    tenant_id: str = Field(..., description="Client tenant identifier")
    file_base64: str = Field(..., description="Invoice file as base64 string")
    file_name: str = Field(
        ...,
        description="Original filename with extension (e.g. invoice.pdf). Required so we know the format.",
    )
    mode: str = Field(default="creation", description="'creation' or 'enrichment'")
    existing_asset_id: Optional[str] = None
    po_data: Optional[dict[str, Any]] = None

    @field_validator("file_base64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("file_base64 is not valid base64")
        return v

    def get_file_bytes(self) -> bytes:
        return base64.b64decode(self.file_base64)

    def get_file_extension(self) -> str:
        if "." not in self.file_name:
            return ""
        return self.file_name.rsplit(".", 1)[-1].lower()


class SingleExtractionResult(BaseModel):
    """Extraction result for one invoice (a multi-page PDF may contain multiple)."""
    extraction_id: str
    source_pages: list[int] = Field(default_factory=list)
    assets_to_create: list[dict[str, Any]] = Field(default_factory=list)
    split_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    group_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    fields_for_review: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    extraction_json: Optional[dict[str, Any]] = None


class ExtractionResponse(BaseModel):
    """
    Response for POST /api/v1/extract.
    Always returns a list of extractions — even for a single invoice.
    """
    status: str  # "success" | "error"
    invoice_count: int = 1
    extractions: list[SingleExtractionResult] = Field(default_factory=list)
    total_confidence: float = 0.0
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    models: dict[str, str] = Field(default_factory=dict)
