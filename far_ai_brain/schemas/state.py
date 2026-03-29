"""
Pipeline state — carried through every LangGraph node.

Uses TypedDict (not Pydantic BaseModel) because LangGraph needs to
pass bytes fields (images) between nodes without serialization loss.
"""
from __future__ import annotations

from typing import Any, TypedDict


class PageGroup(TypedDict):
    """A group of pages belonging to one logical invoice within a multi-page PDF."""
    group_index: int
    page_indices: list[int]
    page_images: list[bytes]


class RegionCrop(TypedDict, total=False):
    """A cropped region of an invoice page."""
    region: str  # "header", "table", "footer"
    image: bytes
    bbox: dict[str, int]  # {"x": int, "y": int, "w": int, "h": int}
    page_index: int


class PipelineState(TypedDict, total=False):
    """Full state object passed through the LangGraph pipeline."""

    # ── Input ──
    tenant_id: str
    upload_id: str
    raw_file_bytes: bytes
    file_type: str
    file_name: str
    mode: str

    # ── Preprocess output ──
    page_images: list[bytes]
    page_count: int
    quality_scores: list[float]
    is_handwritten: bool
    native_text: str | None
    multi_res_images: list[dict[str, bytes | None]]
    region_crops: list[RegionCrop]

    # ── Classify output ──
    document_type: str
    page_pattern: str  # "single_invoice" | "multiple_invoices"
    page_groups: list[PageGroup]
    extraction_plan: dict[str, Any]
    thinking_level: str  # "low" | "high"

    # ── Extract output (list — one per page group / invoice) ──
    extractions: list[dict[str, Any]]
    region_extractions: list[dict[str, dict[str, Any]]]
    source_maps: list[dict[str, Any]]

    # ── Enrich output ──
    split_suggestions: list[dict[str, Any]]
    group_suggestions: list[dict[str, Any]]

    # ── Verify output ──
    validation_results: list[dict[str, Any]]
    fields_for_review: list[dict[str, Any]]
    fields_failing: list[dict[str, Any]]
    final_confidence: float
    retry_count: int

    # ── Control ──
    processing_complete: bool
    error: str | None
