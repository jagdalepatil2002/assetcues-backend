from __future__ import annotations

import io
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("LOG_LEVEL", "WARNING")
from far_ai_brain.logging_setup import configure_terminal_logging

configure_terminal_logging()
from PIL import Image

MINIMAL_PDF = (
    b"%PDF-1.0\n1 0 obj<</Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</MediaBox[0 0 100 100]>>endobj\n"
    b"trailer<</Root 1 0 R>>"
)


def _make_image(color: tuple[int, int, int], fmt: str) -> bytes:
    img = Image.new("RGB", (100, 100), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    return _make_image((255, 0, 0), "JPEG")


@pytest.fixture
def sample_png_bytes() -> bytes:
    return _make_image((255, 0, 0), "PNG")


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return MINIMAL_PDF


@pytest.fixture
def mock_extraction_json() -> dict[str, Any]:
    """Realistic ExtractionOutput-like dict with sample invoice data."""
    return {
        "extraction_meta": {
            "extraction_id": "test-ext-001",
            "model_used": "test-model",
            "model_version": "1.0",
            "timestamp": "2025-01-01T00:00:00Z",
            "processing_time_ms": 1500,
            "overall_confidence": 0.92,
            "pages_processed": 1,
        },
        "document_type": {"value": "printed_invoice", "confidence": 0.95},
        "vendor_details": {
            "vendor_name": {"value": "Test Vendor Pvt Ltd", "confidence": 0.95},
            "vendor_gstin": {"value": "27AABCT1332L1ZS", "confidence": 0.90},
            "vendor_pan": {"value": None, "confidence": None},
            "vendor_address": {"value": "123 Test St, Mumbai", "confidence": 0.88},
            "vendor_phone": {"value": None, "confidence": None},
            "vendor_email": {"value": None, "confidence": None},
        },
        "buyer_details": {
            "buyer_name": {"value": "Buyer Corp", "confidence": 0.93},
            "buyer_gstin": {"value": "29AABCU9603R1ZM", "confidence": 0.91},
            "buyer_address": {"value": "456 Buyer Ave", "confidence": 0.87},
            "ship_to_address": {"value": None, "confidence": None},
        },
        "invoice_header": {
            "invoice_number": {"value": "INV-2025-001", "confidence": 0.97},
            "invoice_date": {"value": "2025-01-15", "confidence": 0.94},
            "due_date": {"value": None, "confidence": None},
            "po_number": {"value": "PO-100", "confidence": 0.85},
            "po_date": {"value": None, "confidence": None},
            "grn_number": {"value": None, "confidence": None},
            "delivery_note_number": {"value": None, "confidence": None},
            "challan_number": {"value": None, "confidence": None},
            "payment_terms": {"value": "Net 30", "confidence": 0.80},
            "currency": {"value": "INR", "confidence": 0.99},
        },
        "line_items": [
            {
                "line_index": 0,
                "description": {"value": "Dell Laptop Latitude 5540", "confidence": 0.96},
                "hsn_sac_code": {"value": "84713010", "confidence": 0.88},
                "quantity": {"value": 1, "confidence": 0.98},
                "unit": {"value": "Nos", "confidence": 0.90},
                "unit_price": {"value": 55000.0, "confidence": 0.94},
                "discount_percent": {"value": None, "confidence": None},
                "taxable_amount": {"value": 55000.0, "confidence": 0.93},
                "cgst_rate": {"value": 9.0, "confidence": 0.91},
                "cgst_amount": {"value": 4950.0, "confidence": 0.91},
                "sgst_rate": {"value": 9.0, "confidence": 0.91},
                "sgst_amount": {"value": 4950.0, "confidence": 0.91},
                "igst_rate": {"value": None, "confidence": None},
                "igst_amount": {"value": None, "confidence": None},
                "line_total": {"value": 64900.0, "confidence": 0.92},
                "serial_numbers_listed": ["SN-ABC-001"],
            },
        ],
        "totals": {
            "subtotal_before_tax": {"value": 55000.0, "confidence": 0.93},
            "total_cgst": {"value": 4950.0, "confidence": 0.91},
            "total_sgst": {"value": 4950.0, "confidence": 0.91},
            "total_igst": {"value": None, "confidence": None},
            "total_tax": {"value": 9900.0, "confidence": 0.92},
            "rounding_off": {"value": None, "confidence": None},
            "grand_total": {"value": 64900.0, "confidence": 0.95},
            "amount_in_words": {
                "value": "Sixty Four Thousand Nine Hundred Rupees Only",
                "confidence": 0.85,
            },
        },
        "bank_details": {
            "bank_name": {"value": "HDFC Bank", "confidence": 0.87},
            "account_number": {"value": "50100123456789", "confidence": 0.85},
            "ifsc_code": {"value": "HDFC0001234", "confidence": 0.86},
            "branch": {"value": "Mumbai Main", "confidence": 0.80},
        },
        "warranty_amc_info": {
            "warranty_period": {"value": "3 Years", "confidence": 0.82},
            "warranty_start_date": {"value": None, "confidence": None},
            "amc_period": {"value": None, "confidence": None},
            "amc_start_date": {"value": None, "confidence": None},
            "support_contact": {"value": None, "confidence": None},
        },
        "assets_to_create": [
            {
                "temp_asset_id": "tmp_ast_001",
                "source_line_index": 0,
                "asset_name": "Dell Laptop Latitude 5540",
                "quantity_index": 1,
                "quantity_total": 1,
                "individual_cost_before_tax": 55000.0,
                "individual_tax": 9900.0,
                "individual_cost_with_tax": 64900.0,
                "serial_number": "SN-ABC-001",
                "suggested_category": "IT Equipment",
                "suggested_sub_category": "Computing",
                "suggested_asset_class": "Laptop",
                "date_of_acquisition": "2025-01-15",
                "audit_indicator": "physical",
                "audit_method": "visual_inspection",
                "group_action": "none",
                "confidence_overall": 0.93,
            },
        ],
        "raw_complete_extraction": {
            "all_text_blocks": [],
            "stamps_and_signatures": [],
            "tables_raw": [],
            "other_visible_info": [],
            "handwritten_notes": [],
            "full_text_dump": "",
        },
        "validation_results": {
            "math_check": {
                "line_totals_valid": True,
                "subtotal_valid": True,
                "tax_calculations_valid": True,
                "grand_total_valid": True,
                "issues": [],
            },
            "format_checks": {
                "gstin_valid": True,
                "pan_valid": None,
                "dates_valid": True,
                "hsn_valid": True,
                "issues": [],
            },
        },
        "ai_reasoning_log": [],
    }


@pytest.fixture
def mock_vlm_adapter(mocker, mock_extraction_json):
    """Patch VLMAdapter in all node modules to return mock data."""
    instance = AsyncMock()
    instance.extract = AsyncMock(return_value=mock_extraction_json)
    instance.simple_query = AsyncMock(return_value="test response")
    instance.extract_batch = AsyncMock(return_value=[mock_extraction_json])
    instance.model = "test-model"
    instance.provider = "test"
    instance.role = "primary"

    mock_cls = MagicMock(return_value=instance)

    for module in [
        "far_ai_brain.ai.vlm_adapter",
        "far_ai_brain.nodes.classify",
        "far_ai_brain.nodes.extract",
    ]:
        mocker.patch(f"{module}.VLMAdapter", mock_cls)

    return mock_cls
