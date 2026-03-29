from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from far_ai_brain.schemas.api import (
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
    SingleExtractionResult,
)
from far_ai_brain.schemas.extraction import (
    AssetToCreate,
    ConfidenceField,
    ExtractionMeta,
    ExtractionOutput,
    LineItem,
    MathCheck,
    Totals,
    ValidationResults,
    VendorDetails,
)


class TestConfidenceField:
    def test_string_value(self):
        cf = ConfidenceField[str](value="hello", confidence=0.95)
        assert cf.value == "hello"
        assert cf.confidence == 0.95

    def test_float_value(self):
        cf = ConfidenceField[float](value=123.45, confidence=0.8)
        assert cf.value == 123.45

    def test_int_value(self):
        cf = ConfidenceField[int](value=42, confidence=0.99)
        assert cf.value == 42

    def test_none_value(self):
        cf = ConfidenceField[str]()
        assert cf.value is None
        assert cf.confidence is None

    def test_source_region_and_bbox(self):
        cf = ConfidenceField[str](
            value="INV-001",
            confidence=0.97,
            source_region="header",
            bbox={"x": 10, "y": 20, "w": 100, "h": 30},
        )
        assert cf.source_region == "header"
        assert cf.bbox["x"] == 10

    def test_json_round_trip(self):
        cf = ConfidenceField[str](value="test", confidence=0.5)
        data = cf.model_dump()
        restored = ConfidenceField[str](**data)
        assert restored.value == cf.value
        assert restored.confidence == cf.confidence


class TestExtractionOutput:
    def test_defaults(self):
        output = ExtractionOutput()
        assert output.extraction_meta.extraction_id == ""
        assert output.line_items == []
        assert output.assets_to_create == []

    def test_full_instantiation(self, mock_extraction_json):
        meta = ExtractionMeta(**mock_extraction_json["extraction_meta"])
        assert meta.extraction_id == "test-ext-001"
        assert meta.overall_confidence == 0.92

    def test_vendor_details_defaults(self):
        vd = VendorDetails()
        assert vd.vendor_name.value is None
        assert vd.vendor_gstin.confidence is None

    def test_line_item_with_serials(self):
        item = LineItem(
            line_index=0,
            description=ConfidenceField[str](value="Laptop", confidence=0.95),
            serial_numbers_listed=["SN-001", "SN-002"],
        )
        assert len(item.serial_numbers_listed) == 2

    def test_totals_defaults(self):
        t = Totals()
        assert t.grand_total.value is None

    def test_math_check_defaults(self):
        mc = MathCheck()
        assert mc.line_totals_valid is True
        assert mc.issues == []

    def test_validation_results_defaults(self):
        vr = ValidationResults()
        assert vr.math_check.grand_total_valid is True
        assert vr.cross_model_verification is None

    def test_asset_to_create_defaults(self):
        asset = AssetToCreate()
        assert asset.group_action == "none"
        assert asset.split_action == "none"
        assert asset.quantity_index == 1


class TestExtractionRequest:
    def test_valid_base64(self, sample_png_bytes):
        b64 = base64.b64encode(sample_png_bytes).decode()
        req = ExtractionRequest(
            tenant_id="t1",
            file_base64=b64,
            file_name="test.png",
        )
        assert req.get_file_bytes() == sample_png_bytes

    def test_invalid_base64(self):
        with pytest.raises(ValidationError):
            ExtractionRequest(
                tenant_id="t1",
                file_base64="!!!not_valid!!!",
                file_name="test.png",
            )

    def test_get_file_extension_with_dot(self):
        b64 = base64.b64encode(b"dummy").decode()
        req = ExtractionRequest(
            tenant_id="t1",
            file_base64=b64,
            file_name="invoice.PDF",
        )
        assert req.get_file_extension() == "pdf"

    def test_get_file_extension_no_dot(self):
        b64 = base64.b64encode(b"dummy").decode()
        req = ExtractionRequest(
            tenant_id="t1",
            file_base64=b64,
            file_name="noextension",
        )
        assert req.get_file_extension() == ""

    def test_get_file_extension_multiple_dots(self):
        b64 = base64.b64encode(b"dummy").decode()
        req = ExtractionRequest(
            tenant_id="t1",
            file_base64=b64,
            file_name="my.invoice.file.JPEG",
        )
        assert req.get_file_extension() == "jpeg"


class TestResponseModels:
    def test_single_extraction_result(self):
        ser = SingleExtractionResult(
            extraction_id="ext-001",
            source_pages=[0, 1],
            confidence=0.88,
        )
        assert ser.extraction_id == "ext-001"
        assert len(ser.source_pages) == 2

    def test_extraction_response(self):
        resp = ExtractionResponse(
            status="success",
            invoice_count=1,
            total_confidence=0.9,
            extractions=[
                SingleExtractionResult(extraction_id="e1", confidence=0.9),
            ],
        )
        assert resp.status == "success"
        assert len(resp.extractions) == 1

    def test_health_response_defaults(self):
        hr = HealthResponse()
        assert hr.status == "healthy"
        assert hr.models == {}
