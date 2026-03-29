from __future__ import annotations

from typing import Any

import pytest

from far_ai_brain.nodes.verify import (
    _amount_words_check,
    _format_checks,
    _math_checks,
    _num,
    _parse_indian_amount_words,
    _words_to_number,
    verify_node,
)


class TestNum:
    def test_float(self):
        assert _num(42.5) == 42.5

    def test_int(self):
        assert _num(10) == 10.0

    def test_none(self):
        assert _num(None) == 0.0

    def test_string_number(self):
        assert _num("123.45") == 123.45

    def test_invalid_string(self):
        assert _num("abc") == 0.0


class TestMathChecks:
    def test_valid_line_totals(self):
        ext = {
            "line_items": [
                {"line_index": 0, "quantity": 2, "unit_price": 100, "line_total": 200,
                 "taxable_amount": 200, "cgst_rate": 9, "cgst_amount": 18,
                 "sgst_rate": 9, "sgst_amount": 18},
            ],
            "subtotal_before_tax": 200,
            "total_tax": 36,
            "grand_total": 236,
        }
        result = _math_checks(ext)
        assert result["line_totals_valid"] is True
        assert result["grand_total_valid"] is True
        assert result["issues"] == []

    def test_invalid_line_total(self):
        ext = {
            "line_items": [
                {"line_index": 0, "quantity": 2, "unit_price": 100, "line_total": 999},
            ],
        }
        result = _math_checks(ext)
        assert result["line_totals_valid"] is False
        assert len(result["issues"]) > 0

    def test_invalid_grand_total(self):
        ext = {
            "line_items": [],
            "subtotal_before_tax": 1000,
            "total_tax": 180,
            "grand_total": 5000,
        }
        result = _math_checks(ext)
        assert result["grand_total_valid"] is False

    def test_empty_extraction(self):
        result = _math_checks({})
        assert result["line_totals_valid"] is True
        assert result["issues"] == []


class TestFormatChecks:
    def test_valid_gstin(self):
        ext = {"vendor_gstin": "27AABCT1332L1ZS"}
        result = _format_checks(ext)
        assert result["gstin_valid"] is True
        assert result["issues"] == []

    def test_invalid_gstin_format(self):
        ext = {"vendor_gstin": "INVALID123"}
        result = _format_checks(ext)
        assert result["gstin_valid"] is False
        assert any("GSTIN" in i for i in result["issues"])

    def test_invalid_gstin_state_code(self):
        # State code 99 is now valid (Centre Jurisdiction)
        # Use state code 55 which is genuinely invalid
        ext = {"vendor_gstin": "55AABCT1332L1ZS"}
        result = _format_checks(ext)
        assert result["gstin_valid"] is False

    def test_valid_pan(self):
        ext = {"vendor_pan": "ABCDE1234F"}
        result = _format_checks(ext)
        assert result["pan_valid"] is True

    def test_invalid_pan(self):
        ext = {"vendor_pan": "123INVALID"}
        result = _format_checks(ext)
        assert result["pan_valid"] is False

    def test_valid_hsn(self):
        ext = {"line_items": [{"hsn_sac_code": "84713010"}]}
        result = _format_checks(ext)
        assert result["hsn_valid"] is True

    def test_invalid_hsn(self):
        ext = {"line_items": [{"hsn_sac_code": "AB"}]}
        result = _format_checks(ext)
        assert result["hsn_valid"] is False

    def test_negative_amount(self):
        ext = {"line_items": [{"line_index": 0, "line_total": -500}]}
        result = _format_checks(ext)
        assert any("negative" in i for i in result["issues"])

    def test_empty_extraction(self):
        result = _format_checks({})
        assert result["gstin_valid"] is None
        assert result["pan_valid"] is None
        assert result["hsn_valid"] is None


class TestWordsToNumber:
    def test_simple_number(self):
        assert _words_to_number("forty two") == 42

    def test_hundred(self):
        assert _words_to_number("three hundred") == 300

    def test_thousand(self):
        assert _words_to_number("five thousand") == 5000

    def test_lakh(self):
        assert _words_to_number("twelve lakh") == 1200000

    def test_complex_number(self):
        result = _words_to_number("twelve lakh thirty four thousand five hundred sixty seven")
        assert result == 1234567

    def test_empty_string(self):
        assert _words_to_number("") is None

    def test_invalid_word(self):
        assert _words_to_number("banana") is None


class TestParseIndianAmountWords:
    def test_simple_amount(self):
        result = _parse_indian_amount_words("Sixty Four Thousand Nine Hundred Rupees Only")
        assert result == 64900

    def test_with_paise(self):
        result = _parse_indian_amount_words("One Hundred Rupees and Fifty Paise Only")
        assert result is not None
        assert abs(result - 100.5) < 1

    def test_none_on_empty(self):
        assert _parse_indian_amount_words("") is None
        assert _parse_indian_amount_words(None) is None

    def test_lakh_amount(self):
        result = _parse_indian_amount_words("Five Lakh Rupees Only")
        assert result == 500000


class TestAmountWordsCheck:
    def test_matching_amount(self):
        ext: dict[str, Any] = {
            "amount_in_words": "Sixty Four Thousand Nine Hundred Rupees Only",
            "grand_total": 64900,
        }
        review: list[dict] = []
        issues: list[str] = []
        _amount_words_check(ext, 0, review, issues)
        assert len(issues) == 0
        assert len(review) == 0

    def test_mismatching_amount(self):
        ext: dict[str, Any] = {
            "amount_in_words": "One Thousand Rupees Only",
            "grand_total": 99999,
        }
        review: list[dict] = []
        issues: list[str] = []
        _amount_words_check(ext, 0, review, issues)
        assert len(issues) > 0
        assert len(review) == 1

    def test_missing_words_skips(self):
        ext: dict[str, Any] = {"grand_total": 1000}
        review: list[dict] = []
        issues: list[str] = []
        _amount_words_check(ext, 0, review, issues)
        assert len(issues) == 0


class TestVerifyNode:
    @pytest.mark.asyncio
    async def test_empty_extractions(self):
        state = {"extractions": [], "page_images": []}
        result = await verify_node(state)
        assert result["final_confidence"] == 0.0
        assert result["fields_failing"] == []

    @pytest.mark.asyncio
    async def test_produces_validation_results(self, sample_png_bytes, mock_extraction_json):
        state = {
            "extractions": [mock_extraction_json],
            "page_images": [sample_png_bytes],
        }
        result = await verify_node(state)
        assert "validation_results" in result
        assert isinstance(result["final_confidence"], float)
        assert isinstance(result["fields_failing"], list)
