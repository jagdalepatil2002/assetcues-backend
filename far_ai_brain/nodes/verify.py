"""
Verify node — three layers of validation culminating in a retry/accept decision.

Layer 1: Deterministic math & format checks (free, instant).
Layer 2: Amount-in-words cross-check (Indian numbering).
Layer 3: Confidence scoring from deterministic check results.
"""
from __future__ import annotations

import re
from typing import Any

import structlog

from far_ai_brain.schemas.state import PipelineState

logger = structlog.get_logger()


def _scalar(val: Any) -> Any:
    """Unwrap ConfidenceField-shaped dicts from VLM JSON."""
    if isinstance(val, dict) and "value" in val:
        return val.get("value")
    return val


def _totals_block(extraction: dict[str, Any]) -> dict[str, Any]:
    t = extraction.get("totals")
    return t if isinstance(t, dict) else {}


def _scalar_from_totals_or_root(extraction: dict[str, Any], key: str) -> Any:
    tb = _totals_block(extraction)
    if key in tb:
        return _scalar(tb.get(key))
    return _scalar(extraction.get(key))


def _collect_gstin_candidates(extraction: dict[str, Any]) -> list[tuple[str, Any]]:
    """Top-level or nested vendor/buyer GSTIN values for format checks."""
    out: list[tuple[str, Any]] = []
    for key in ("vendor_gstin", "buyer_gstin"):
        if extraction.get(key) is not None:
            out.append((key, extraction[key]))
    vd = extraction.get("vendor_details")
    if isinstance(vd, dict) and vd.get("vendor_gstin") is not None:
        out.append(("vendor_gstin", vd["vendor_gstin"]))
    bd = extraction.get("buyer_details")
    if isinstance(bd, dict) and bd.get("buyer_gstin") is not None:
        out.append(("buyer_gstin", bd["buyer_gstin"]))
    return out


def _collect_pan_candidates(extraction: dict[str, Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for key in ("vendor_pan", "buyer_pan"):
        if extraction.get(key) is not None:
            out.append((key, extraction[key]))
    vd = extraction.get("vendor_details")
    if isinstance(vd, dict) and vd.get("vendor_pan") is not None:
        out.append(("vendor_pan", vd["vendor_pan"]))
    bd = extraction.get("buyer_details")
    if isinstance(bd, dict) and bd.get("buyer_pan") is not None:
        out.append(("buyer_pan", bd["buyer_pan"]))
    return out


_AMOUNT_FIELDS = frozenset({
    "grand_total", "subtotal_before_tax", "total_tax",
    "total_cgst", "total_sgst", "total_igst",
    "individual_cost_before_tax", "individual_cost_with_tax",
    "line_total", "taxable_amount",
    "cgst_amount", "sgst_amount", "igst_amount",
})

_GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z\d][A-Z]$")
_PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_HSN_RE = re.compile(r"^\d{4,8}$")
_VALID_STATE_CODES = set(range(1, 39)) | {96, 97, 99}  # 38=Ladakh, 96/97/99=territory codes


async def verify_node(state: PipelineState) -> dict:
    """Run all verification layers and return a retry/accept decision."""
    extractions: list[dict[str, Any]] = state.get("extractions", [])

    all_validation: list[dict[str, Any]] = []
    all_review: list[dict[str, Any]] = []
    all_failing: list[dict[str, Any]] = []
    confidence_scores: list[float] = []

    for ext_idx, extraction in enumerate(extractions):
        # Layer 1 — deterministic checks
        math = _math_checks(extraction)
        fmt = _format_checks(extraction)

        issues = math["issues"] + fmt["issues"]

        # Layer 2 — amount-in-words cross-check
        _amount_words_check(extraction, ext_idx, all_review, issues)

        # Layer 3 — deterministic confidence scoring
        ext_confidence, ext_failing = _compute_confidence_and_failures(
            extraction, ext_idx, math, fmt,
        )
        confidence_scores.append(ext_confidence)
        all_failing.extend(ext_failing)

        all_validation.append({
            "extraction_index": ext_idx,
            "math": math,
            "format": fmt,
            "issues": issues,
        })

    # Average confidence across all extractions
    if confidence_scores:
        final_confidence = sum(confidence_scores) / len(confidence_scores)
    else:
        final_confidence = 0.0

    retry_count: int = state.get("retry_count", 0)

    logger.info(
        "verify_complete",
        final_confidence=round(final_confidence, 4),
        fields_for_review=len(all_review),
        fields_failing=len(all_failing),
        retry_count=retry_count,
    )

    return {
        "validation_results": all_validation,
        "fields_for_review": all_review,
        "fields_failing": all_failing,
        "final_confidence": round(final_confidence, 4),
        "retry_count": retry_count,
    }


# ═══════════════════════════════════════════════════════════════════════
# Layer 1 helpers
# ═══════════════════════════════════════════════════════════════════════


def _math_checks(extraction: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    line_totals_valid = True
    tax_valid = True

    for item in extraction.get("line_items", []):
        qty = _num(item.get("quantity", 1))
        rate = _num(item.get("unit_price", 0))
        taxable = _num(item.get("taxable_amount", 0))
        line_total = _num(item.get("line_total", 0))

        # Prefer taxable_amount for qty×rate check (line_total may include tax)
        check_target = taxable if taxable else line_total
        if check_target and qty and rate and abs(qty * rate - check_target) > 1:
            line_totals_valid = False
            label = "taxable_amount" if taxable else "line_total"
            issues.append(
                f"Line {item.get('line_index')}: qty×rate "
                f"({qty}×{rate}={qty*rate}) ≠ {label} ({check_target})"
            )

        taxable = _num(item.get("taxable_amount", 0))
        for tax_key in ("cgst", "sgst", "igst"):
            rate_val = _num(item.get(f"{tax_key}_rate", 0))
            amount_val = _num(item.get(f"{tax_key}_amount", 0))
            if rate_val and amount_val and taxable:
                expected = taxable * rate_val / 100
                if abs(expected - amount_val) > 1:
                    tax_valid = False
                    issues.append(
                        f"Line {item.get('line_index')}: {tax_key} "
                        f"({taxable}×{rate_val}%={expected:.2f}) ≠ {amount_val}"
                    )

    subtotal = _num(_scalar_from_totals_or_root(extraction, "subtotal_before_tax"))
    taxable_sum = sum(
        _num(i.get("taxable_amount", 0))
        for i in extraction.get("line_items", [])
    )
    subtotal_valid = not subtotal or abs(taxable_sum - subtotal) <= 1
    if subtotal and not subtotal_valid:
        issues.append(
            f"Subtotal mismatch: sum(taxable)={taxable_sum} ≠ subtotal={subtotal}"
        )

    grand_total = _num(_scalar_from_totals_or_root(extraction, "grand_total"))
    total_tax = _num(_scalar_from_totals_or_root(extraction, "total_tax"))
    rounding = _num(_scalar_from_totals_or_root(extraction, "rounding_off"))
    grand_total_valid = (
        not grand_total
        or not subtotal
        or abs(subtotal + total_tax + rounding - grand_total) <= 1
    )
    if grand_total and subtotal and not grand_total_valid:
        issues.append(
            f"Grand total mismatch: {subtotal}+{total_tax}+rounding({rounding})="
            f"{subtotal+total_tax+rounding} ≠ {grand_total}"
        )

    return {
        "line_totals_valid": line_totals_valid,
        "subtotal_valid": subtotal_valid,
        "tax_valid": tax_valid,
        "grand_total_valid": grand_total_valid,
        "issues": issues,
    }


def _format_checks(extraction: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []

    # GSTIN (top-level or nested vendor_details / buyer_details)
    gstin_valid: bool | None = None
    for key, raw in _collect_gstin_candidates(extraction):
        val = _scalar(raw)
        if val is None or val == "":
            continue
        clean = str(val).strip().upper()
        if not _GSTIN_RE.match(clean):
            gstin_valid = False
            issues.append(f"{key} '{val}' fails GSTIN format check")
        elif int(clean[:2]) not in _VALID_STATE_CODES:
            gstin_valid = False
            issues.append(f"{key} '{val}' has invalid state code {clean[:2]}")
        else:
            gstin_valid = True if gstin_valid is None else gstin_valid

    # PAN
    pan_valid: bool | None = None
    for key, raw in _collect_pan_candidates(extraction):
        val = _scalar(raw)
        if val is None or val == "":
            continue
        if _PAN_RE.match(str(val).strip().upper()):
            pan_valid = True if pan_valid is None else pan_valid
        else:
            pan_valid = False
            issues.append(f"{key} '{val}' fails PAN format check")

    # Dates — must not be empty when confidence is reasonable
    dates_valid = True
    ih = extraction.get("invoice_header")
    for key in ("invoice_date", "due_date"):
        conf = extraction.get(f"{key}_confidence")
        val = extraction.get(key)
        if isinstance(ih, dict):
            leaf = ih.get(key)
            if isinstance(leaf, dict):
                if conf is None:
                    c = leaf.get("confidence")
                    conf = float(c) if isinstance(c, (int, float)) else None
                if val is None:
                    val = _scalar(leaf)
        if conf is not None and conf > 0.5 and not val:
            dates_valid = False
            issues.append(f"{key} is empty despite confidence {conf}")

    # HSN/SAC codes
    hsn_valid: bool | None = None
    for item in extraction.get("line_items", []):
        code = _scalar(item.get("hsn_sac_code"))
        if not code:
            continue
        if _HSN_RE.match(str(code).strip()):
            hsn_valid = True if hsn_valid is None else hsn_valid
        else:
            hsn_valid = False
            issues.append(f"HSN/SAC '{code}' is not 4-8 digits")

    # Positive amounts
    for item in extraction.get("line_items", []):
        for amt_key in ("line_total", "taxable_amount", "unit_price"):
            val = _num(item.get(amt_key))
            if val < 0:
                issues.append(
                    f"Line {item.get('line_index')}: {amt_key}={val} is negative"
                )

    return {
        "gstin_valid": gstin_valid,
        "pan_valid": pan_valid,
        "dates_valid": dates_valid,
        "hsn_valid": hsn_valid,
        "issues": issues,
    }


# ═══════════════════════════════════════════════════════════════════════
# Layer 2 — amount-in-words
# ═══════════════════════════════════════════════════════════════════════

_WORD_VALUES: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_MULTIPLIERS: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 1_00_000,
    "lac": 1_00_000,
    "lakhs": 1_00_000,
    "lacs": 1_00_000,
    "crore": 1_00_00_000,
    "crores": 1_00_00_000,
    # Hindi equivalents
    "करोड़": 1_00_00_000,
    "लाख": 1_00_000,
    "हज़ार": 1_000,
    "हजार": 1_000,
    "सौ": 100,
}


def _parse_indian_amount_words(text: str) -> float | None:
    """
    Parse Indian-English (or Hindi) amount words into a numeric value.

    Handles patterns like "Twelve Lakh Thirty-Four Thousand Five Hundred
    Sixty-Seven Rupees and Eighty-Nine Paise Only".
    """
    if not text:
        return None

    cleaned = (
        text.lower()
        .replace("-", " ")
        .replace(",", "")
        .replace("rupees", "")
        .replace("rupee", "")
        .replace("only", "")
        .replace("rs.", "")
        .replace("rs", "")
        .replace("inr", "")
        .strip()
    )

    paise_part = 0.0
    if "paise" in cleaned or "paisa" in cleaned:
        parts = re.split(r"\band\b|paise|paisa", cleaned)
        if len(parts) >= 2:
            paise_val = _words_to_number(parts[-2].strip() if len(parts) > 1 else "")
            if paise_val is not None:
                paise_part = paise_val / 100
        # The rupee part is everything before "and" (or empty if paise-only)
        if "and" in cleaned:
            cleaned = re.split(r"\band\b", cleaned)[0].strip()
        else:
            cleaned = ""  # paise-only, no rupee amount

    main = _words_to_number(cleaned)
    if main is None and paise_part > 0:
        return paise_part
    if main is None:
        return None
    return main + paise_part


def _words_to_number(text: str) -> float | None:
    """Convert a sequence of English/Hindi number words to an integer."""
    if not text:
        return None

    tokens = text.split()
    if not tokens:
        return None

    total = 0
    current = 0

    for token in tokens:
        word = token.strip().lower()
        if not word or word in ("and",):
            continue

        if word in _WORD_VALUES:
            current += _WORD_VALUES[word]
        elif word in _MULTIPLIERS:
            mult = _MULTIPLIERS[word]
            if mult == 100:
                current *= mult
            elif current == 0:
                current = mult
            else:
                if mult >= 1_00_000:
                    total += current * mult
                    current = 0
                else:
                    current *= mult
                    total += current
                    current = 0
        else:
            try:
                current += float(word)
            except ValueError:
                return None

    total += current
    return float(total) if total or current == 0 else None


def _amount_words_check(
    extraction: dict[str, Any],
    ext_idx: int,
    review_list: list[dict[str, Any]],
    issues: list[str],
) -> None:
    words_raw = _scalar_from_totals_or_root(extraction, "amount_in_words")
    words = words_raw if isinstance(words_raw, str) else extraction.get("amount_in_words")
    if words is not None and not isinstance(words, str):
        words = _scalar(words)
    grand_total = _num(_scalar_from_totals_or_root(extraction, "grand_total"))
    if not words or not grand_total:
        return

    parsed = _parse_indian_amount_words(words)
    if parsed is None:
        issues.append(f"Could not parse amount_in_words: '{words}'")
        return

    if abs(parsed - grand_total) <= 10:
        logger.info(
            "amount_words_match",
            parsed=parsed,
            grand_total=grand_total,
        )
    else:
        issues.append(
            f"amount_in_words ({parsed}) ≠ grand_total ({grand_total})"
        )
        review_list.append({
            "extraction_index": ext_idx,
            "field": "grand_total",
            "original_value": grand_total,
            "verified_value": parsed,
            "reason": "amount_in_words mismatch",
        })


# ═══════════════════════════════════════════════════════════════════════
# Layer 3 — deterministic confidence & retry decision
# ═══════════════════════════════════════════════════════════════════════


def _compute_confidence_and_failures(
    extraction: dict[str, Any],
    ext_idx: int,
    math: dict[str, Any],
    fmt: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    """Return (confidence_score, failing_fields) based on deterministic checks."""
    failing: list[dict[str, Any]] = []
    confidence = 0.95  # Start optimistic

    # Math failures are critical
    if not math["line_totals_valid"]:
        confidence = min(confidence, 0.70)
        failing.append({"extraction_index": ext_idx, "field": "line_totals", "reason": "math mismatch"})
    if not math["grand_total_valid"]:
        confidence = min(confidence, 0.70)
        failing.append({"extraction_index": ext_idx, "field": "grand_total", "reason": "math mismatch"})
    if not math.get("subtotal_valid", True):
        confidence = min(confidence, 0.75)
        failing.append({"extraction_index": ext_idx, "field": "subtotal", "reason": "math mismatch"})
    if not math.get("tax_valid", True):
        confidence = min(confidence, 0.75)
        failing.append({"extraction_index": ext_idx, "field": "tax", "reason": "tax calculation mismatch"})

    # Format failures are moderate
    if fmt.get("gstin_valid") is False:
        confidence = min(confidence, 0.80)
        failing.append({"extraction_index": ext_idx, "field": "gstin", "reason": "format invalid"})
    if fmt.get("pan_valid") is False:
        confidence = min(confidence, 0.85)

    # Missing critical fields
    grand_total = _num(_scalar_from_totals_or_root(extraction, "grand_total"))
    if not grand_total:
        confidence = min(confidence, 0.50)
        failing.append({"extraction_index": ext_idx, "field": "grand_total", "reason": "missing"})

    # Check vendor and invoice number exist
    vd = extraction.get("vendor_details", {})
    vendor_name = _scalar(vd.get("vendor_name")) if isinstance(vd, dict) else None
    if not vendor_name:
        confidence = min(confidence, 0.60)

    ih = extraction.get("invoice_header", {})
    inv_num = _scalar(ih.get("invoice_number")) if isinstance(ih, dict) else None
    if not inv_num:
        confidence = min(confidence, 0.60)

    return confidence, failing


# ── Numeric helper ───────────────────────────────────────────────────


def _num(val: Any) -> float:
    """Safely cast to float, defaulting to 0."""
    if val is None:
        return 0.0
    v = _scalar(val)
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
