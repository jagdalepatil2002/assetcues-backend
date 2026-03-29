"""
Extract node — single-call extraction for maximum accuracy.

This is the core extraction node of the Assetcues Invoice Agentic AI pipeline.
It sends each page image to the VLM once with a simplified prompt, validates
math consistency, expands assets, and returns structured results.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from far_ai_brain.ai.vlm_adapter import VLMAdapter
from far_ai_brain.config.settings import settings
from far_ai_brain.prompts.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    MULTI_PAGE_CHUNK_EXTRACTION_PROMPT,
    MULTI_PAGE_EXTRACTION_PROMPT,
    SIMPLE_EXTRACTION_PROMPT,
    SIMPLE_EXTRACTION_SYSTEM_PROMPT,
    WEB_STYLE_ASSET_ENTRY_PROMPT,
)
from far_ai_brain.schemas.state import PageGroup, PipelineState, RegionCrop

logger = structlog.get_logger()

_ROUNDING_TOLERANCE = 1.0  # ₹1 rounding tolerance for math checks


# ── Main node ─────────────────────────────────────────────────────────


async def extract_node(state: PipelineState) -> dict:
    """
    Core extraction node.  Processes each page_group independently using
    a single VLM call per page, validates math consistency, and expands
    assets.

    Returns ``extractions``, ``region_extractions``, and ``source_maps``
    (one entry per page_group).
    """
    start = time.monotonic()
    page_groups: list[PageGroup] = state.get("page_groups", [])
    region_crops: list[RegionCrop] = state.get("region_crops", [])
    multi_res: list[dict[str, bytes | None]] = state.get("multi_res_images", [])
    thinking_level: str = state.get("thinking_level", "low")
    mode: str = state.get("mode", "creation")

    if not page_groups:
        page_groups = [
            PageGroup(
                group_index=0,
                page_indices=list(range(state.get("page_count", 1))),
                page_images=state.get("page_images", []),
            )
        ]

    all_extractions: list[dict[str, Any]] = []
    all_region_extractions: list[dict[str, dict[str, Any]]] = []
    all_source_maps: list[dict[str, Any]] = []

    for group in page_groups:
        log = logger.bind(group_index=group["group_index"])
        try:
            extraction, region_raw, source_map = await _process_group(
                group=group,
                region_crops=region_crops,
                multi_res=multi_res,
                thinking_level=thinking_level,
                mode=mode,
                log=log,
            )
            all_extractions.append(extraction)
            all_region_extractions.append(region_raw)
            all_source_maps.append(source_map)
        except Exception:
            log.exception("group_extraction_failed")
            all_extractions.append({})
            all_region_extractions.append({})
            all_source_maps.append({})

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "extract_node_complete",
        groups_processed=len(page_groups),
        elapsed_ms=elapsed_ms,
    )

    return {
        "extractions": all_extractions,
        "region_extractions": all_region_extractions,
        "source_maps": all_source_maps,
    }


# ── Per-group orchestration ──────────────────────────────────────────


async def _process_group(
    *,
    group: PageGroup,
    region_crops: list[RegionCrop],
    multi_res: list[dict[str, bytes | None]],
    thinking_level: str,
    mode: str,
    log: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the full extract → validate → expand pipeline for one group."""
    adapter = VLMAdapter(role="primary")
    page_images: list[bytes] = group["page_images"]
    page_indices: list[int] = group["page_indices"]
    page_count = len(page_indices)

    # ── Step 1: Extract ───────────────────────────────────────────────
    if page_count > 1:
        extraction, region_raw, source_map = await _extract_multi_page(
            adapter=adapter,
            page_images=page_images,
            page_count=page_count,
            log=log,
        )
    elif mode == "web_like":
        extraction, region_raw, source_map = await _extract_full_page_web_style(
            adapter=adapter,
            page_image=page_images[0],
            thinking_level=thinking_level,
            log=log,
        )
    else:
        extraction, region_raw, source_map = await _extract_full_page(
            adapter=adapter,
            page_image=page_images[0],
            thinking_level=thinking_level,
            log=log,
        )

    # ── Step 2: Math consistency check ────────────────────────────────
    if mode != "web_like":
        math_issues = _check_math_consistency(extraction)
        if math_issues:
            log.warning("math_issues_detected", issues=math_issues, count=len(math_issues))
            extraction.setdefault("validation_results", {}).setdefault(
                "math_check", {}
            )["issues"] = math_issues

    # ── Step 4: Expand assets ─────────────────────────────────────────
    if mode == "web_like" and extraction.get("assets_to_create"):
        pass  # Keep VLM-provided assets for web_like mode
    else:
        extraction["assets_to_create"] = _expand_assets(extraction)

    # ── Step 5: Metadata ──────────────────────────────────────────────
    extraction.setdefault("extraction_meta", {}).update(
        {
            "extraction_id": str(uuid.uuid4()),
            "model_used": adapter.model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pages_processed": page_count,
        }
    )
    if mode == "web_like" and extraction["extraction_meta"].get("overall_confidence") is None:
        extraction["extraction_meta"]["overall_confidence"] = 0.85

    return extraction, region_raw, source_map


# ── Extraction strategies ─────────────────────────────────────────────


async def _extract_full_page_with_fallback(
    *,
    adapter: VLMAdapter,
    page_image: bytes,
    thinking_level: str,
    log: Any,
) -> dict[str, Any]:
    """Single VLM call with simple prompt; on failure retry without schema."""
    try:
        return await adapter.extract(
            images=[page_image],
            system_prompt=SIMPLE_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=SIMPLE_EXTRACTION_PROMPT,
            json_schema=None,
            thinking_level=thinking_level,
        )
    except Exception as e:
        log.warning(
            "full_extract_failed",
            error=str(e)[:300],
            fallback="retry_no_schema",
        )
        try:
            return await adapter.extract(
                images=[page_image],
                system_prompt=SIMPLE_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=SIMPLE_EXTRACTION_PROMPT
                + "\n\nRespond with one JSON object only (no markdown), matching invoice fields.",
                json_schema=None,
                thinking_level=thinking_level,
            )
        except Exception as e2:
            log.error("full_extract_all_attempts_failed", error=str(e2)[:300])
            return {
                "extraction_meta": {
                    "overall_confidence": 0.0,
                    "error": "vlm_failed",
                    "detail": str(e2)[:500],
                },
            }


async def _extract_full_page(
    *,
    adapter: VLMAdapter,
    page_image: bytes,
    thinking_level: str,
    log: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Single full-page extraction with simplified prompt."""
    log.info("full_page_extraction_start")
    result = await _extract_full_page_with_fallback(
        adapter=adapter,
        page_image=page_image,
        thinking_level=thinking_level,
        log=log,
    )

    source_map = {k: {"region": "full", "confidence": None} for k in result}
    return result, {"full": result}, source_map


async def _extract_full_page_web_style(
    *,
    adapter: VLMAdapter,
    page_image: bytes,
    thinking_level: str,
    log: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Single-call web-style extraction aimed at posting-ready asset entries."""
    log.info("full_page_web_style_extraction_start")
    try:
        result = await adapter.extract(
            images=[page_image],
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=WEB_STYLE_ASSET_ENTRY_PROMPT,
            json_schema=None,
            thinking_level=thinking_level,
        )
        if not isinstance(result, dict):
            result = {}
    except Exception as e:
        log.error("full_page_web_style_extract_failed", error=str(e)[:300])
        result = {
            "extraction_meta": {
                "overall_confidence": 0.0,
                "error": "vlm_failed",
                "detail": str(e)[:500],
            },
        }

    source_map = {k: {"region": "full_web_style", "confidence": None} for k in result}
    return result, {"full_web_style": result}, source_map


def _empty_merged_extraction() -> dict[str, Any]:
    """Skeleton dict for chunked multi-page merge."""
    return {
        "extraction_meta": {},
        "document_type": {},
        "vendor_details": {},
        "buyer_details": {},
        "invoice_header": {},
        "line_items": [],
        "totals": {},
        "bank_details": {},
        "warranty_amc_info": {},
        "assets_to_create": [],
        "raw_complete_extraction": {},
        "validation_results": {},
        "ai_reasoning_log": [],
    }


def _deep_merge_invoice_section(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Merge nested extraction dicts; prefer non-empty ConfidenceField-style leaves."""
    if not isinstance(src, dict):
        return
    for k, v in src.items():
        if v is None:
            continue
        if isinstance(v, dict):
            if "value" in v and "confidence" in v:
                if v.get("value") not in (None, "", [], {}):
                    dst[k] = v
                elif k not in dst:
                    dst[k] = v
                continue
            dst.setdefault(k, {})
            if isinstance(dst[k], dict):
                _deep_merge_invoice_section(dst[k], v)
            else:
                dst[k] = v
        elif isinstance(v, list):
            if v:
                dst[k] = v
        else:
            dst[k] = v


def _merge_chunk_into_merged(
    merged: dict[str, Any],
    part: dict[str, Any],
    *,
    chunk_index: int,
    chunk_count: int,
) -> None:
    header_keys = ("document_type", "vendor_details", "buyer_details", "invoice_header")
    footer_keys = ("totals", "bank_details", "warranty_amc_info")

    if chunk_index == 0:
        for hk in header_keys:
            if hk in part and isinstance(part[hk], dict):
                merged.setdefault(hk, {})
                if isinstance(merged[hk], dict):
                    _deep_merge_invoice_section(merged[hk], part[hk])

    if chunk_index == chunk_count - 1:
        for fk in footer_keys:
            if fk in part and isinstance(part[fk], dict):
                merged.setdefault(fk, {})
                if isinstance(merged[fk], dict):
                    _deep_merge_invoice_section(merged[fk], part[fk])

    em = part.get("extraction_meta")
    if isinstance(em, dict) and em:
        merged.setdefault("extraction_meta", {})
        if isinstance(merged["extraction_meta"], dict):
            _deep_merge_invoice_section(merged["extraction_meta"], em)

    raw = part.get("raw_complete_extraction")
    if isinstance(raw, dict) and raw.get("full_text_dump"):
        rdst = merged.setdefault("raw_complete_extraction", {})
        if not isinstance(rdst, dict):
            merged["raw_complete_extraction"] = {"full_text_dump": ""}
            rdst = merged["raw_complete_extraction"]
        prev = str(rdst.get("full_text_dump") or "")
        chunk_txt = str(raw.get("full_text_dump") or "")
        rdst["full_text_dump"] = (prev + ("\n\n---\n\n" if prev else "") + chunk_txt).strip()


async def _extract_multi_page_single_batch(
    *,
    adapter: VLMAdapter,
    page_images: list[bytes],
    page_count: int,
    log: Any,
) -> dict[str, Any]:
    prompt = MULTI_PAGE_EXTRACTION_PROMPT.format(page_count=page_count)

    try:
        return await adapter.extract(
            images=page_images,
            system_prompt=SIMPLE_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            json_schema=None,
            thinking_level="high",
        )
    except Exception as e:
        log.warning(
            "multi_page_extract_failed",
            error=str(e)[:300],
            fallback="retry",
        )
        try:
            return await adapter.extract(
                images=page_images,
                system_prompt=SIMPLE_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt
                + "\n\nRespond with one JSON object only (no markdown), matching invoice fields.",
                json_schema=None,
                thinking_level="high",
            )
        except Exception as e2:
            log.error("multi_page_all_attempts_failed", error=str(e2)[:300])
            return {
                "extraction_meta": {
                    "overall_confidence": 0.0,
                    "error": "vlm_failed",
                    "detail": str(e2)[:500],
                },
            }


async def _extract_multi_page_chunked(
    *,
    adapter: VLMAdapter,
    page_images: list[bytes],
    page_count: int,
    chunk_size: int,
    log: Any,
) -> dict[str, Any]:
    merged = _empty_merged_extraction()
    next_line_index = 1
    chunks: list[list[bytes]] = [
        page_images[i : i + chunk_size] for i in range(0, len(page_images), chunk_size)
    ]
    chunk_count = len(chunks)

    log.info(
        "multi_page_chunked_start",
        page_count=page_count,
        chunk_size=chunk_size,
        chunk_count=chunk_count,
    )

    for ci, chunk_imgs in enumerate(chunks):
        page_lo = ci * chunk_size + 1
        page_hi = page_lo + len(chunk_imgs) - 1
        prompt = MULTI_PAGE_CHUNK_EXTRACTION_PROMPT.format(
            page_lo=page_lo,
            page_hi=page_hi,
            page_total=page_count,
            next_line_index=next_line_index,
        )
        try:
            part = await adapter.extract(
                images=chunk_imgs,
                system_prompt=SIMPLE_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                json_schema=None,
                thinking_level="high",
            )
        except Exception as e:
            log.error(
                "multi_page_chunk_failed",
                chunk_index=ci,
                error=str(e)[:300],
            )
            return {
                "extraction_meta": {
                    "overall_confidence": 0.0,
                    "error": "vlm_chunk_failed",
                    "detail": f"chunk {ci + 1}/{chunk_count}: {str(e)[:400]}",
                },
                "line_items": merged.get("line_items", []),
            }

        raw_lines = part.get("line_items") or []
        if isinstance(raw_lines, list):
            sorted_lines = sorted(
                raw_lines,
                key=lambda x: int(x.get("line_index", 0) or 0) if isinstance(x, dict) else 0,
            )
            for item in sorted_lines:
                if not isinstance(item, dict):
                    continue
                ni = dict(item)
                ni["line_index"] = next_line_index
                next_line_index += 1
                merged["line_items"].append(ni)

        _merge_chunk_into_merged(merged, part, chunk_index=ci, chunk_count=chunk_count)

    return merged


async def _extract_multi_page(
    *,
    adapter: VLMAdapter,
    page_images: list[bytes],
    page_count: int,
    log: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Multi-page single invoice: one VLM call up to chunk limit, else batched merge."""
    log.info("multi_page_extraction_start", page_count=page_count)
    max_p = max(1, settings.vlm_max_pages_per_request)

    if page_count <= max_p:
        result = await _extract_multi_page_single_batch(
            adapter=adapter,
            page_images=page_images,
            page_count=page_count,
            log=log,
        )
    else:
        result = await _extract_multi_page_chunked(
            adapter=adapter,
            page_images=page_images,
            page_count=page_count,
            chunk_size=max_p,
            log=log,
        )

    source_map = {k: {"region": "multi_page", "confidence": None} for k in result}
    return result, {"multi_page": result}, source_map


# ── Math consistency check ────────────────────────────────────────────


def _check_math_consistency(extraction: dict[str, Any]) -> list[str]:
    """Validate arithmetic relationships between extracted amounts."""
    issues: list[str] = []
    line_items = extraction.get("line_items", [])
    totals = extraction.get("totals", {})
    tol = _ROUNDING_TOLERANCE

    for i, item in enumerate(line_items):
        qty = _conf_num(item.get("quantity"))
        unit_price = _conf_num(item.get("unit_price"))
        line_total = _conf_num(item.get("line_total"))
        taxable = _conf_num(item.get("taxable_amount"))

        if qty is not None and unit_price is not None:
            expected = qty * unit_price
            target = taxable if taxable is not None else line_total
            if target is not None and abs(expected - target) > tol:
                issues.append(
                    f"Line {i}: qty({qty}) × unit_price({unit_price}) = "
                    f"{expected:.2f}, but {'taxable' if taxable is not None else 'line_total'} = {target}"
                )

    subtotal = _conf_num(totals.get("subtotal_before_tax"))
    if subtotal is not None and line_items:
        line_sum = sum(
            _conf_num(it.get("taxable_amount")) or _conf_num(it.get("line_total")) or 0
            for it in line_items
        )
        if line_sum > 0 and abs(line_sum - subtotal) > tol:
            issues.append(f"sum(line_taxable) = {line_sum:.2f}, but subtotal = {subtotal}")

    grand_total = _conf_num(totals.get("grand_total"))
    total_tax = _conf_num(totals.get("total_tax"))
    rounding = _conf_num(totals.get("rounding_off")) or 0.0

    if grand_total is not None and subtotal is not None and total_tax is not None:
        expected_grand = subtotal + total_tax + rounding
        if abs(expected_grand - grand_total) > tol:
            issues.append(
                f"subtotal({subtotal}) + tax({total_tax}) + rounding({rounding}) = "
                f"{expected_grand:.2f}, but grand_total = {grand_total}"
            )

    return issues


# ── Asset expansion ───────────────────────────────────────────────────


def _expand_assets(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand line items with quantity > 1 into individual asset entries.
    
    Unit-aware: Bulk units (Kg, L, Ton, etc.) create a single asset
    with the full quantity as metadata, not N individual assets.
    Countable units (Nos, Pcs, Sets) expand normally.
    """
    BULK_UNITS = {
        "kg", "kgs", "g", "gm", "gram", "grams",
        "l", "ltr", "litre", "litres", "liter", "liters", "ml",
        "mt", "ton", "tons", "tonne", "tonnes", "quintal", "qtl",
        "mtr", "meter", "meters", "metre", "ft", "feet", "sqft",
        "mm", "cm", "inch", "inches",
    }

    assets: list[dict[str, Any]] = []
    counter = 1
    line_items = extraction.get("line_items", [])
    invoice_date = _conf_str(
        extraction.get("invoice_header", {}).get("invoice_date")
    )

    for item in line_items:
        qty_raw = _conf_num(item.get("quantity"))
        qty = max(round(qty_raw), 1) if qty_raw is not None else 1

        # Check unit type — bulk units create single asset
        unit_raw = _conf_str(item.get("unit")) or "Nos"
        unit_normalized = unit_raw.lower().strip().rstrip(".")
        is_bulk = unit_normalized in BULK_UNITS

        if is_bulk:
            # Bulk asset: don't expand, keep as single asset with full qty
            expand_qty = 1
            bulk_quantity = qty_raw  # Original quantity (e.g., 100 for 100kg)
        else:
            expand_qty = qty
            bulk_quantity = None

        taxable = _conf_num(item.get("taxable_amount")) or 0.0
        line_total = _conf_num(item.get("line_total")) or taxable
        tax_per_line = (
            (_conf_num(item.get("cgst_amount")) or 0.0)
            + (_conf_num(item.get("sgst_amount")) or 0.0)
            + (_conf_num(item.get("igst_amount")) or 0.0)
        )

        unit_cost = taxable / expand_qty if expand_qty else taxable
        unit_tax = tax_per_line / expand_qty if expand_qty else tax_per_line
        unit_total = unit_cost + unit_tax

        description = _conf_str(item.get("description")) or ""
        serials_raw = item.get("serial_numbers_listed", [])
        serials: list[str] = serials_raw if isinstance(serials_raw, list) else []
        group_action = item.get("group_action", "none")
        group_parent = item.get("group_parent_temp_id")
        group_reason = item.get("group_reason")

        category, sub_cat, asset_class = _suggest_category(description)
        physical = _is_physical_asset(description)

        for q in range(expand_qty):
            temp_id = f"tmp_ast_{counter:03d}"
            assets.append(
                {
                    "temp_asset_id": temp_id,
                    "source_line_index": item.get("line_index", 0),
                    "asset_name": description,
                    "quantity_index": q + 1,
                    "quantity_total": expand_qty,
                    "individual_cost_before_tax": round(unit_cost, 2),
                    "individual_tax": round(unit_tax, 2),
                    "individual_cost_with_tax": round(unit_total, 2),
                    "serial_number": serials[q] if q < len(serials) else None,
                    "suggested_category": category,
                    "suggested_sub_category": sub_cat,
                    "suggested_asset_class": asset_class,
                    "date_of_acquisition": invoice_date,
                    "audit_indicator": "physical" if physical else "non_physical",
                    "audit_method": _audit_method(physical, unit_total),
                    "group_action": group_action if q == 0 else "none",
                    "group_parent_temp_id": group_parent if q == 0 else None,
                    "group_reason": group_reason if q == 0 else None,
                    "confidence_overall": None,
                    # Bulk/unit metadata
                    "unit_of_measure": unit_raw,
                    "bulk_quantity": bulk_quantity,
                    "is_bulk_asset": is_bulk,
                }
            )
            counter += 1

    return assets


# ── Shared helpers ────────────────────────────────────────────────────


def _conf_num(field: Any) -> float | None:
    """Extract a numeric value from a ConfidenceField dict or raw value."""
    if field is None:
        return None
    if isinstance(field, dict):
        v = field.get("value")
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
    try:
        return float(field)
    except (ValueError, TypeError):
        return None


def _conf_str(field: Any) -> str | None:
    """Extract a string value from a ConfidenceField dict or raw value."""
    if field is None:
        return None
    if isinstance(field, dict):
        v = field.get("value")
        return str(v) if v is not None else None
    return str(field) if field else None


# ── Category suggestion ───────────────────────────────────────────────

_CATEGORY_MAP: list[tuple[list[str], tuple[str, str, str]]] = [
    (["laptop", "notebook"], ("IT Equipment", "Computing", "Laptop")),
    (["desktop", "pc", "workstation"], ("IT Equipment", "Computing", "Desktop")),
    (["server"], ("IT Equipment", "Infrastructure", "Server")),
    (["monitor", "display", "screen"], ("IT Equipment", "Peripherals", "Monitor")),
    (["printer", "scanner", "copier", "mfp"], ("IT Equipment", "Peripherals", "Printer")),
    (["router", "switch", "firewall", "access point"], ("IT Equipment", "Networking", "Network Equipment")),
    (["ups", "battery"], ("Electrical Equipment", "Power", "UPS")),
    (["ac", "air conditioner", "hvac"], ("Electrical Equipment", "Climate", "Air Conditioner")),
    (["chair", "table", "desk", "cabinet", "rack", "shelf", "almirah", "cupboard"], ("Furniture", "Office Furniture", "Furniture")),
    (["phone", "mobile", "tablet", "ipad"], ("IT Equipment", "Mobile Devices", "Mobile Device")),
    (["camera", "cctv", "surveillance"], ("Security Equipment", "Surveillance", "Camera")),
    (["projector"], ("IT Equipment", "Peripherals", "Projector")),
    (["software", "license", "subscription"], ("Intangible Assets", "Software", "Software License")),
    (["vehicle", "car", "bike", "scooter"], ("Vehicles", "Transport", "Vehicle")),
    (["generator", "genset"], ("Plant & Machinery", "Power", "Generator")),
]

_NON_PHYSICAL_KEYWORDS = frozenset({
    "software", "license", "subscription", "saas", "warranty",
    "amc", "service", "installation", "training",
})


def _suggest_category(description: str) -> tuple[str | None, str | None, str | None]:
    """Match description keywords to standard Indian asset categories."""
    lower = description.lower()
    for keywords, triple in _CATEGORY_MAP:
        if any(kw in lower for kw in keywords):
            return triple
    return ("Office Equipment", "General", None)


def _is_physical_asset(description: str) -> bool:
    lower = description.lower()
    return not any(kw in lower for kw in _NON_PHYSICAL_KEYWORDS)


def _audit_method(is_physical: bool, cost: float) -> str:
    if not is_physical:
        return "document_verification"
    if cost > 100_000:
        return "physical_verification_with_photo"
    return "visual_inspection"
