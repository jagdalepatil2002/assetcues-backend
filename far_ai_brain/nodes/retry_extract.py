"""
Retry-extract node — re-reads ONLY the fields that failed verification,
using focused region crops for maximum accuracy.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from far_ai_brain.ai.vlm_adapter import VLMAdapter
from far_ai_brain.prompts.extraction import (
    TARGETED_REREAD_AMOUNT_PROMPT,
    TARGETED_REREAD_TEXT_PROMPT,
)
from far_ai_brain.config.settings import settings
from far_ai_brain.schemas.state import PipelineState
from far_ai_brain.utils.image import crop_field_region, crop_top_header_strip

logger = structlog.get_logger()

_AMOUNT_FIELDS = frozenset({
    "grand_total", "subtotal_before_tax", "total_tax",
    "total_cgst", "total_sgst", "total_igst",
    "individual_cost_before_tax", "individual_cost_with_tax",
    "line_total", "taxable_amount", "unit_price",
    "cgst_amount", "sgst_amount", "igst_amount",
    "rounding_off",
})

_TOTALS_FIELDS = frozenset({
    "grand_total", "subtotal_before_tax", "total_tax",
    "total_cgst", "total_sgst", "total_igst",
    "rounding_off", "amount_in_words",
})


async def retry_extract_node(state: PipelineState) -> dict:
    """Re-extract every failing field from its region crop."""
    fields_failing: list[dict[str, Any]] = state.get("fields_failing", [])
    extractions: list[dict[str, Any]] = list(state.get("extractions", []))
    source_maps: list[dict[str, Any]] = state.get("source_maps", [])
    page_images: list[bytes] = state.get("page_images", [])
    retry_count: int = state.get("retry_count", 0) + 1

    if not fields_failing:
        logger.info("retry_extract_skip", reason="no failing fields")
        return {"extractions": extractions, "retry_count": retry_count}

    tasks: list[dict[str, Any]] = []
    for failing in fields_failing:
        task = _prepare_reread_task(
            failing, extractions, source_maps, page_images,
        )
        if task is not None:
            tasks.append(task)

    if not tasks:
        logger.info("retry_extract_skip", reason="no crops available")
        return {"extractions": extractions, "retry_count": retry_count}

    adapter = VLMAdapter(role="primary")
    # Limit concurrency to avoid overwhelming the VLM API
    sem = asyncio.Semaphore(settings.vlm_concurrency_limit)

    async def _throttled_query(t: dict[str, Any]) -> str:
        async with sem:
            return await adapter.simple_query(prompt=t["prompt"], image=t["crop"])

    coros = [_throttled_query(t) for t in tasks]
    results = await asyncio.gather(*coros, return_exceptions=True)

    updated = 0
    for task_meta, result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.warning(
                "retry_reread_failed",
                field=task_meta["field"],
                error=str(result),
            )
            continue

        raw = str(result).strip()
        if not raw:
            continue

        value: Any
        if task_meta["is_amount"]:
            value = _parse_amount(raw)
            if value is None:
                logger.warning(
                    "retry_amount_parse_failed",
                    field=task_meta["field"],
                    raw=raw,
                )
                continue
        else:
            value = raw

        _write_back(
            extractions,
            task_meta["ext_idx"],
            task_meta["field"],
            value,
        )
        updated += 1
        logger.info(
            "retry_field_updated",
            field=task_meta["field"],
            ext_idx=task_meta["ext_idx"],
            new_value=value,
        )

    logger.info(
        "retry_extract_complete",
        attempted=len(tasks),
        updated=updated,
        retry_count=retry_count,
    )

    # Only remove fields that were actually fixed; keep the rest as still-failing
    fixed_keys = set()
    for task_meta, result in zip(tasks, results):
        if isinstance(result, Exception):
            continue
        raw = str(result).strip()
        if not raw:
            continue
        if task_meta["is_amount"] and _parse_amount(raw) is None:
            continue
        fixed_keys.add((task_meta["ext_idx"], task_meta["field"]))

    still_failing = [
        f for f in fields_failing
        if (f.get("extraction_index", 0), f.get("field", "")) not in fixed_keys
    ]

    return {
        "extractions": extractions,
        "retry_count": retry_count,
        "fields_failing": still_failing,
    }


# ── Helpers ──────────────────────────────────────────────────────────


def _prepare_reread_task(
    failing: dict[str, Any],
    extractions: list[dict[str, Any]],
    source_maps: list[dict[str, Any]],
    page_images: list[bytes],
) -> dict[str, Any] | None:
    """Build a reread task dict with crop bytes and the right prompt, or None."""
    ext_idx: int = failing.get("extraction_index", 0)
    field: str = failing.get("field", "")

    if ext_idx >= len(source_maps) or ext_idx >= len(extractions):
        logger.warning("retry_skip_no_source", field=field, ext_idx=ext_idx)
        return None

    smap = source_maps[ext_idx]
    field_meta: dict[str, Any] = smap.get(field, {})
    bbox = field_meta.get("bbox")

    crop: bytes | None = None
    page_idx: int = int(field_meta.get("page_index", 0))

    if bbox:
        if page_idx >= len(page_images):
            logger.warning("retry_skip_bad_page", field=field, page_idx=page_idx)
            return None
        crop = crop_field_region(page_images[page_idx], bbox)
    elif field == "gstin" and page_images:
        # Multi-page extract has no per-field bbox; GSTIN is usually in page-1 header band.
        crop = crop_top_header_strip(page_images[0])
        page_idx = 0
    else:
        logger.info("retry_skip_no_bbox", field=field)
        return None

    if crop is None:
        logger.warning("retry_crop_failed", field=field)
        return None

    is_amount = _is_amount_field(field)
    prompt = TARGETED_REREAD_AMOUNT_PROMPT if is_amount else TARGETED_REREAD_TEXT_PROMPT

    return {
        "ext_idx": ext_idx,
        "field": field,
        "crop": crop,
        "prompt": prompt,
        "is_amount": is_amount,
    }


def _is_amount_field(field: str) -> bool:
    """Determine whether a field path refers to a numeric amount."""
    if field in _AMOUNT_FIELDS:
        return True
    if "." in field:
        _, key = field.rsplit(".", 1)
        return key in _AMOUNT_FIELDS
    return False


def _parse_amount(raw: str) -> float | None:
    """Parse a VLM amount response into a float."""
    cleaned = raw.replace(",", "").replace("₹", "").replace("Rs", "").strip()
    cleaned = cleaned.rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _write_back(
    extractions: list[dict[str, Any]],
    ext_idx: int,
    field: str,
    value: Any,
) -> None:
    """Write a re-read value back into the correct extraction location."""
    if ext_idx >= len(extractions):
        return

    extraction = extractions[ext_idx]

    # Handle GSTIN — differentiate vendor vs buyer
    if field in ("gstin", "vendor_gstin"):
        container = extraction.setdefault("vendor_details", {})
        if not isinstance(container, dict):
            extraction["vendor_details"] = {}
            container = extraction["vendor_details"]
        leaf = container.get("vendor_gstin")
        if isinstance(leaf, dict):
            leaf["value"] = str(value).strip().upper()
            leaf["confidence"] = 0.92
        else:
            container["vendor_gstin"] = {
                "value": str(value).strip().upper(),
                "confidence": 0.92,
            }
        return

    if field == "buyer_gstin":
        container = extraction.setdefault("buyer_details", {})
        if not isinstance(container, dict):
            extraction["buyer_details"] = {}
            container = extraction["buyer_details"]
        leaf = container.get("buyer_gstin")
        if isinstance(leaf, dict):
            leaf["value"] = str(value).strip().upper()
            leaf["confidence"] = 0.92
        else:
            container["buyer_gstin"] = {
                "value": str(value).strip().upper(),
                "confidence": 0.92,
            }
        return

    if "." in field:
        parts = field.split(".", 1)
        line_prefix, key = parts
        try:
            line_idx = int(line_prefix.split("_")[1])
        except (IndexError, ValueError):
            extraction[field] = value
            return
        for item in extraction.get("line_items", []):
            if item.get("line_index") == line_idx:
                item[key] = value
                return
    else:
        # Write to nested totals dict if the field belongs there
        if field in _TOTALS_FIELDS:
            totals = extraction.get("totals")
            if isinstance(totals, dict) and field in totals:
                leaf = totals[field]
                if isinstance(leaf, dict) and "value" in leaf:
                    leaf["value"] = value
                    leaf["confidence"] = 0.92
                else:
                    totals[field] = value
                return
        extraction[field] = value
