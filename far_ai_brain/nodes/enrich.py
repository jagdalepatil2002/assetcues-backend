"""
Enrich node — post-extraction enrichment for extracted assets.

Two enrichments run concurrently per extraction:

1. Split research (IND AS 16 componentization)
   - Runs on any asset with individual cost >= RESEARCH_SPLIT_MIN_VALUE (default ₹1L)
   - Asks the VLM: should this asset be broken into components with different
     useful lives / depreciation rates per Schedule II?
   - Returns split_suggestions: [{temp_asset_id, should_split, components, reason}]

2. Group validation
   - The extraction node already flags assets with group_action="child"/"accessory"
     and a group_parent_temp_id pointing to their proposed parent.
   - This node validates those suggestions with a second, focused VLM call.
   - Returns group_suggestions: [{child_temp_id, parent_temp_id, confirmed, reason}]

Both enrichments are optional — failure silently skips, never blocks the pipeline.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from far_ai_brain.ai.vlm_adapter import VLMAdapter
from far_ai_brain.config.settings import settings
from far_ai_brain.prompts.group_validation import GROUP_VALIDATION_PROMPT_TEMPLATE
from far_ai_brain.prompts.split_research import (
    SPLIT_RESEARCH_PROMPT_TEMPLATE,
    SPLIT_RESEARCH_SYSTEM_PROMPT,
)
from far_ai_brain.schemas.state import PipelineState

logger = structlog.get_logger()


async def enrich_node(state: PipelineState) -> dict:
    """Run split research and group validation on all extracted assets."""
    extractions: list[dict[str, Any]] = state.get("extractions", [])
    if not extractions:
        return {"split_suggestions": [], "group_suggestions": []}

    adapter = VLMAdapter(role="cheap")
    sem = asyncio.Semaphore(settings.vlm_concurrency_limit)

    tasks = [
        _enrich_extraction(adapter, sem, extraction, ext_idx)
        for ext_idx, extraction in enumerate(extractions)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    split_suggestions: list[dict[str, Any]] = []
    group_suggestions: list[dict[str, Any]] = []

    for result in results:
        if isinstance(result, Exception):
            logger.warning("enrich_extraction_failed", error=str(result)[:200])
            continue
        splits, groups = result
        split_suggestions.extend(splits)
        group_suggestions.extend(groups)

    logger.info(
        "enrich_complete",
        split_count=len(split_suggestions),
        group_count=len(group_suggestions),
    )
    return {
        "split_suggestions": split_suggestions,
        "group_suggestions": group_suggestions,
    }


# ── Per-extraction ────────────────────────────────────────────────────


async def _enrich_extraction(
    adapter: VLMAdapter,
    sem: asyncio.Semaphore,
    extraction: dict[str, Any],
    ext_idx: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assets: list[dict[str, Any]] = extraction.get("assets_to_create", [])
    if not assets:
        return [], []

    vendor = _get_vendor(extraction)

    split_coros = []
    group_coros = []

    for asset in assets:
        cost = _asset_cost(asset)

        # Split research: only for high-value assets
        if cost >= settings.research_split_min_value:
            split_coros.append(_research_split(adapter, sem, asset, cost, vendor, ext_idx))

        # Group validation: only for assets the extraction flagged as children
        if asset.get("group_action") in ("child", "accessory"):
            parent_temp_id = asset.get("group_parent_temp_id")
            parent = next(
                (a for a in assets if a.get("temp_asset_id") == parent_temp_id),
                None,
            )
            if parent:
                group_coros.append(_validate_group(adapter, sem, asset, parent, ext_idx))

    all_coros = split_coros + group_coros
    if not all_coros:
        return [], []

    raw_results = await asyncio.gather(*all_coros, return_exceptions=True)

    splits: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    n_splits = len(split_coros)

    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            logger.warning("enrich_task_failed", task_index=i, error=str(result)[:200])
            continue
        if result is None:
            continue
        if i < n_splits:
            splits.append(result)
        else:
            groups.append(result)

    return splits, groups


# ── Split research ────────────────────────────────────────────────────


async def _research_split(
    adapter: VLMAdapter,
    sem: asyncio.Semaphore,
    asset: dict[str, Any],
    cost: float,
    vendor: str,
    ext_idx: int,
) -> dict[str, Any] | None:
    """Ask the VLM whether this high-value asset should be componentized per IND AS 16."""
    name = asset.get("asset_name") or asset.get("description") or "Unknown Asset"
    category = asset.get("suggested_category") or "Unknown"

    prompt = SPLIT_RESEARCH_PROMPT_TEMPLATE.format(
        asset_name=name,
        category=category,
        value=f"{cost:,.0f}",
        vendor=vendor,
    )
    try:
        async with sem:
            result = await adapter.extract(
                images=[],
                system_prompt=SPLIT_RESEARCH_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
        if not isinstance(result, dict):
            return None
        should_split = bool(result.get("should_split", False))
        components = result.get("components") or []
        if not should_split or not components:
            return None  # No suggestion if no split needed
        logger.info(
            "split_suggested",
            asset=name,
            cost=cost,
            components=len(components),
            ext_idx=ext_idx,
        )
        return {
            "extraction_index": ext_idx,
            "temp_asset_id": asset.get("temp_asset_id"),
            "asset_name": name,
            "total_value": cost,
            "should_split": True,
            "reason": result.get("reason", ""),
            "components": components,
        }
    except Exception as e:
        logger.warning("split_research_failed", asset=name, error=str(e)[:200])
        return None


# ── Group validation ──────────────────────────────────────────────────


async def _validate_group(
    adapter: VLMAdapter,
    sem: asyncio.Semaphore,
    child: dict[str, Any],
    parent: dict[str, Any],
    ext_idx: int,
) -> dict[str, Any] | None:
    """Confirm whether child asset belongs under parent (YES/NO with reason)."""
    child_name = child.get("asset_name") or child.get("description") or "Child"
    parent_name = parent.get("asset_name") or parent.get("description") or "Parent"
    child_cost = _asset_cost(child)
    parent_cost = _asset_cost(parent)

    prompt = GROUP_VALIDATION_PROMPT_TEMPLATE.format(
        child_name=child_name,
        child_value=f"{child_cost:,.0f}",
        child_category=child.get("suggested_category") or "Unknown",
        parent_name=parent_name,
        parent_value=f"{parent_cost:,.0f}",
        parent_category=parent.get("suggested_category") or "Unknown",
    )
    try:
        async with sem:
            raw = await adapter.simple_query(prompt=prompt)
        raw = raw.strip()
        confirmed = raw.upper().startswith("YES")
        reason = raw.split("—", 1)[-1].strip() if "—" in raw else raw
        logger.info(
            "group_validated",
            child=child_name,
            parent=parent_name,
            confirmed=confirmed,
            ext_idx=ext_idx,
        )
        return {
            "extraction_index": ext_idx,
            "child_temp_id": child.get("temp_asset_id"),
            "parent_temp_id": child.get("group_parent_temp_id"),
            "child_name": child_name,
            "parent_name": parent_name,
            "confirmed": confirmed,
            "reason": reason,
        }
    except Exception as e:
        logger.warning("group_validation_failed", child=child_name, error=str(e)[:200])
        return None


# ── Helpers ───────────────────────────────────────────────────────────


def _asset_cost(asset: dict[str, Any]) -> float:
    """Extract the per-unit total cost from an asset dict."""
    for key in ("individual_cost_with_tax", "individual_cost_before_tax"):
        v = asset.get(key)
        if v is None:
            continue
        if isinstance(v, dict):
            v = v.get("value")
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _get_vendor(extraction: dict[str, Any]) -> str:
    vd = extraction.get("vendor_details", {})
    if not isinstance(vd, dict):
        return "Unknown"
    v = vd.get("vendor_name")
    if isinstance(v, dict):
        return str(v.get("value") or "Unknown")
    return str(v or "Unknown")
