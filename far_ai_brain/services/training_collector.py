"""
Persist extraction artifacts and metadata for future Qwen fine-tuning.

Gated by SAVE_TRAINING_DATA / settings.save_training_data. Failures are logged
and never propagated — the pipeline always continues.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any

import aiofiles
import structlog

from far_ai_brain.config.settings import settings
from far_ai_brain.schemas.state import PipelineState, RegionCrop

logger = structlog.get_logger()

_JSONL_NAME = "extractions.jsonl"


def _safe_file_stem(value: str) -> str:
    """Filesystem-safe stem derived from extraction_id or similar."""
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-") or "extraction"
    return cleaned[:200]


def _rel_image(filename: str) -> str:
    """Path relative to training_data_dir (POSIX for portability)."""
    return f"images/{filename}".replace("\\", "/")


def _extraction_meta(extraction: dict[str, Any]) -> dict[str, Any]:
    raw = extraction.get("extraction_meta")
    return raw if isinstance(raw, dict) else {}


def _resolve_extraction_id(extraction: dict[str, Any], upload_id: str, index: int) -> str:
    meta = _extraction_meta(extraction)
    eid = meta.get("extraction_id")
    if isinstance(eid, str) and eid.strip():
        return eid.strip()
    base = upload_id.strip() if upload_id.strip() else "unknown"
    return f"{base}_{index}"


def _document_type_str(extraction: dict[str, Any], state_fallback: str) -> str:
    dt = extraction.get("document_type")
    if isinstance(dt, dict):
        v = dt.get("value")
        if v is not None:
            return str(v)
    if isinstance(dt, str):
        return dt
    return state_fallback


def _confidence_float(extraction: dict[str, Any]) -> float:
    meta = _extraction_meta(extraction)
    oc = meta.get("overall_confidence")
    if isinstance(oc, (int, float)):
        return float(oc)
    return 0.0


def _model_used_str(extraction: dict[str, Any]) -> str:
    meta = _extraction_meta(extraction)
    mu = meta.get("model_used")
    return str(mu) if isinstance(mu, str) else ""


def _page_images_for_extraction(state: PipelineState, index: int, num_extractions: int) -> list[bytes]:
    page_groups = state.get("page_groups") or []
    if index < len(page_groups):
        group = page_groups[index]
        imgs = group.get("page_images")
        if isinstance(imgs, list) and imgs:
            return [b for b in imgs if isinstance(b, (bytes, bytearray))]

    all_pages = state.get("page_images") or []
    if num_extractions == 1 and isinstance(all_pages, list):
        return [b for b in all_pages if isinstance(b, (bytes, bytearray))]
    return []


def _allowed_page_indices_for_extraction(state: PipelineState, index: int) -> set[int] | None:
    """
    None = no filter (use all region crops). Empty set = no pages matched.
    """
    page_groups = state.get("page_groups") or []
    if index >= len(page_groups) or not page_groups:
        return None
    group = page_groups[index]
    indices = group.get("page_indices")
    if isinstance(indices, list) and indices:
        return {int(i) for i in indices if isinstance(i, int)}
    imgs = group.get("page_images")
    if isinstance(imgs, list) and imgs:
        return set(range(len(imgs)))
    return None


def _region_crops_for_extraction(
    state: PipelineState,
    index: int,
    num_extractions: int,
) -> list[RegionCrop]:
    raw = state.get("region_crops") or []
    if not isinstance(raw, list):
        return []

    crops: list[RegionCrop] = [c for c in raw if isinstance(c, dict)]

    if num_extractions <= 1:
        return crops

    allowed = _allowed_page_indices_for_extraction(state, index)
    if allowed is None:
        return crops

    out: list[RegionCrop] = []
    for c in crops:
        pidx = c.get("page_index", 0)
        pi = int(pidx) if isinstance(pidx, int) else 0
        if pi in allowed:
            out.append(c)
    return out


async def collect_training_data(state: PipelineState) -> None:
    if not settings.save_training_data:
        return

    try:
        base_dir = os.path.abspath(os.path.expanduser(settings.training_data_dir))
        images_dir = os.path.join(base_dir, "images")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        extractions = state.get("extractions")
        if not extractions:
            logger.debug("training_collect_skipped", reason="no_extractions")
            return

        upload_id = str(state.get("upload_id") or "")
        tenant_id = str(state.get("tenant_id") or "")
        file_name = str(state.get("file_name") or "")
        state_doc_type = str(state.get("document_type") or "")
        is_hw = bool(state.get("is_handwritten", False))
        quality_scores = state.get("quality_scores")
        qs_list: list[float] = (
            [float(x) for x in quality_scores if isinstance(x, (int, float))]
            if isinstance(quality_scores, list)
            else []
        )

        jsonl_path = os.path.join(base_dir, _JSONL_NAME)
        total_images_written = 0
        lines_appended = 0

        num_ext = len(extractions)

        for idx, raw_extraction in enumerate(extractions):
            try:
                if not isinstance(raw_extraction, dict):
                    logger.warning("training_collect_skip_non_dict", index=idx)
                    continue

                extraction: dict[str, Any] = raw_extraction
                extraction_id = _resolve_extraction_id(extraction, upload_id, idx)
                safe_stem = _safe_file_stem(extraction_id)
                if not safe_stem or safe_stem == "extraction":
                    safe_stem = _safe_file_stem(f"{upload_id or 'u'}_{idx}_{uuid.uuid4().hex[:8]}")

                pages = _page_images_for_extraction(state, idx, num_ext)
                image_paths: list[str] = []
                for i, img_bytes in enumerate(pages):
                    fname = f"{safe_stem}_page{i}.png"
                    fpath = os.path.join(images_dir, fname)
                    async with aiofiles.open(fpath, "wb") as out_f:
                        await out_f.write(bytes(img_bytes))
                    image_paths.append(_rel_image(fname))
                    total_images_written += 1

                region_crop_paths: dict[str, str] = {}
                crops = _region_crops_for_extraction(state, idx, num_ext)
                region_serial: defaultdict[str, int] = defaultdict(int)

                for crop in crops:
                    img = crop.get("image")
                    if not isinstance(img, (bytes, bytearray)):
                        continue
                    region = str(crop.get("region") or "region")
                    region_serial[region] += 1
                    n = region_serial[region]
                    fname = f"{safe_stem}_{region}.png" if n == 1 else f"{safe_stem}_{region}_{n}.png"
                    fpath = os.path.join(images_dir, fname)
                    async with aiofiles.open(fpath, "wb") as out_f:
                        await out_f.write(bytes(img))
                    rel = _rel_image(fname)
                    total_images_written += 1

                    key = region if n == 1 else f"{region}_{n}"
                    region_crop_paths[key] = rel

                record = {
                    "extraction_id": extraction_id,
                    "tenant_id": tenant_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "file_name": file_name,
                    "document_type": _document_type_str(extraction, state_doc_type),
                    "is_handwritten": is_hw,
                    "page_count": len(pages),
                    "image_paths": image_paths,
                    "region_crop_paths": region_crop_paths,
                    "model_used": _model_used_str(extraction),
                    "extraction_json": extraction,
                    "confidence": _confidence_float(extraction),
                    "quality_scores": qs_list,
                    "was_corrected": False,
                    "corrected_json": None,
                }

                line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
                async with aiofiles.open(jsonl_path, "a", encoding="utf-8") as jf:
                    await jf.write(line)
                lines_appended += 1

            except Exception:
                logger.exception(
                    "training_collect_extraction_failed",
                    extraction_index=idx,
                    upload_id=upload_id or None,
                )

        logger.info(
            "training_collect_done",
            images_saved=total_images_written,
            jsonl_lines_appended=lines_appended,
            training_data_dir=base_dir,
        )

    except Exception:
        logger.exception("training_collect_failed")


async def collect_training_node(state: PipelineState) -> dict[str, Any]:
    await collect_training_data(state)
    return {}
