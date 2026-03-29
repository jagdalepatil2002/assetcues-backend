from __future__ import annotations

from typing import Any

import structlog

from far_ai_brain.config.settings import settings
from far_ai_brain.schemas.state import PipelineState
from far_ai_brain.utils.image import (
    detect_and_convert,
    detect_handwriting,
    enhance_image,
    pdf_page_count,
    preprocess_for_handwriting,
    score_quality,
)

logger = structlog.get_logger()


async def preprocess_node(state: PipelineState) -> dict[str, Any]:
    """Preprocess raw file bytes into normalised page images.

    Converts the incoming document to page images, scores quality,
    detects handwriting, and applies appropriate enhancement.
    """
    raw_bytes: bytes = state["raw_file_bytes"]
    file_type: str = state["file_type"]

    if file_type == "pdf":
        pc = pdf_page_count(raw_bytes)
        dpi = (
            settings.pdf_render_dpi_reduced
            if pc >= settings.pdf_large_page_threshold
            else settings.pdf_render_dpi_default
        )
        if pc >= settings.pdf_large_page_threshold:
            logger.info(
                "preprocess.pdf_reduced_dpi",
                page_count=pc,
                dpi=dpi,
                threshold=settings.pdf_large_page_threshold,
            )
        page_images, native_text = detect_and_convert(raw_bytes, file_type, pdf_dpi=dpi)
    else:
        page_images, native_text = detect_and_convert(raw_bytes, file_type)
    page_count = len(page_images)
    logger.info("preprocess.pages_detected", page_count=page_count)

    quality_scores: list[float] = []
    handwritten_flags: list[bool] = []
    processed_images: list[Any] = []

    for idx, img in enumerate(page_images):
        quality = score_quality(img)
        is_hw = detect_handwriting(img, quality)
        quality_scores.append(quality)
        handwritten_flags.append(is_hw)

        if is_hw:
            logger.info("preprocess.handwriting_detected", page=idx)
            processed = preprocess_for_handwriting(img)
        elif quality < settings.quality_threshold_enhance:
            logger.info(
                "preprocess.enhancing_low_quality",
                page=idx,
                quality=quality,
            )
            processed = enhance_image(img)
        else:
            processed = img

        processed_images.append(processed)

    # --- multi-resolution & region crops disabled (single VLM call strategy) ---
    multi_res_images: list[dict[str, Any]] = []
    region_crops: list[Any] = []

    is_handwritten = any(handwritten_flags)

    logger.info(
        "preprocess.complete",
        page_count=page_count,
        is_handwritten=is_handwritten,
        avg_quality=sum(quality_scores) / max(len(quality_scores), 1),
    )

    return {
        "page_images": processed_images,
        "page_count": page_count,
        "quality_scores": quality_scores,
        "is_handwritten": is_handwritten,
        "native_text": native_text,
        "multi_res_images": multi_res_images,
        "region_crops": region_crops,
    }
