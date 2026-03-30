from __future__ import annotations

from typing import Any

import structlog

from far_ai_brain.ai.vlm_adapter import VLMAdapter
from far_ai_brain.prompts.classification import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT,
    MULTIPAGE_CLASSIFICATION_PROMPT,
)
from far_ai_brain.schemas.state import PageGroup, PipelineState

logger = structlog.get_logger()

_DEFAULT_DOCUMENT_TYPE = "printed_invoice"
_DEFAULT_THINKING_LEVEL = "high"


async def classify_node(state: PipelineState) -> dict[str, Any]:
    """Classify the document type and detect multi-page patterns.

    Uses a cheap VLM call on the first page to determine the document
    type, complexity, and an extraction plan.  For multi-page documents
    a second call detects whether the pages form a single invoice or
    multiple invoices and groups them accordingly.
    """
    page_images: list[Any] = state["page_images"]
    page_count: int = state["page_count"]
    quality_scores: list[float] = state["quality_scores"]
    is_handwritten: bool = state["is_handwritten"]

    # For single-page docs we avoid an extra LLM classify call:
    # extraction node can handle full-page extraction directly.
    if page_count == 1:
        if is_handwritten or min(quality_scores) < 0.5:
            thinking_level = "high"
        else:
            thinking_level = "low"
        page_groups = [
            PageGroup(
                group_index=0,
                page_indices=[0],
                page_images=[page_images[0]],
            )
        ]
        logger.info(
            "classify.single_page_fastpath",
            document_type=_DEFAULT_DOCUMENT_TYPE,
            page_pattern="single_invoice",
            groups=1,
            thinking_level=thinking_level,
        )
        return {
            "document_type": _DEFAULT_DOCUMENT_TYPE,
            "page_pattern": "single_invoice",
            "page_groups": page_groups,
            "extraction_plan": {},
            "thinking_level": thinking_level,
        }

    adapter = VLMAdapter(role="cheap")

    # ------------------------------------------------------------------
    # Step 1 – classify the first page
    # ------------------------------------------------------------------
    try:
        classification_response = await adapter.extract(
            images=[page_images[0]],
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=CLASSIFICATION_USER_PROMPT,
        )

        document_type: str = classification_response.get(
            "document_type", _DEFAULT_DOCUMENT_TYPE
        )
        complexity: str = classification_response.get("complexity", "simple")
        extraction_plan: dict[str, Any] = classification_response.get(
            "extraction_plan", {}
        )
        try:
            estimated_line_items: int = int(
                classification_response.get("estimated_line_items") or 0
            )
        except (TypeError, ValueError):
            estimated_line_items = 0

        logger.info(
            "classify.first_page",
            document_type=document_type,
            complexity=complexity,
            estimated_line_items=estimated_line_items,
        )
    except Exception:
        logger.warning("classify.classification_failed", fallback=True)
        document_type = _DEFAULT_DOCUMENT_TYPE
        complexity = "complex"
        extraction_plan = {}
        estimated_line_items = 0

    # ------------------------------------------------------------------
    # Step 2 – determine thinking level
    # high   → complex/handwritten/low-quality docs
    # medium → simple docs with multiple line items or multi-page (likely
    #          to have multi-unit quantities with slash-separated serials)
    # low    → single-page, simple, few line items
    # ------------------------------------------------------------------
    needs_high_thinking = (
        complexity == "complex"
        or is_handwritten
        or min(quality_scores) < 0.5
    )
    needs_medium_thinking = (
        not needs_high_thinking
        and (page_count > 1 or estimated_line_items > 3)
    )
    if needs_high_thinking:
        thinking_level: str = "high"
    elif needs_medium_thinking:
        thinking_level = "medium"
    else:
        thinking_level = "low"

    logger.info("classify.thinking_level", thinking_level=thinking_level)

    # ------------------------------------------------------------------
    # Step 3 – multi-page pattern detection
    # ------------------------------------------------------------------
    page_pattern: str
    page_groups: list[PageGroup]

    if page_count > 1:
        try:
            thumbnails = [img for img in page_images]
            multipage_response = await adapter.extract(
                images=thumbnails,
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                user_prompt=MULTIPAGE_CLASSIFICATION_PROMPT,
            )

            page_pattern = multipage_response.get("pattern", "single_invoice")
            raw_groups: list[dict[str, Any]] = multipage_response.get(
                "page_groups", []
            )

            logger.info(
                "classify.multipage",
                pattern=page_pattern,
                group_count=len(raw_groups),
            )

            if page_pattern == "multiple_invoices" and raw_groups:
                page_groups = []
                for i, g in enumerate(raw_groups):
                    indices = g.get("page_indices", [])
                    # Validate indices are within bounds
                    valid_indices = [
                        p for p in indices
                        if isinstance(p, int) and 0 <= p < page_count
                    ]
                    if not valid_indices:
                        logger.warning(
                            "classify.invalid_page_indices",
                            group_index=i,
                            raw_indices=indices,
                        )
                        continue
                    page_groups.append(
                        PageGroup(
                            group_index=g.get("group_index", i),
                            page_indices=valid_indices,
                            page_images=[
                                page_images[p] for p in valid_indices
                            ],
                        )
                    )
                if not page_groups:
                    # Fallback: treat all pages as single invoice
                    page_pattern = "single_invoice"
                    page_groups = [
                        PageGroup(
                            group_index=0,
                            page_indices=list(range(page_count)),
                            page_images=list(page_images),
                        )
                    ]
            else:
                page_pattern = "single_invoice"
                page_groups = [
                    PageGroup(
                        group_index=0,
                        page_indices=list(range(page_count)),
                        page_images=list(page_images),
                    )
                ]
        except Exception:
            logger.warning("classify.multipage_failed", fallback=True)
            page_pattern = "single_invoice"
            page_groups = [
                PageGroup(
                    group_index=0,
                    page_indices=list(range(page_count)),
                    page_images=list(page_images),
                )
            ]
    else:
        page_pattern = "single_invoice"
        page_groups = [
            PageGroup(
                group_index=0,
                page_indices=[0],
                page_images=[page_images[0]],
            )
        ]

    logger.info(
        "classify.complete",
        document_type=document_type,
        page_pattern=page_pattern,
        groups=len(page_groups),
        thinking_level=thinking_level,
    )

    return {
        "document_type": document_type,
        "page_pattern": page_pattern,
        "page_groups": page_groups,
        "extraction_plan": extraction_plan,
        "thinking_level": thinking_level,
    }
