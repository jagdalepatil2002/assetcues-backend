"""
Enrich node — kept for backwards compatibility; enrichment is now disabled.
"""
from __future__ import annotations

import structlog

from far_ai_brain.schemas.state import PipelineState

logger = structlog.get_logger()


async def enrich_node(state: PipelineState) -> dict:
    """Enrichment disabled — split/group suggestions handled post-extraction."""
    return {
        "split_suggestions": [],
        "group_suggestions": [],
    }
