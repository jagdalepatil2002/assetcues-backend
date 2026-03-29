"""
LangGraph pipeline wiring — simplified linear graph.

5 nodes: preprocess → classify → extract → verify → collect_training → END
"""
from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from far_ai_brain.nodes.classify import classify_node
from far_ai_brain.nodes.extract import extract_node
from far_ai_brain.nodes.preprocess import preprocess_node
from far_ai_brain.nodes.verify import verify_node
from far_ai_brain.schemas.state import PipelineState
from far_ai_brain.services.training_collector import collect_training_node

_pipeline_log = structlog.get_logger("far_ai_brain.pipeline")


def _log_node(step: str, fn: Callable[[PipelineState], Awaitable[dict[str, Any]]]):
    """Wrap a pipeline node so start/finish and duration always print to the terminal."""

    @functools.wraps(fn)
    async def _wrapped(state: PipelineState) -> dict[str, Any]:
        uid = state.get("upload_id", "")
        _pipeline_log.info(
            "pipeline_step_start",
            step=step,
            upload_id=uid,
            file_name=state.get("file_name", ""),
        )
        t0 = time.perf_counter()
        try:
            out = await fn(state)
            ms = int((time.perf_counter() - t0) * 1000)
            _pipeline_log.info(
                "pipeline_step_done",
                step=step,
                upload_id=uid,
                elapsed_ms=ms,
            )
            return out
        except Exception:
            ms = int((time.perf_counter() - t0) * 1000)
            _pipeline_log.error(
                "pipeline_step_failed",
                step=step,
                upload_id=uid,
                elapsed_ms=ms,
                exc_info=True,
            )
            raise

    return _wrapped


def build_pipeline() -> StateGraph:
    """Build and compile the extraction pipeline graph."""
    graph = StateGraph(PipelineState)

    graph.add_node("preprocess", _log_node("preprocess", preprocess_node))
    graph.add_node("classify", _log_node("classify", classify_node))
    graph.add_node("extract", _log_node("extract", extract_node))
    graph.add_node("verify", _log_node("verify", verify_node))
    graph.add_node("collect_training", _log_node("collect_training", collect_training_node))

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "classify")
    graph.add_edge("classify", "extract")
    graph.add_edge("extract", "verify")
    graph.add_edge("verify", "collect_training")
    graph.add_edge("collect_training", END)

    return graph.compile()


pipeline = build_pipeline()
