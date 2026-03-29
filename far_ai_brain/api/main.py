"""
FastAPI application — the single entry point for the Assetcues Invoice Agentic AI.
Accepts invoice images (base64 JSON or multipart), returns structured JSON.
"""
from __future__ import annotations

import uuid
from typing import Any

from far_ai_brain.logging_setup import configure_terminal_logging

configure_terminal_logging()

import structlog
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from fastapi.responses import JSONResponse

from far_ai_brain.config.models import model_config
from far_ai_brain.config.settings import settings
from far_ai_brain.pipeline.graph import pipeline
from far_ai_brain.schemas.api import ExtractionResponse, HealthResponse, SingleExtractionResult

logger = structlog.get_logger()

app = FastAPI(
    title="Assetcues Invoice Agentic AI",
    version="0.1.0",
    description="Agentic AI invoice extraction microservice by Assetcues",
)

# ── POC CORS — remove this block when shipping production frontend ──
from starlette.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── END POC CORS ──



# ── Middleware: Request ID ──


@app.middleware("http")
async def add_request_id(request: Request, call_next: Any) -> Any:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Routes ──


@app.post("/api/v1/extract", response_model=ExtractionResponse)
async def extract_invoice() -> ExtractionResponse:
    """Deprecated JSON endpoint. Use multipart upload instead."""
    raise HTTPException(
        status_code=410,
        detail=(
            "POST /api/v1/extract (base64 JSON) is disabled. "
            "Use POST /api/v1/extract/upload with multipart file upload."
        ),
    )


@app.post("/api/v1/extract/upload", response_model=ExtractionResponse)
async def extract_invoice_multipart(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    mode: str = Form(default="creation"),
) -> ExtractionResponse:
    """Extract structured data from an invoice image (multipart file upload)."""
    file_bytes = await file.read()
    file_name = file.filename or "unknown.png"
    file_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    return await _run_extraction(
        tenant_id=tenant_id,
        file_bytes=file_bytes,
        file_ext=file_ext,
        file_name=file_name,
        mode=mode,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — returns model configuration."""
    return HealthResponse(
        status="healthy",
        models={
            "primary": f"{model_config.primary_vlm_provider}/{model_config.primary_vlm_model}",
            "verification": f"{model_config.verification_vlm_provider}/{model_config.verification_vlm_model}",
            "cheap": f"{model_config.cheap_vlm_provider}/{model_config.cheap_vlm_model}",
        },
    )


# ── Internal ──

# Map file extensions to normalized types
_EXT_MAP: dict[str, str] = {
    "pdf": "pdf", "jpeg": "jpeg", "jpg": "jpeg", "png": "png",
    "tiff": "tiff", "tif": "tiff", "heic": "heic", "bmp": "bmp", "webp": "webp",
}


def _sniff_file_type(file_bytes: bytes) -> str:
    """Infer pdf/png/jpeg/webp from magic bytes when filename has no extension."""
    if len(file_bytes) < 12:
        return ""
    if file_bytes[:4] == b"%PDF":
        return "pdf"
    if file_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if file_bytes[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WEBP":
        return "webp"
    if file_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if file_bytes[:2] == b"BM":
        return "bmp"
    return ""


async def _run_extraction(
    tenant_id: str,
    file_bytes: bytes,
    file_ext: str,
    file_name: str,
    mode: str = "creation",
) -> ExtractionResponse:
    """Shared extraction logic for both JSON and multipart endpoints."""

    file_type = _EXT_MAP.get(file_ext, "")
    if not file_type:
        file_type = _sniff_file_type(file_bytes)
    if not file_type:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format: .{file_ext or '(none)'} — "
                "for POST /api/v1/extract set file_name with a real extension "
                "(e.g. invoice.pdf). Or use POST /api/v1/extract/upload with a file "
                "that has a .pdf / .png / .jpg name."
            ),
        )

    # Size check FIRST to avoid parsing huge files
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB (max {settings.max_file_size_mb}MB)",
        )

    if file_type == "pdf":
        from far_ai_brain.utils.image import pdf_page_count

        page_count_pdf = pdf_page_count(file_bytes)
        if page_count_pdf > settings.max_pdf_pages:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"PDF has {page_count_pdf} pages; maximum supported is "
                    f"{settings.max_pdf_pages}. Increase MAX_PDF_PAGES in .env if needed."
                ),
            )

    upload_id = str(uuid.uuid4())
    logger.info("extraction_started", upload_id=upload_id, tenant_id=tenant_id,
                file_type=file_type, file_name=file_name, size_mb=round(size_mb, 2))

    initial_state = {
        "tenant_id": tenant_id,
        "upload_id": upload_id,
        "raw_file_bytes": file_bytes,
        "file_type": file_type,
        "file_name": file_name,
        "mode": mode,
        "retry_count": 0,
    }

    try:
        logger.info(
            "pipeline_invoke_start",
            upload_id=upload_id,
            tenant_id=tenant_id,
            file_name=file_name,
        )
        result = await pipeline.ainvoke(initial_state)
        logger.info("pipeline_invoke_done", upload_id=upload_id)
    except Exception as e:
        logger.error("pipeline_failed", upload_id=upload_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Processing failed. Check server logs for details.")

    error = result.get("error")
    if error:
        logger.error("pipeline_error", upload_id=upload_id, error=error)
        raise HTTPException(status_code=500, detail=error)

    # Build response
    extractions_data = result.get("extractions", [])
    validation_results = result.get("validation_results", [])
    fields_for_review = result.get("fields_for_review", [])
    split_suggestions = result.get("split_suggestions", [])
    group_suggestions = result.get("group_suggestions", [])
    final_confidence = result.get("final_confidence", 0.0)

    extraction_results: list[SingleExtractionResult] = []
    for i, ext in enumerate(extractions_data):
        meta = ext.get("extraction_meta", {})
        extraction_id = meta.get("extraction_id", f"{upload_id}_{i}")

        page_groups = result.get("page_groups", [])
        if i < len(page_groups):
            source_pages = page_groups[i]["page_indices"]
        else:
            logger.warning("source_pages_fallback", extraction_index=i, page_groups_count=len(page_groups))
            source_pages = [0]

        extraction_results.append(SingleExtractionResult(
            extraction_id=extraction_id,
            source_pages=source_pages,
            assets_to_create=ext.get("assets_to_create", []),
            split_suggestions=[s for s in split_suggestions if s.get("extraction_index", 0) == i],
            group_suggestions=[g for g in group_suggestions if g.get("extraction_index", 0) == i],
            fields_for_review=[f for f in fields_for_review if f.get("extraction_index", 0) == i],
            confidence=meta.get("overall_confidence", final_confidence),
            extraction_json=ext,
        ))

    logger.info("extraction_complete", upload_id=upload_id, invoice_count=len(extraction_results),
                confidence=final_confidence)

    warn_msg: str | None = None
    for er in extraction_results:
        ej = er.extraction_json or {}
        if not ej or (len(ej) <= 1 and not ej.get("invoice_header") and not ej.get("vendor_details")):
            warn_msg = (
                "One or more extractions are empty — the model call likely failed. "
                "Check Uvicorn logs for group_extraction_failed, vlm_call_failed, or full_extract_all_attempts_failed. "
                "Retry or switch PRIMARY_VLM_MODEL in .env."
            )
            break

    return ExtractionResponse(
        status="success",
        invoice_count=len(extraction_results),
        extractions=extraction_results,
        total_confidence=final_confidence,
        message=warn_msg,
    )


# ── AssetCues AI Agent ──

@app.post("/api/v1/agent/chat")
async def agent_chat(request: Request) -> JSONResponse:
    """AssetCues AI Agent — answer questions about assets using Gemini."""
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        context = body.get("context", "")

        if not question:
            return JSONResponse(status_code=400, content={"error": "Question is required"})

        from far_ai_brain.ai.vlm_adapter import VLMAdapter
        adapter = VLMAdapter(role="cheap")

        system_prompt = (
            "You are AssetCues AI Agent — an intelligent assistant for enterprise asset management. "
            "You help users understand their asset data, answer questions about invoices, vendors, "
            "depreciation, compliance, and asset tracking. "
            "Be concise, precise, and use Indian currency format (₹) when showing amounts. "
            "If the data doesn't contain enough info to answer, say so honestly. "
            "Format responses in clean markdown with bullet points where helpful."
        )

        user_prompt = f"""Based on the following asset management data, answer the user's question.

DATA CONTEXT:
{context}

USER QUESTION: {question}

Provide a helpful, concise answer:"""

        response = await adapter.simple_query(
            prompt=f"{system_prompt}\n\n{user_prompt}"
        )

        logger.info("agent_chat_complete", question=question[:100], response_length=len(response))
        return JSONResponse(content={"answer": response.strip()})

    except Exception as e:
        logger.error("agent_chat_failed", error=str(e))
        return JSONResponse(status_code=500, content={"error": "AI Agent is temporarily unavailable"})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"},
    )
