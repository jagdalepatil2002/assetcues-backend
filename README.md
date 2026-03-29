# Assetcues Invoice Agentic AI v1

Agentic AI invoice extraction microservice by Assetcues. Image in, structured JSON out.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env
# Set your GOOGLE_API_KEY in .env
uvicorn far_ai_brain.api.main:app --reload --port 8000
```

Logs go to the terminal (stderr). Set `LOG_LEVEL=DEBUG` in `.env` or the environment for more detail. You should see `pipeline_step_start` / `pipeline_step_done` for each stage (preprocess → classify → extract → …).

## API

```bash
# Extract from invoice (multipart upload)
curl -X POST http://localhost:8000/api/v1/extract/upload \
  -F "tenant_id=test" \
  -F "mode=creation" \
  -F "file=@invoice.pdf"

# Health check
curl http://localhost:8000/health
```

## Docker

```bash
docker build -t assetcues-invoice-ai .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key assetcues-invoice-ai
```

## POC Frontend (safe to delete — zero impact on backend)

A full-featured UI demo lives in `../poc-frontend/`. It is **completely separate** from this backend — no shared files, no shared dependencies, no code changes needed.

**Run both together:**

```bash
# Terminal 1: Backend
cd far-ai-brain
uvicorn far_ai_brain.api.main:app --reload --port 8000

# Terminal 2: Frontend
cd poc-frontend
.\serve.ps1
# Open http://127.0.0.1:5174
```

**To remove the POC frontend entirely:**

| What to remove | How |
|----------------|-----|
| POC Frontend | `rm -rf ../poc-frontend/` |
| POC CORS middleware | In `far_ai_brain/api/main.py`, delete the block between `# ── POC CORS` and `# ── END POC CORS ──` |

**Backend impact: NONE.** The extraction pipeline, prompts, schemas, and all business logic remain untouched. The POC frontend only calls `POST /api/v1/extract/upload` and `GET /health`.

