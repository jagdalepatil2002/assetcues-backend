FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install deps first (cached layer), then copy source so far_ai_brain is properly installed
COPY far_ai_brain/ far_ai_brain/
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "far_ai_brain.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
