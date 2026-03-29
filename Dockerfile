FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY far_ai_brain/ far_ai_brain/

EXPOSE 8000
CMD ["uvicorn", "far_ai_brain.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
