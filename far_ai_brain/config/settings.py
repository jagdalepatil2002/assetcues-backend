"""
Application settings — thresholds, limits, feature flags.
No database URLs. No Redis. No S3. Just the Assetcues Invoice Agentic AI config.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Loads application settings from environment variables."""

    # Confidence thresholds
    confidence_auto_accept: float = 0.90
    confidence_reread_threshold: float = 0.80
    confidence_verification_min: float = 0.50
    confidence_verification_max: float = 0.90

    # Split research
    research_split_min_value: float = 100000

    # Image quality thresholds
    quality_threshold_enhance: float = 0.4
    quality_threshold_handwriting: float = 0.6

    # API limits
    max_file_size_mb: int = 50
    api_timeout_seconds: int = 120

    # Large PDFs: cap pages, batch VLM calls, reduce render DPI when many pages
    max_pdf_pages: int = 100
    vlm_max_pages_per_request: int = 8
    pdf_render_dpi_default: int = 300
    pdf_render_dpi_reduced: int = 200
    pdf_large_page_threshold: int = 12

    # VLM concurrency
    vlm_concurrency_limit: int = 5

    # Retry loop cap
    max_retry_loops: int = 1

    # Training data collection (toggleable)
    save_training_data: bool = False
    training_data_dir: str = "./training_data"

    # CORS — comma-separated list of allowed origins.
    # Set to "*" only for local dev. In production list your exact Netlify / custom domain.
    # Example: ALLOWED_ORIGINS=https://your-app.netlify.app,https://assetcues.com
    allowed_origins: str = "*"

    # Supported input formats
    supported_formats: frozenset[str] = frozenset(
        {"pdf", "jpeg", "jpg", "png", "tiff", "tif", "heic", "bmp", "webp"}
    )

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
