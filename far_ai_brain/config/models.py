"""
MODEL CONFIGURATION — SINGLE SOURCE OF TRUTH

All Gemini for v1. To swap ANY model to OpenAI / self-hosted:
change .env values. Zero code changes required.

This file + ai/vlm_adapter.py are the ONLY files that reference
provider or model names. Grep the codebase to verify.
"""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    """Loads model configuration from environment variables."""

    primary_vlm_provider: str = "google"
    primary_vlm_model: str = "gemini-2.0-flash"
    primary_vlm_base_url: Optional[str] = None

    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    verification_vlm_provider: str = "google"
    verification_vlm_model: str = "gemini-2.5-flash-lite"
    verification_vlm_base_url: Optional[str] = None

    cheap_vlm_provider: str = "google"
    cheap_vlm_model: str = "gemini-2.5-flash-lite"
    cheap_vlm_base_url: Optional[str] = None

    primary_thinking_level_simple: str = "low"
    primary_thinking_level_complex: str = "high"

    def get_provider_and_model(self, role: str) -> tuple[str, str, Optional[str]]:
        """Return (provider, model, base_url) for a given role."""
        if role == "primary":
            return self.primary_vlm_provider, self.primary_vlm_model, self.primary_vlm_base_url
        if role == "verification":
            return self.verification_vlm_provider, self.verification_vlm_model, self.verification_vlm_base_url
        if role == "cheap":
            return self.cheap_vlm_provider, self.cheap_vlm_model, self.cheap_vlm_base_url
        raise ValueError(f"Unknown role: {role}. Must be 'primary', 'verification', or 'cheap'.")

    def get_api_key(self, provider: str) -> str:
        """Return the API key for a given provider."""
        if provider == "google":
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")
            return self.google_api_key
        if provider == "self_hosted":
            return self.openai_api_key or "not-needed"
        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
            return self.openai_api_key
        raise ValueError(f"Unknown provider: {provider}")

    model_config = {"env_file": ".env", "extra": "ignore"}


model_config = ModelConfig()
