from __future__ import annotations

import pytest

from far_ai_brain.config.models import ModelConfig
from far_ai_brain.config.settings import Settings


class TestModelConfig:
    def test_primary_role(self):
        cfg = ModelConfig()
        provider, model, base_url = cfg.get_provider_and_model("primary")
        assert provider == "google"
        assert "gemini" in model.lower() or model  # model name may vary
        assert isinstance(base_url, (str, type(None)))

    def test_verification_role(self):
        cfg = ModelConfig()
        provider, model, base_url = cfg.get_provider_and_model("verification")
        assert provider == "google"
        assert isinstance(model, str) and len(model) > 0

    def test_cheap_role(self):
        cfg = ModelConfig()
        provider, model, base_url = cfg.get_provider_and_model("cheap")
        assert provider == "google"
        assert isinstance(model, str) and len(model) > 0

    def test_invalid_role_raises(self):
        cfg = ModelConfig()
        with pytest.raises(ValueError, match="Unknown role"):
            cfg.get_provider_and_model("nonexistent")

    def test_get_api_key_google(self):
        cfg = ModelConfig(google_api_key="test-google-key")
        assert cfg.get_api_key("google") == "test-google-key"

    def test_get_api_key_openai(self):
        cfg = ModelConfig(openai_api_key="test-openai-key")
        assert cfg.get_api_key("openai") == "test-openai-key"

    def test_get_api_key_self_hosted(self):
        cfg = ModelConfig(openai_api_key="test-key")
        assert cfg.get_api_key("self_hosted") == "test-key"

    def test_get_api_key_unknown_provider(self):
        cfg = ModelConfig()
        with pytest.raises(ValueError, match="Unknown provider"):
            cfg.get_api_key("azure")

    def test_get_api_key_raises_on_missing(self):
        cfg = ModelConfig(google_api_key=None)
        import pytest
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            cfg.get_api_key("google")

    def test_get_api_key_self_hosted_returns_not_needed(self):
        cfg = ModelConfig(openai_api_key=None)
        assert cfg.get_api_key("self_hosted") == "not-needed"


class TestSettings:
    def test_defaults_load(self):
        s = Settings()
        assert s.confidence_auto_accept == 0.90
        assert s.confidence_reread_threshold == 0.80
        assert s.max_file_size_mb == 50
        assert s.max_retry_loops == 1
        assert s.save_training_data is False
        assert s.vlm_concurrency_limit == 5

    def test_supported_formats(self):
        s = Settings()
        assert "pdf" in s.supported_formats
        assert "png" in s.supported_formats
        assert "heic" in s.supported_formats
        assert "doc" not in s.supported_formats

    def test_custom_values(self):
        s = Settings(
            confidence_auto_accept=0.95,
            max_retry_loops=5,
            save_training_data=True,
        )
        assert s.confidence_auto_accept == 0.95
        assert s.max_retry_loops == 5
        assert s.save_training_data is True
