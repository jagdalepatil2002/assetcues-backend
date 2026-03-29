"""
Unified VLM adapter — model-agnostic interface to any vision-language model.

Supports 3 providers:
  - google: Google Gemini via google-genai SDK (google.genai)
  - openai: OpenAI models via openai SDK
  - self_hosted: vLLM or any OpenAI-compatible endpoint

This file + config/models.py are the ONLY files that reference provider
or model names. The rest of the codebase calls ONLY this adapter.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Optional

import structlog
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from google import genai
from google.genai import types as genai_types

from far_ai_brain.config.models import model_config
from far_ai_brain.config.settings import settings

logger = structlog.get_logger()

_semaphores: dict[int, asyncio.Semaphore] = {}


def _get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if loop_id not in _semaphores:
        _semaphores[loop_id] = asyncio.Semaphore(settings.vlm_concurrency_limit)
    return _semaphores[loop_id]


class ExtractionError(Exception):
    """Raised when VLM returns invalid or unparseable output."""

    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response


class VLMAdapter:
    """
    Model-agnostic VLM interface.

    Usage:
        adapter = VLMAdapter(role="primary")
        result = await adapter.extract(images=[img_bytes], ...)
    """

    def __init__(self, role: str = "primary") -> None:
        self.role = role
        self.provider, self.model, self.base_url = model_config.get_provider_and_model(role)
        self.api_key = model_config.get_api_key(self.provider)

    async def extract(
        self,
        images: list[bytes],
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict] = None,
        thinking_level: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Send images + prompts to VLM, expect structured JSON back.

        Args:
            images: List of PNG image bytes.
            system_prompt: System-level instructions.
            user_prompt: User-level prompt.
            json_schema: Optional JSON schema for structured output.
            thinking_level: "low" or "high" (Gemini-specific, ignored by other providers).

        Returns:
            Parsed dict from VLM JSON response.

        Raises:
            ExtractionError: If VLM returns invalid JSON after retry.
        """
        return await self._extract_with_retry(images, system_prompt, user_prompt, json_schema, thinking_level)

    async def simple_query(self, prompt: str, image: Optional[bytes] = None) -> str:
        """Send a simple text prompt (optionally with one image), get text back."""
        images = [image] if image else []

        if self.provider == "google":
            return await self._google_simple_query(prompt, images)
        else:
            return await self._openai_simple_query(prompt, images)

    async def extract_batch(
        self,
        calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Run multiple extract() calls concurrently, respecting the semaphore.

        Args:
            calls: List of kwargs dicts for extract(). Each must have
                   'images', 'system_prompt', 'user_prompt', and optionally
                   'json_schema', 'thinking_level'.

        Returns:
            List of parsed dicts in the same order as calls.
        """
        tasks = [self.extract(**call_kwargs) for call_kwargs in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("extract_batch_item_failed", index=i, error=str(result))
                processed.append({
                    "extraction_meta": {
                        "overall_confidence": 0.0,
                        "error": "batch_item_failed",
                        "detail": str(result)[:500],
                    },
                })
            else:
                processed.append(result)
        return processed

    # ── Internal implementation ──

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_not_exception_type(ExtractionError),
        reraise=True,
    )
    async def _extract_with_retry(
        self,
        images: list[bytes],
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict],
        thinking_level: Optional[str],
    ) -> dict[str, Any]:
        start = time.monotonic()
        try:
            if self.provider == "google":
                raw = await self._google_extract(images, system_prompt, user_prompt, json_schema, thinking_level)
            elif self.provider in ("openai", "self_hosted"):
                raw = await self._openai_extract(images, system_prompt, user_prompt, json_schema)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            latency_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "vlm_call_complete",
                role=self.role,
                provider=self.provider,
                model=self.model,
                latency_ms=latency_ms,
                response_length=len(raw),
            )

            return self._parse_json_response(raw)

        except ExtractionError:
            raise
        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.error("vlm_call_failed", role=self.role, provider=self.provider, model=self.model,
                         latency_ms=latency_ms, error=str(e))
            raise

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """Parse JSON from VLM response, handling common issues."""
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ExtractionError(
                f"VLM returned invalid JSON: {e}",
                raw_response=raw[:2000],
            )

    # ── Google Gemini provider (google-genai SDK) ──

    @staticmethod
    def _gemini_thinking_config(thinking_level: Optional[str], model: str) -> Optional[genai_types.ThinkingConfig]:
        """Return the right ThinkingConfig for the given model family.

        Gemini 2.5+: thinking_level enum causes 400; use thinking_budget (token count) instead.
          high  → 8 192 tokens  (thorough reasoning for complex/multi-page invoices)
          low   → 1 024 tokens  (fast path for simple single-page invoices)
        Older models (pre-2.5): use ThinkingLevel enum directly.
        Gemini 2.0: no thinking support — return None.
        """
        if not thinking_level:
            return None
        tl = thinking_level.strip().lower()
        m = model.strip().lower()

        if "gemini-2.0" in m:
            return None  # 2.0 has no thinking support

        if m.startswith("gemini-2.5-"):
            # Use thinking_budget (int tokens) — the only way 2.5 accepts it
            budget = 8192 if tl == "high" else 1024
            return genai_types.ThinkingConfig(thinking_budget=budget)

        # Pre-2.5 models: use the ThinkingLevel enum
        if tl == "high":
            return genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.HIGH)
        if tl == "low":
            return genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.LOW)
        return None

    @staticmethod
    def _google_model_supports_thinking(model: str) -> bool:
        """Returns False only for models that reject thinking_config entirely."""
        return "gemini-2.0" not in model.strip().lower()

    @staticmethod
    def _google_error_is_transient(exc: BaseException) -> bool:
        """Google sometimes returns 5xx on otherwise valid requests; safe to retry."""
        s = str(exc).upper()
        if "500" in s and ("INTERNAL" in s or "UNKNOWN" in s):
            return True
        if "503" in s or "UNAVAILABLE" in s:
            return True
        if "429" in s or "RESOURCE_EXHAUSTED" in s:
            return True
        return False

    def _google_client(self) -> genai.Client:
        timeout_ms = max(1, int(settings.api_timeout_seconds * 1000))
        return genai.Client(
            api_key=self.api_key or None,
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
        )

    async def _google_extract(
        self,
        images: list[bytes],
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict],
        thinking_level: Optional[str],
    ) -> str:
        parts: list[genai_types.Part] = []
        for img in images:
            parts.append(genai_types.Part.from_bytes(data=img, mime_type="image/png"))
        parts.append(genai_types.Part.from_text(text=user_prompt))
        contents: genai_types.ContentListUnion = [
            genai_types.Content(role="user", parts=parts)
        ]

        # Multi-image + response_json_schema often yields 400 INVALID_ARGUMENT on Gemini.
        # Prompt-only JSON (no schema) is reliable for batched pages.
        effective_schema = json_schema if len(images) <= 1 else None

        # Gemini: `response_mime_type: application/json` requires `response_json_schema`.
        config_kwargs: dict[str, Any] = {"system_instruction": system_prompt}
        if effective_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = effective_schema
        if self._google_model_supports_thinking(self.model):
            tc = self._gemini_thinking_config(thinking_level, self.model)
            if tc is not None:
                config_kwargs["thinking_config"] = tc

        config = genai_types.GenerateContentConfig(**config_kwargs)
        delays = (2.0, 5.0, 10.0)
        for attempt in range(len(delays) + 1):
            client = self._google_client()
            try:
                sem = _get_semaphore()
                async with sem:
                    response = await client.aio.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                return response.text or ""
            except Exception as e:
                if attempt < len(delays) and self._google_error_is_transient(e):
                    logger.warning(
                        "vlm_google_transient_retry",
                        role=self.role,
                        model=self.model,
                        attempt=attempt + 1,
                        max_attempts=len(delays) + 1,
                        error=str(e)[:240],
                    )
                    await asyncio.sleep(delays[attempt])
                else:
                    raise
            finally:
                await client.aio.aclose()

    async def _google_simple_query(self, prompt: str, images: list[bytes]) -> str:
        parts: list[genai_types.Part] = []
        for img in images:
            parts.append(genai_types.Part.from_bytes(data=img, mime_type="image/png"))
        parts.append(genai_types.Part.from_text(text=prompt))

        config = genai_types.GenerateContentConfig(system_instruction="")
        client = self._google_client()
        try:
            sem = _get_semaphore()
            async with sem:
                response = await client.aio.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=config,
                )
            return response.text or ""
        finally:
            await client.aio.aclose()

    # ── OpenAI / self-hosted provider ──

    async def _openai_extract(
        self,
        images: list[bytes],
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict],
    ) -> str:
        from openai import AsyncOpenAI

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.provider == "self_hosted" and self.base_url:
            client_kwargs["base_url"] = self.base_url
        elif self.provider == "self_hosted":
            client_kwargs["api_key"] = "not-needed"

        client = AsyncOpenAI(**client_kwargs)

        # Build user content: images as data URIs (document-before-instructions)
        content: list[dict[str, Any]] = []
        for img in images:
            b64 = base64.b64encode(img).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        content.append({"type": "text", "text": user_prompt})

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": 16384,
            "timeout": settings.api_timeout_seconds,
        }

        if json_schema:
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": json_schema},
            }
        else:
            request_kwargs["response_format"] = {"type": "json_object"}

        sem = _get_semaphore()
        async with sem:
            response = await client.chat.completions.create(**request_kwargs)

        return response.choices[0].message.content or ""

    async def _openai_simple_query(self, prompt: str, images: list[bytes]) -> str:
        from openai import AsyncOpenAI

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.provider == "self_hosted" and self.base_url:
            client_kwargs["base_url"] = self.base_url
        elif self.provider == "self_hosted":
            client_kwargs["api_key"] = "not-needed"

        client = AsyncOpenAI(**client_kwargs)

        content: list[dict[str, Any]] = []
        for img in images:
            b64 = base64.b64encode(img).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        sem = _get_semaphore()
        async with sem:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                timeout=settings.api_timeout_seconds,
            )

        return response.choices[0].message.content or ""
