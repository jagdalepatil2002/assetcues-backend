from __future__ import annotations

import pytest

from far_ai_brain.utils.image import (
    auto_crop_regions,
    create_multi_resolution,
    detect_and_convert,
    detect_handwriting,
    enhance_image,
    preprocess_for_handwriting,
    score_quality,
)


def test_score_quality_returns_0_to_1(sample_png_bytes: bytes):
    score = score_quality(sample_png_bytes)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_detect_handwriting_false_for_clean(sample_png_bytes: bytes):
    result = detect_handwriting(sample_png_bytes, 0.9)
    assert result is False


def test_enhance_image_returns_bytes(sample_png_bytes: bytes):
    enhanced = enhance_image(sample_png_bytes)
    assert isinstance(enhanced, bytes)
    assert len(enhanced) > 0


def test_preprocess_handwriting_returns_bytes(sample_png_bytes: bytes):
    result = preprocess_for_handwriting(sample_png_bytes)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_create_multi_resolution(sample_png_bytes: bytes):
    result = create_multi_resolution(sample_png_bytes)
    assert isinstance(result, dict)
    assert "full" in result
    assert "high" in result
    assert "table_crop" in result
    assert isinstance(result["full"], bytes)
    assert isinstance(result["high"], bytes)
    assert len(result["full"]) > 0
    assert len(result["high"]) > 0


def test_auto_crop_regions_returns_list(sample_png_bytes: bytes):
    regions = auto_crop_regions(sample_png_bytes)
    assert isinstance(regions, list)
    for region in regions:
        assert "region" in region
        assert "image" in region
        assert isinstance(region["image"], bytes)


def test_detect_and_convert_png(sample_png_bytes: bytes):
    images, native_text = detect_and_convert(sample_png_bytes, "png")
    assert isinstance(images, list)
    assert len(images) == 1
    assert isinstance(images[0], bytes)
    assert len(images[0]) > 0
    assert native_text is None
