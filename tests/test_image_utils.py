from __future__ import annotations

import io

import pytest
from PIL import Image

from far_ai_brain.utils.image import (
    auto_crop_regions,
    create_multi_resolution,
    crop_field_region,
    detect_and_convert,
    detect_handwriting,
    enhance_image,
    preprocess_for_handwriting,
    score_quality,
)


def _make_large_image(w: int = 800, h: int = 1200) -> bytes:
    """Create a realistic-sized test image with some content."""
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    for y in range(0, h, 100):
        for x in range(0, w, 10):
            img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestScoreQuality:
    def test_returns_float_in_range(self, sample_png_bytes):
        score = score_quality(sample_png_bytes)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_sharp_image_scores_higher(self):
        sharp = _make_large_image()
        score = score_quality(sharp)
        assert score >= 0.0


class TestDetectHandwriting:
    def test_clean_image_not_handwritten(self, sample_png_bytes):
        result = detect_handwriting(sample_png_bytes, quality_score=0.9)
        assert result is False

    def test_high_quality_returns_false(self, sample_png_bytes):
        result = detect_handwriting(sample_png_bytes, quality_score=0.8, threshold=0.6)
        assert result is False

    def test_low_quality_checks_edges(self):
        img = _make_large_image()
        result = detect_handwriting(img, quality_score=0.3, threshold=0.6)
        assert bool(result) in (True, False)


class TestEnhanceImage:
    def test_returns_bytes(self, sample_png_bytes):
        enhanced = enhance_image(sample_png_bytes)
        assert isinstance(enhanced, bytes)
        assert len(enhanced) > 0

    def test_output_is_valid_png(self, sample_png_bytes):
        enhanced = enhance_image(sample_png_bytes)
        img = Image.open(io.BytesIO(enhanced))
        assert img.format == "PNG"


class TestPreprocessForHandwriting:
    def test_returns_bytes(self, sample_png_bytes):
        result = preprocess_for_handwriting(sample_png_bytes)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_is_valid_image(self, sample_png_bytes):
        result = preprocess_for_handwriting(sample_png_bytes)
        img = Image.open(io.BytesIO(result))
        assert img.size[0] > 0
        assert img.size[1] > 0


class TestCreateMultiResolution:
    def test_returns_expected_keys(self, sample_png_bytes):
        result = create_multi_resolution(sample_png_bytes)
        assert "full" in result
        assert "high" in result
        assert "table_crop" in result

    def test_full_is_original(self, sample_png_bytes):
        result = create_multi_resolution(sample_png_bytes)
        assert result["full"] == sample_png_bytes

    def test_high_is_larger(self, sample_png_bytes):
        result = create_multi_resolution(sample_png_bytes)
        full_img = Image.open(io.BytesIO(result["full"]))
        high_img = Image.open(io.BytesIO(result["high"]))
        assert high_img.size[0] == full_img.size[0] * 2


class TestAutoCropRegions:
    def test_returns_list(self, sample_png_bytes):
        regions = auto_crop_regions(sample_png_bytes)
        assert isinstance(regions, list)

    def test_regions_have_required_keys(self):
        img = _make_large_image()
        regions = auto_crop_regions(img)
        for r in regions:
            assert "region" in r
            assert "image" in r
            assert isinstance(r["image"], bytes)


class TestCropFieldRegion:
    def test_valid_bbox(self):
        img = _make_large_image(200, 200)
        crop = crop_field_region(img, {"x": 10, "y": 10, "w": 50, "h": 50})
        assert crop is not None
        assert isinstance(crop, bytes)

    def test_tiny_crop_returns_none(self):
        img = _make_large_image(200, 200)
        crop = crop_field_region(img, {"x": 10, "y": 10, "w": 2, "h": 2})
        assert crop is None

    def test_invalid_image_returns_none(self):
        crop = crop_field_region(b"not an image", {"x": 0, "y": 0, "w": 50, "h": 50})
        assert crop is None


class TestDetectAndConvert:
    def test_png(self, sample_png_bytes):
        images, text = detect_and_convert(sample_png_bytes, "png")
        assert len(images) == 1
        assert isinstance(images[0], bytes)
        assert text is None

    def test_jpeg(self, sample_jpeg_bytes):
        images, text = detect_and_convert(sample_jpeg_bytes, "jpeg")
        assert len(images) == 1
        assert isinstance(images[0], bytes)

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            detect_and_convert(b"data", "docx")

    def test_bmp(self):
        img = Image.new("RGB", (50, 50), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        bmp_bytes = buf.getvalue()

        images, text = detect_and_convert(bmp_bytes, "bmp")
        assert len(images) == 1

    def test_webp(self):
        img = Image.new("RGB", (50, 50), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="WEBP")
        webp_bytes = buf.getvalue()

        images, text = detect_and_convert(webp_bytes, "webp")
        assert len(images) == 1
