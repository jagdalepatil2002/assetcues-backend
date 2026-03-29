"""
Image processing utilities — format conversion, quality scoring,
enhancement, multi-resolution generation, and region cropping.

All functions are pure (bytes in, bytes out) with no side effects.
"""
from __future__ import annotations

import io
import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import structlog

logger = structlog.get_logger()


# ── Format conversion ──


def pdf_page_count(file_bytes: bytes) -> int:
    """Return number of pages in a PDF without full rasterization."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(file_bytes)
    try:
        return len(doc)
    finally:
        doc.close()


def pdf_to_images(file_bytes: bytes, dpi: int = 300) -> tuple[list[bytes], Optional[str]]:
    """
    Convert PDF pages to PNG images using pypdfium2.

    Returns:
        Tuple of (list of PNG bytes per page, native text if digital PDF).
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(file_bytes)
    images: list[bytes] = []
    native_text_parts: list[str] = []

    try:
        for i in range(len(doc)):
            page = doc[i]

            # Render to bitmap
            scale = dpi / 72
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            images.append(_pil_to_png_bytes(pil_image))

            # Extract native text if available
            text = page.get_textpage().get_text_range()
            if text and text.strip():
                native_text_parts.append(text.strip())
    finally:
        doc.close()

    native_text = "\n\n".join(native_text_parts) if native_text_parts else None
    return images, native_text


def detect_and_convert(
    file_bytes: bytes,
    file_type: str,
    *,
    pdf_dpi: Optional[int] = None,
) -> tuple[list[bytes], Optional[str]]:
    """
    Universal format handler. Converts any supported format to a list
    of PNG image bytes (one per page).

    Args:
        pdf_dpi: If set and file_type is pdf, render at this DPI (default 300).

    Returns:
        Tuple of (list of PNG bytes, native text or None).
    """
    if file_type == "pdf":
        dpi = pdf_dpi if pdf_dpi is not None else 300
        return pdf_to_images(file_bytes, dpi=dpi)

    if file_type == "heic":
        return [_heic_to_png(file_bytes)], None

    if file_type in ("tiff", "tif"):
        return _tiff_to_pages(file_bytes), None

    # JPEG, PNG, BMP, WEBP — single image
    if file_type in ("jpeg", "jpg", "png", "bmp", "webp"):
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
        return [_pil_to_png_bytes(img)], None

    raise ValueError(f"Unsupported file type: {file_type}")


def _heic_to_png(file_bytes: bytes) -> bytes:
    """Convert HEIC to PNG using pillow-heif."""
    from pillow_heif import register_heif_opener
    register_heif_opener()
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")
    return _pil_to_png_bytes(img)


def _tiff_to_pages(file_bytes: bytes) -> list[bytes]:
    """Extract all pages from a multi-page TIFF."""
    img = Image.open(io.BytesIO(file_bytes))
    pages: list[bytes] = []
    try:
        while True:
            frame = img.convert("RGB")
            pages.append(_pil_to_png_bytes(frame))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return pages


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Quality scoring ──


def score_quality(img_bytes: bytes) -> float:
    """
    Score image quality using Laplacian variance (blur detection).
    Returns 0.0 (very blurry) to 1.0 (sharp).
    """
    arr = _bytes_to_cv2(img_bytes)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(laplacian_var / 500.0, 1.0)


def detect_handwriting(img_bytes: bytes, quality_score: float, threshold: float = 0.6) -> bool:
    """
    Heuristic handwriting detection based on edge density.
    Handwritten docs have high edge density with lower overall quality.
    """
    if quality_score > threshold:
        return False

    arr = _bytes_to_cv2(img_bytes)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    # Handwritten docs typically have edge density > 0.08 with moderate blur
    return edge_density > 0.08


# ── Enhancement ──


def enhance_image(img_bytes: bytes) -> bytes:
    """
    Enhance a poor-quality image: deskew, CLAHE contrast, denoise.
    For printed documents.
    """
    arr = _bytes_to_cv2(img_bytes)

    arr = _deskew(arr)

    # CLAHE contrast enhancement
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a, b])
    arr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Denoise
    arr = cv2.fastNlMeansDenoisingColored(arr, None, 10, 10, 7, 21)

    return _cv2_to_png_bytes(arr)


def preprocess_for_handwriting(img_bytes: bytes) -> bytes:
    """
    Specialized enhancement for handwritten documents.
    Uses Sauvola-like adaptive binarization and morphological closing.
    No sharpening (creates artifacts on handwriting strokes).
    """
    arr = _bytes_to_cv2(img_bytes)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    # Sauvola-like adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10,
    )

    # Morphological closing to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Boost contrast (dark ink on light paper)
    result = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    return _cv2_to_png_bytes(result)


# ── Multi-resolution generation ──


def create_multi_resolution(img_bytes: bytes) -> dict[str, bytes | None]:
    """
    Create multiple resolution versions for cross-referencing by VLM.

    Returns:
        {"full": original PNG, "high": 2x upscale, "table_crop": cropped table or None}
    """
    arr = _bytes_to_cv2(img_bytes)
    h, w = arr.shape[:2]

    # Full resolution (as-is)
    full = img_bytes

    # 2x upscale for reading fine print and handwriting
    upscaled = cv2.resize(arr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    high = _cv2_to_png_bytes(upscaled)

    # Table crop via contour detection (best-effort)
    table_crop = _detect_and_crop_table(arr)

    return {"full": full, "high": high, "table_crop": table_crop}


def _detect_and_crop_table(arr: np.ndarray) -> bytes | None:
    """
    Detect the largest table-like rectangular region using line detection.
    Returns cropped PNG bytes, or None if no table found.
    """
    try:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 10, 40), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 10, 40)))

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        table_mask = cv2.add(h_lines, v_lines)

        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find the largest contour by area (likely the main table)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Table should be at least 5% of the image
        if area < (h * w * 0.05):
            return None

        x, y, cw, ch = cv2.boundingRect(largest)
        # Pad by 10px
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        cw = min(w - x, cw + 2 * pad)
        ch = min(h - y, ch + 2 * pad)

        crop = arr[y : y + ch, x : x + cw]
        return _cv2_to_png_bytes(crop)
    except Exception:
        logger.warning("table_detection_failed", exc_info=True)
        return None


# ── Region cropping ──


def auto_crop_regions(img_bytes: bytes) -> list[dict]:
    """
    Crop invoice into header, table, and footer regions using
    horizontal projection profile analysis.

    Returns list of {"region": str, "image": bytes, "bbox": dict, "page_index": 0}.
    Returns empty list if segmentation fails (pipeline falls back to full-image).
    """
    try:
        arr = _bytes_to_cv2(img_bytes)
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        # Horizontal projection profile
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        h_proj = np.sum(binary, axis=1) / 255

        # Find large gaps (low projection) to segment regions
        threshold = w * 0.02
        is_content = h_proj > threshold

        # Simple 3-way split: header (top 25%), table (middle 50%), footer (bottom 25%)
        header_end = int(h * 0.25)
        footer_start = int(h * 0.75)

        # Refine boundaries using projection gaps
        for row in range(int(h * 0.20), int(h * 0.35)):
            if not is_content[row]:
                header_end = row
                break

        for row in range(int(h * 0.80), int(h * 0.65), -1):
            if not is_content[row]:
                footer_start = row
                break

        regions = []
        pad = 5

        # Header
        if header_end > 20:
            crop = arr[0 : header_end + pad, :]
            regions.append({
                "region": "header",
                "image": _cv2_to_png_bytes(crop),
                "bbox": {"x": 0, "y": 0, "w": w, "h": header_end + pad},
                "page_index": 0,
            })

        # Table (middle)
        table_start = max(0, header_end - pad)
        table_end = min(h, footer_start + pad)
        if table_end - table_start > 50:
            crop = arr[table_start:table_end, :]
            regions.append({
                "region": "table",
                "image": _cv2_to_png_bytes(crop),
                "bbox": {"x": 0, "y": table_start, "w": w, "h": table_end - table_start},
                "page_index": 0,
            })

        # Footer
        if h - footer_start > 20:
            crop = arr[max(0, footer_start - pad) :, :]
            regions.append({
                "region": "footer",
                "image": _cv2_to_png_bytes(crop),
                "bbox": {"x": 0, "y": max(0, footer_start - pad), "w": w, "h": h - max(0, footer_start - pad)},
                "page_index": 0,
            })

        return regions

    except Exception:
        logger.warning("region_crop_failed", exc_info=True)
        return []


def crop_top_header_strip(img_bytes: bytes, height_fraction: float = 0.38) -> bytes | None:
    """Crop the top band of a page (vendor/GSTIN area) for retries when no bbox exists."""
    try:
        arr = _bytes_to_cv2(img_bytes)
        h, w = arr.shape[:2]
        ch = max(8, int(h * max(0.05, min(0.6, height_fraction))))
        crop = arr[0:ch, 0:w]
        return _cv2_to_png_bytes(crop)
    except Exception:
        logger.warning("header_strip_crop_failed", exc_info=True)
        return None


def crop_field_region(img_bytes: bytes, bbox: dict) -> bytes | None:
    """Crop a specific bounding box region for targeted re-reads."""
    try:
        arr = _bytes_to_cv2(img_bytes)
        h, w = arr.shape[:2]

        x = max(0, int(bbox.get("x", 0)))
        y = max(0, int(bbox.get("y", 0)))
        cw = min(w - x, int(bbox.get("w", w)))
        ch = min(h - y, int(bbox.get("h", h)))

        if cw < 5 or ch < 5:
            return None

        crop = arr[y : y + ch, x : x + cw]
        return _cv2_to_png_bytes(crop)
    except Exception:
        logger.warning("field_crop_failed", bbox=bbox, exc_info=True)
        return None


# ── Internal helpers ──


def _bytes_to_cv2(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _cv2_to_png_bytes(arr: np.ndarray) -> bytes:
    success, buf = cv2.imencode(".png", arr)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return buf.tobytes()


def _deskew(img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Deskew image using Hough line transform."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
