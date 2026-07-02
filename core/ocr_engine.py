"""
Local OCR Engine
=================
Dual-engine OCR optimized for engineering drawings.

Primary: Tesseract OCR — fast, great for clean printed text and dimension callouts.
Secondary: EasyOCR — deep-learning based, better on rotated/curved text and mixed fonts.

Both run entirely locally. Tesseract requires a system install; EasyOCR downloads
its model on first run (~100MB, one-time) and then works offline.

The merger logic takes the union of detections, deduplicates overlapping boxes,
and keeps the highest-confidence result for each text region.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Lazy imports — we check availability at call time so the app can still
# load if one engine isn't installed.
_tesseract_available = None
_easyocr_available = None
_easyocr_reader = None


def _check_tesseract() -> bool:
    global _tesseract_available
    if _tesseract_available is None:
        try:
            import pytesseract
            # Quick sanity check — will throw if tesseract binary isn't found
            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except Exception:
            _tesseract_available = False
    return _tesseract_available


def _check_easyocr() -> bool:
    global _easyocr_available
    if _easyocr_available is None:
        try:
            import easyocr  # noqa: F401
            _easyocr_available = True
        except ImportError:
            _easyocr_available = False
    return _easyocr_available


def _get_easyocr_reader(languages: List[str] = None):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        langs = languages or ["en"]
        _easyocr_reader = easyocr.Reader(langs, gpu=False)
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """A single detected text region."""
    text: str
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in image coordinates
    confidence: float                  # 0.0 – 1.0
    source: str                        # "tesseract" | "easyocr" | "merged"
    angle: float = 0.0                # rotation angle in degrees (0 = horizontal)

    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


@dataclass
class OCRResult:
    """Full OCR output for an image."""
    text_blocks: List[TextBlock] = field(default_factory=list)
    full_text: str = ""                # all text concatenated, reading-order
    engine_used: str = ""              # "tesseract", "easyocr", "hybrid"
    image_shape: Tuple[int, int] = (0, 0)  # (height, width) of source

    def get_dimensions(self) -> List[TextBlock]:
        """Return text blocks that look like numeric dimensions."""
        dim_pattern = re.compile(
            r'[\d]+[.,]?\d*\s*(?:mm|cm|m|in|"|°|deg|±|Ø|R\d|x\d)', re.IGNORECASE
        )
        return [b for b in self.text_blocks if dim_pattern.search(b.text)]

    def get_text_in_region(self, roi: Tuple[int, int, int, int],
                           margin: int = 10) -> List[TextBlock]:
        """Return text blocks whose centers fall inside the given ROI (x,y,w,h)."""
        rx, ry, rw, rh = roi
        results = []
        for b in self.text_blocks:
            cx, cy = b.center()
            if (rx - margin <= cx <= rx + rw + margin and
                    ry - margin <= cy <= ry + rh + margin):
                results.append(b)
        return results


# ---------------------------------------------------------------------------
# Image pre-processing for OCR
# ---------------------------------------------------------------------------

def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Optimize an engineering drawing image for OCR accuracy.
    Engineering drawings are typically black lines/text on white background,
    so we enhance contrast and clean up noise without destroying thin strokes.
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Adaptive histogram equalization for mixed-contrast areas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Adaptive threshold — works well for line drawings with varying background
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=10
    )

    # Light denoise — remove scanner speckle without blurring text
    binary = cv2.medianBlur(binary, 3)

    return binary


def preprocess_for_ocr_light(img: np.ndarray) -> np.ndarray:
    """
    Lighter preprocessing that preserves more detail — use when the image
    is already clean (e.g., direct CAD export vs. a scan).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Simple Otsu threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Engine: Tesseract
# ---------------------------------------------------------------------------

def _run_tesseract(img: np.ndarray,
                   config: str = "--oem 3 --psm 6") -> List[TextBlock]:
    """
    Run Tesseract OCR and return structured results.
    PSM 6 = assume a single uniform block of text (good for engineering drawings
    where text appears in clusters: notes, title blocks, dimension callouts).
    """
    import pytesseract

    preprocessed = preprocess_for_ocr(img)

    # Get detailed data with bounding boxes and confidence
    data = pytesseract.image_to_data(preprocessed, config=config,
                                      output_type=pytesseract.Output.DICT)

    blocks = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < 10:  # skip empty / very low confidence
            continue

        blocks.append(TextBlock(
            text=text,
            bbox=(data["left"][i], data["top"][i],
                  data["width"][i], data["height"][i]),
            confidence=conf / 100.0,
            source="tesseract",
        ))

    return blocks


def _run_tesseract_multiconfig(img: np.ndarray) -> List[TextBlock]:
    """
    Run Tesseract with multiple PSM modes and merge results.
    Different PSMs catch different text layouts (blocks, single lines, scattered words).
    """
    configs = [
        "--oem 3 --psm 6",   # uniform block
        "--oem 3 --psm 11",  # sparse text (scattered annotations)
        "--oem 3 --psm 3",   # fully automatic page segmentation
    ]
    all_blocks = []
    for cfg in configs:
        try:
            blocks = _run_tesseract(img, config=cfg)
            all_blocks.extend(blocks)
        except Exception:
            continue
    return all_blocks


# ---------------------------------------------------------------------------
# Engine: EasyOCR
# ---------------------------------------------------------------------------

def _run_easyocr(img: np.ndarray,
                 languages: List[str] = None) -> List[TextBlock]:
    """
    Run EasyOCR and return structured results.
    EasyOCR handles rotated text and mixed fonts better than Tesseract.
    """
    reader = _get_easyocr_reader(languages)

    # EasyOCR works on the original image (it does its own preprocessing)
    if img.ndim == 2:
        input_img = img
    else:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = reader.readtext(input_img, detail=1, paragraph=False)

    blocks = []
    for (corners, text, conf) in results:
        text = text.strip()
        if not text:
            continue

        # corners is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] — convert to (x,y,w,h)
        xs = [int(c[0]) for c in corners]
        ys = [int(c[1]) for c in corners]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y

        # Estimate rotation angle from the first edge
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        angle = np.degrees(np.arctan2(dy, dx))

        blocks.append(TextBlock(
            text=text,
            bbox=(x, y, max(w, 1), max(h, 1)),
            confidence=float(conf),
            source="easyocr",
            angle=angle,
        ))

    return blocks


# ---------------------------------------------------------------------------
# Merge / deduplicate
# ---------------------------------------------------------------------------

def _iou(box1: Tuple[int, int, int, int],
         box2: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union for two (x,y,w,h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi = max(x1, x2)
    yi = max(y1, y2)
    xf = min(x1 + w1, x2 + w2)
    yf = min(y1 + h1, y2 + h2)

    if xf <= xi or yf <= yi:
        return 0.0

    inter = (xf - xi) * (yf - yi)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1)


def _merge_blocks(blocks: List[TextBlock],
                  iou_threshold: float = 0.3) -> List[TextBlock]:
    """
    Deduplicate overlapping text blocks from multiple engines.
    When two detections overlap significantly, keep the one with higher confidence.
    """
    if not blocks:
        return []

    # Sort by confidence descending
    sorted_blocks = sorted(blocks, key=lambda b: b.confidence, reverse=True)
    kept = []

    for block in sorted_blocks:
        is_duplicate = False
        for existing in kept:
            if _iou(block.bbox, existing.bbox) > iou_threshold:
                # Overlapping — skip the lower-confidence one
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(TextBlock(
                text=block.text,
                bbox=block.bbox,
                confidence=block.confidence,
                source="merged" if block.source != kept[0].source if kept else block.source else block.source,
                angle=block.angle,
            ))

    # Sort by reading order: top-to-bottom, left-to-right
    kept.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(img: np.ndarray,
                 use_tesseract: bool = True,
                 use_easyocr: bool = True,
                 languages: List[str] = None) -> OCRResult:
    """
    Extract all text from an engineering drawing image.

    Uses both Tesseract and EasyOCR when available, merges results.
    Falls back to whichever engine is installed if only one is available.

    Args:
        img: BGR or grayscale image (np.ndarray).
        use_tesseract: Try Tesseract if available.
        use_easyocr: Try EasyOCR if available.
        languages: Language codes for EasyOCR (default: ["en"]).

    Returns:
        OCRResult with deduplicated text blocks and concatenated full text.
    """
    all_blocks = []
    engines_used = []

    if use_tesseract and _check_tesseract():
        try:
            tess_blocks = _run_tesseract_multiconfig(img)
            all_blocks.extend(tess_blocks)
            engines_used.append("tesseract")
        except Exception as e:
            pass  # silently skip if tesseract fails

    if use_easyocr and _check_easyocr():
        try:
            easy_blocks = _run_easyocr(img, languages)
            all_blocks.extend(easy_blocks)
            engines_used.append("easyocr")
        except Exception as e:
            pass  # silently skip if easyocr fails

    if not all_blocks:
        return OCRResult(
            engine_used="none",
            image_shape=img.shape[:2],
        )

    # Merge and deduplicate
    merged = _merge_blocks(all_blocks)

    # Build full text in reading order
    full_text = " ".join(b.text for b in merged)

    engine_label = "hybrid" if len(engines_used) > 1 else (
        engines_used[0] if engines_used else "none"
    )

    return OCRResult(
        text_blocks=merged,
        full_text=full_text,
        engine_used=engine_label,
        image_shape=img.shape[:2],
    )


def extract_text_from_region(img: np.ndarray,
                              roi: Tuple[int, int, int, int],
                              pad: int = 10,
                              **kwargs) -> OCRResult:
    """
    Extract text from a specific region of the image.

    Args:
        img: Full image (BGR).
        roi: Region of interest as (x, y, w, h).
        pad: Pixels of padding around the ROI.
    """
    h, w = img.shape[:2]
    x, y, rw, rh = roi
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + rw + pad)
    y1 = min(h, y + rh + pad)
    crop = img[y0:y1, x0:x1]

    result = extract_text(crop, **kwargs)

    # Adjust bounding boxes back to full-image coordinates
    for block in result.text_blocks:
        bx, by, bw, bh = block.bbox
        block.bbox = (bx + x0, by + y0, bw, bh)

    return result


def get_available_engines() -> List[str]:
    """Return which OCR engines are installed and usable."""
    engines = []
    if _check_tesseract():
        engines.append("tesseract")
    if _check_easyocr():
        engines.append("easyocr")
    return engines


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m core.ocr_engine <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not load image: {sys.argv[1]}")
        sys.exit(1)

    print(f"Available engines: {get_available_engines()}")
    result = extract_text(img)
    print(f"Engine used: {result.engine_used}")
    print(f"Text blocks found: {len(result.text_blocks)}")
    print(f"\nFull text:\n{result.full_text}")
    print(f"\nDimensions found: {len(result.get_dimensions())}")
    for d in result.get_dimensions():
        print(f"  {d.text}  @ {d.bbox}  conf={d.confidence:.2f}")
