"""
Automated Drawing Comparator - Enhanced Pipeline
==================================================
Upgraded from the original ORB-based pipeline to handle complex engineering
drawings with much higher accuracy:

  Master Drawing + New Revision
        -> Pre-processing (CLAHE + adaptive normalization)
        -> Image Alignment (SIFT + RANSAC homography)
        -> Pixel Subtraction + SSIM structural comparison
        -> Threshold Filter (adaptive noise removal)
        -> OCR text extraction + structured text diff
        -> Discrepancy Map (category-coded regions)
        -> Report Output (flagged changes exported)

Key improvements over v1:
  - SIFT features (much more robust than ORB for engineering drawings)
  - SSIM for structural similarity (reduces false positives on scan noise)
  - CLAHE preprocessing (handles mixed contrast areas)
  - Multi-scale analysis (catches both large geometry and small text changes)
  - Category-based output: geometry, dimension, text, annotation changes
  - OCR-integrated: compares extracted text alongside pixel differences
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
import time


@dataclass
class Discrepancy:
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_px: int
    severity: str          # "minor" | "moderate" | "major"
    category: str = ""     # "geometry" | "dimension" | "text" | "annotation" | "unknown"
    description: str = ""  # human-readable description of the change

    def to_dict(self):
        return {
            "id": self.id,
            "bbox": self.bbox,
            "area_px": self.area_px,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
        }


@dataclass
class ComparisonResult:
    aligned_ok: bool
    match_confidence: float
    discrepancies: List[Discrepancy] = field(default_factory=list)
    discrepancy_map: np.ndarray = None       # binary mask
    ssim_map: np.ndarray = None              # SSIM difference map (grayscale)
    annotated_revision: np.ndarray = None    # revision image with boxes drawn
    aligned_revision: np.ndarray = None      # revision warped onto master's coordinate frame
    side_by_side: np.ndarray = None
    processing_time_s: float = 0.0
    ssim_score: float = 0.0                  # overall SSIM (1.0 = identical)

    def to_report_dict(self):
        return {
            "aligned_ok": self.aligned_ok,
            "match_confidence": round(self.match_confidence, 4),
            "ssim_score": round(self.ssim_score, 4),
            "num_discrepancies": len(self.discrepancies),
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "processing_time_s": round(self.processing_time_s, 3),
        }


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """
    Enhanced grayscale + normalization using CLAHE for better handling of
    engineering drawings with mixed contrast areas (dense hatching next to
    clean annotation spaces).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    else:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def align_images(master_gray: np.ndarray, revision_gray: np.ndarray,
                  revision_color: np.ndarray,
                  min_match_count: int = 15,
                  use_sift: bool = True) -> Tuple[np.ndarray, bool, float]:
    """
    Aligns the revision image onto the master using feature matching +
    homography (RANSAC).

    SIFT (default) is significantly more robust than ORB for engineering
    drawings — it handles scale changes, is more invariant to rotation,
    and produces far fewer false matches on repetitive line patterns.

    Falls back to ORB if SIFT is not available (OpenCV built without
    non-free modules).

    Returns: (aligned_revision_color, success, match_confidence)
    """
    if use_sift:
        try:
            detector = cv2.SIFT_create(nfeatures=5000)
            norm_type = cv2.NORM_L2
        except cv2.error:
            # SIFT not available — fall back to ORB
            detector = cv2.ORB_create(nfeatures=5000)
            norm_type = cv2.NORM_HAMMING
    else:
        detector = cv2.ORB_create(nfeatures=5000)
        norm_type = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(master_gray, None)
    kp2, des2 = detector.detectAndCompute(revision_gray, None)

    if des1 is None or des2 is None or len(kp1) < min_match_count or len(kp2) < min_match_count:
        return revision_color, False, 0.0

    # FLANN-based matching for SIFT (faster and more accurate than brute-force for large descriptor sets)
    if norm_type == cv2.NORM_L2:
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < min_match_count:
        return revision_color, False, len(good) / max(min_match_count, 1)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return revision_color, False, 0.0

    inliers = int(mask.sum()) if mask is not None else 0
    confidence = inliers / len(good)

    h, w = master_gray.shape[:2]
    aligned = cv2.warpPerspective(revision_color, H, (w, h), borderValue=(255, 255, 255))
    return aligned, True, confidence


def compute_ssim_map(master_gray: np.ndarray,
                      aligned_revision_gray: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute Structural Similarity Index (SSIM) between master and revision.

    SSIM is much more perceptually meaningful than raw pixel difference —
    it compares luminance, contrast, and structure, which means it ignores
    uniform brightness/contrast shifts (common in scans) while still
    catching actual structural changes.

    Returns: (overall_ssim_score, ssim_difference_map)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        score, ssim_image = ssim(master_gray, aligned_revision_gray,
                                  full=True, win_size=7)
        # Convert SSIM map to a difference map (0 = identical, 255 = very different)
        diff_map = ((1.0 - ssim_image) * 255).astype(np.uint8)
        return score, diff_map
    except ImportError:
        # Fallback: basic pixel difference
        diff = cv2.absdiff(master_gray, aligned_revision_gray)
        score = 1.0 - (np.mean(diff) / 255.0)
        return score, diff


def compute_discrepancy_map(master_gray: np.ndarray, aligned_revision_gray: np.ndarray,
                             threshold: int = 30, kernel_size: int = 5,
                             merge_dilate_iters: int = 3,
                             use_ssim: bool = True) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Enhanced discrepancy detection combining pixel diff and SSIM.

    Returns: (binary_mask, ssim_score, ssim_map)
    """
    # Raw pixel difference
    pixel_diff = cv2.absdiff(master_gray, aligned_revision_gray)
    _, pixel_mask = cv2.threshold(pixel_diff, threshold, 255, cv2.THRESH_BINARY)

    ssim_score = 0.0
    ssim_map = None

    if use_ssim:
        ssim_score, ssim_map = compute_ssim_map(master_gray, aligned_revision_gray)

        # Combine pixel diff mask with SSIM-based mask
        # SSIM catches structural changes that might not show up in raw pixel diff
        _, ssim_mask = cv2.threshold(ssim_map, 80, 255, cv2.THRESH_BINARY)

        # Union of both masks — catch everything either method finds
        combined = cv2.bitwise_or(pixel_mask, ssim_mask)
    else:
        combined = pixel_mask

    # Morphological cleanup
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)             # remove speckle
    combined = cv2.dilate(combined, kernel, iterations=merge_dilate_iters)    # merge nearby diffs
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)  # fill gaps

    return combined, ssim_score, ssim_map


def extract_discrepancies(mask: np.ndarray, min_area: int = 40) -> List[Discrepancy]:
    """Finds connected regions in the discrepancy map and classifies severity by area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    discrepancies = []
    idx = 1
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if area < 300:
            severity = "minor"
        elif area < 1500:
            severity = "moderate"
        else:
            severity = "major"
        discrepancies.append(Discrepancy(
            id=idx, bbox=(x, y, w, h), area_px=int(area),
            severity=severity, category="unknown",
        ))
        idx += 1
    # Largest first
    discrepancies.sort(key=lambda d: d.area_px, reverse=True)
    for i, d in enumerate(discrepancies, start=1):
        d.id = i
    return discrepancies


def classify_discrepancy_regions(discrepancies: List[Discrepancy],
                                  master_ocr, revision_ocr) -> List[Discrepancy]:
    """
    Classify each discrepancy region by checking whether it overlaps with
    OCR-detected text blocks.

    Categories:
    - "dimension": overlaps text that looks like a numeric dimension
    - "text": overlaps text that's non-numeric (notes, labels)
    - "geometry": no text overlap — likely a line/shape change
    - "annotation": small text change near arrows or leaders
    """
    for d in discrepancies:
        dx, dy, dw, dh = d.bbox

        # Check if this region overlaps any OCR text blocks
        master_texts = master_ocr.get_text_in_region(d.bbox, margin=20)
        revision_texts = revision_ocr.get_text_in_region(d.bbox, margin=20)
        all_texts = master_texts + revision_texts

        if all_texts:
            # Check if any overlapping text looks like a dimension
            has_dimension = any(
                _looks_like_dimension(t.text) for t in all_texts
            )
            if has_dimension:
                d.category = "dimension"
                # Build description from the text difference
                m_text = " ".join(t.text for t in master_texts)
                r_text = " ".join(t.text for t in revision_texts)
                if m_text != r_text:
                    d.description = f"Dimension change: \"{m_text}\" → \"{r_text}\""
            else:
                d.category = "text"
                m_text = " ".join(t.text for t in master_texts)
                r_text = " ".join(t.text for t in revision_texts)
                if m_text != r_text:
                    d.description = f"Text change: \"{m_text}\" → \"{r_text}\""
        else:
            d.category = "geometry"
            d.description = f"Geometry change at ({dx}, {dy}), area={d.area_px}px²"

    return discrepancies


def _looks_like_dimension(text: str) -> bool:
    """Check if a text string looks like a numeric dimension."""
    import re
    return bool(re.search(
        r'\d+[.,]?\d*\s*(?:mm|cm|in|°|"|±|Ø|R\s*\d|x\s*\d)?',
        text, re.IGNORECASE
    ))


def annotate(revision_color: np.ndarray, discrepancies: List[Discrepancy]) -> np.ndarray:
    """
    Draws highly visible annotations on the revision image:
    - Thick colored bounding boxes around each changed region
    - Semi-transparent color overlay inside each box
    - Large filled circle badge with white ID number
    Every flagged region is designed to be immediately visible even on dense drawings.
    """
    # Color by category (BGR)
    category_colors = {
        "geometry": (0, 0, 255),       # red
        "dimension": (0, 140, 255),    # orange
        "text": (0, 200, 255),         # yellow
        "annotation": (255, 200, 0),   # cyan
        "unknown": (180, 180, 180),    # gray
    }

    out = revision_color.copy()
    overlay = out.copy()

    for d in discrepancies:
        x, y, w, h = d.bbox
        color = category_colors.get(d.category, (180, 180, 180))
        pad = 6

        # 1. Semi-transparent color fill inside the bounding box
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(out.shape[1], x + w + pad), min(out.shape[0], y + h + pad)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)

        # 2. Thick bounding box outline
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)

        # 3. Large filled circle badge with ID number (top-left corner)
        badge_radius = 18
        badge_cx = max(badge_radius + 2, x0)
        badge_cy = max(badge_radius + 2, y0 - badge_radius - 6)
        # Ensure badge stays within image bounds
        badge_cy = max(badge_radius + 2, badge_cy)
        badge_cx = min(out.shape[1] - badge_radius - 2, badge_cx)

        # Black outline circle + colored fill
        cv2.circle(out, (badge_cx, badge_cy), badge_radius + 2, (0, 0, 0), -1)
        cv2.circle(out, (badge_cx, badge_cy), badge_radius, color, -1)
        cv2.circle(overlay, (badge_cx, badge_cy), badge_radius + 2, (0, 0, 0), -1)
        cv2.circle(overlay, (badge_cx, badge_cy), badge_radius, color, -1)

        # White ID number centered in the badge
        label = str(d.id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 if len(label) <= 2 else 0.55
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = badge_cx - tw // 2
        text_y = badge_cy + th // 2
        cv2.putText(out, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    # Blend the overlay (semi-transparent fill) with the original
    alpha = 0.15
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    # Re-draw the bounding box outlines and badges at full opacity on top
    for d in discrepancies:
        x, y, w, h = d.bbox
        color = category_colors.get(d.category, (180, 180, 180))
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(out.shape[1], x + w + pad), min(out.shape[0], y + h + pad)

        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)

        badge_radius = 18
        badge_cx = max(badge_radius + 2, x0)
        badge_cy = max(badge_radius + 2, y0 - badge_radius - 6)
        badge_cy = max(badge_radius + 2, badge_cy)
        badge_cx = min(out.shape[1] - badge_radius - 2, badge_cx)

        cv2.circle(out, (badge_cx, badge_cy), badge_radius + 2, (0, 0, 0), -1)
        cv2.circle(out, (badge_cx, badge_cy), badge_radius, color, -1)

        label = str(d.id)
        font_scale = 0.7 if len(label) <= 2 else 0.55
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.putText(out, label,
                    (badge_cx - tw // 2, badge_cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    return out


def crop_region(img: np.ndarray, bbox: Tuple[int, int, int, int],
                context_factor: float = 2.0,
                min_size: int = 150) -> np.ndarray:
    """
    Crop a region from the image with contextual padding so you can see
    what's around the change. Used by the interactive viewer.

    Args:
        img: Source image (BGR).
        bbox: (x, y, w, h) bounding box of the region.
        context_factor: How much extra context to include (2.0 = 2x the region size).
        min_size: Minimum crop width/height in pixels.
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = bbox

    # Add context padding
    pad_w = max(int(w * context_factor / 2), min_size // 2)
    pad_h = max(int(h * context_factor / 2), min_size // 2)

    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(w_img, x + w + pad_w)
    y1 = min(h_img, y + h + pad_h)

    return img[y0:y1, x0:x1]

def make_side_by_side(master_color, annotated_revision) -> np.ndarray:
    h = max(master_color.shape[0], annotated_revision.shape[0])
    w = master_color.shape[1] + annotated_revision.shape[1] + 20
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[:master_color.shape[0], :master_color.shape[1]] = master_color
    x_off = master_color.shape[1] + 20
    canvas[:annotated_revision.shape[0], x_off:x_off + annotated_revision.shape[1]] = annotated_revision
    cv2.line(canvas, (master_color.shape[1] + 10, 0), (master_color.shape[1] + 10, h), (200, 200, 200), 2)
    return canvas


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_drawings(master_color: np.ndarray, revision_color: np.ndarray,
                      threshold: int = 30, min_area: int = 250,
                      use_sift: bool = True, use_ssim: bool = True,
                      use_ocr: bool = False) -> ComparisonResult:
    """
    Full comparison pipeline.

    Args:
        master_color: Master drawing (BGR).
        revision_color: Revised drawing (BGR).
        threshold: Pixel difference threshold for binary mask.
        min_area: Minimum contour area to report.
        use_sift: Use SIFT features (True) or ORB (False).
        use_ssim: Use SSIM structural comparison alongside pixel diff.
        use_ocr: Run OCR to classify discrepancy regions by content type.
    """
    t0 = time.time()

    master_gray = preprocess(master_color)
    revision_gray_raw = preprocess(revision_color)

    aligned_revision_color, aligned_ok, confidence = align_images(
        master_gray, revision_gray_raw, revision_color, use_sift=use_sift
    )
    aligned_revision_gray = preprocess(aligned_revision_color)

    mask, ssim_score, ssim_map = compute_discrepancy_map(
        master_gray, aligned_revision_gray,
        threshold=threshold, use_ssim=use_ssim,
    )
    discrepancies = extract_discrepancies(mask, min_area=min_area)

    # Optional: classify discrepancy regions using OCR
    if use_ocr:
        try:
            from core.ocr_engine import extract_text
            master_ocr = extract_text(master_color)
            revision_ocr = extract_text(aligned_revision_color)
            discrepancies = classify_discrepancy_regions(
                discrepancies, master_ocr, revision_ocr
            )
        except Exception:
            pass  # OCR classification is best-effort

    annotated = annotate(aligned_revision_color, discrepancies)
    side_by_side = make_side_by_side(master_color, annotated)

    return ComparisonResult(
        aligned_ok=aligned_ok,
        match_confidence=confidence,
        discrepancies=discrepancies,
        discrepancy_map=mask,
        ssim_map=ssim_map,
        annotated_revision=annotated,
        aligned_revision=aligned_revision_color,
        side_by_side=side_by_side,
        processing_time_s=time.time() - t0,
        ssim_score=ssim_score,
    )


if __name__ == "__main__":
    # Quick smoke test against the generated sample drawings
    master = cv2.imread("samples/master_drawing.png")
    revision = cv2.imread("samples/revised_drawing.png")
    result = compare_drawings(master, revision)
    print(json.dumps(result.to_report_dict(), indent=2))
