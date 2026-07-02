"""
Automated Drawing Comparator - Core Pipeline
==============================================
Implements the pipeline from the tech-stack slide:

  Master Drawing + New Revision
        -> Pre-processing (grayscale + normalization)
        -> Image Alignment (OpenCV feature matching)
        -> Pixel Subtraction (NumPy matrix diff)
        -> Threshold Filter (noise removal)
        -> Discrepancy Map (highlighted error regions)
        -> Report Output (flagged changes exported)

Language: Python | Libraries: OpenCV + NumPy
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import json
import time


@dataclass
class Discrepancy:
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_px: int
    severity: str  # "minor" | "moderate" | "major"

    def to_dict(self):
        return {
            "id": self.id,
            "bbox": self.bbox,
            "area_px": self.area_px,
            "severity": self.severity,
        }


@dataclass
class ComparisonResult:
    aligned_ok: bool
    match_confidence: float
    discrepancies: List[Discrepancy] = field(default_factory=list)
    discrepancy_map: np.ndarray = None       # binary mask
    annotated_revision: np.ndarray = None    # revision image with boxes drawn
    aligned_revision: np.ndarray = None      # revision image warped onto master's coordinate frame (no boxes)
    side_by_side: np.ndarray = None
    processing_time_s: float = 0.0

    def to_report_dict(self):
        return {
            "aligned_ok": self.aligned_ok,
            "match_confidence": round(self.match_confidence, 4),
            "num_discrepancies": len(self.discrepancies),
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "processing_time_s": round(self.processing_time_s, 3),
        }


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray) -> np.ndarray:
    """Grayscale + normalization + light denoise."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def align_images(master_gray: np.ndarray, revision_gray: np.ndarray,
                  revision_color: np.ndarray,
                  min_match_count: int = 15) -> Tuple[np.ndarray, bool, float]:
    """
    Aligns the revision image onto the master using ORB feature matching +
    homography (RANSAC). This corrects scan skew, minor rotation, and offset
    so that pixel subtraction only reflects real design changes, not
    misalignment noise.

    Returns: (aligned_revision_color, success, match_confidence)
    """
    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(master_gray, None)
    kp2, des2 = orb.detectAndCompute(revision_gray, None)

    if des1 is None or des2 is None or len(kp1) < min_match_count or len(kp2) < min_match_count:
        return revision_color, False, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
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


def compute_discrepancy_map(master_gray: np.ndarray, aligned_revision_gray: np.ndarray,
                             threshold: int = 30, kernel_size: int = 5,
                             merge_dilate_iters: int = 3) -> np.ndarray:
    """
    Pixel subtraction (NumPy) + threshold filter + morphological noise removal.

    Line-art drawings are thin strokes, so even a well-aligned pair produces
    scattered edge-level diffs (sub-pixel jitter along every line). The extra
    dilation pass merges these scattered fragments that sit close together
    into single contiguous "changed area" blobs, rather than dozens of tiny
    slivers, which is what a human reviewer actually wants to see.
    """
    diff = cv2.absdiff(master_gray, aligned_revision_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)                    # remove speckle noise
    mask = cv2.dilate(mask, kernel, iterations=merge_dilate_iters)           # merge nearby edge-diffs
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)     # fill small gaps

    return mask


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
        discrepancies.append(Discrepancy(id=idx, bbox=(x, y, w, h), area_px=int(area), severity=severity))
        idx += 1
    # Largest first
    discrepancies.sort(key=lambda d: d.area_px, reverse=True)
    for i, d in enumerate(discrepancies, start=1):
        d.id = i
    return discrepancies


def annotate(revision_color: np.ndarray, discrepancies: List[Discrepancy]) -> np.ndarray:
    """Draws color-coded bounding boxes + labels on the revision image."""
    color_map = {"minor": (0, 200, 255), "moderate": (0, 140, 255), "major": (0, 0, 255)}
    out = revision_color.copy()
    for d in discrepancies:
        x, y, w, h = d.bbox
        color = color_map[d.severity]
        cv2.rectangle(out, (x - 4, y - 4), (x + w + 4, y + h + 4), color, 2)
        cv2.putText(out, f"#{d.id}", (x - 4, max(y - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


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
                      threshold: int = 30, min_area: int = 250) -> ComparisonResult:
    t0 = time.time()

    master_gray = preprocess(master_color)
    revision_gray_raw = preprocess(revision_color)

    aligned_revision_color, aligned_ok, confidence = align_images(
        master_gray, revision_gray_raw, revision_color
    )
    aligned_revision_gray = preprocess(aligned_revision_color)

    mask = compute_discrepancy_map(master_gray, aligned_revision_gray, threshold=threshold)
    discrepancies = extract_discrepancies(mask, min_area=min_area)
    annotated = annotate(aligned_revision_color, discrepancies)
    side_by_side = make_side_by_side(master_color, annotated)

    return ComparisonResult(
        aligned_ok=aligned_ok,
        match_confidence=confidence,
        discrepancies=discrepancies,
        discrepancy_map=mask,
        annotated_revision=annotated,
        aligned_revision=aligned_revision_color,
        side_by_side=side_by_side,
        processing_time_s=time.time() - t0,
    )


if __name__ == "__main__":
    # Quick smoke test against the generated sample drawings
    master = cv2.imread("samples/master_drawing.png")
    revision = cv2.imread("samples/revised_drawing.png")
    result = compare_drawings(master, revision)
    print(json.dumps(result.to_report_dict(), indent=2))
    cv2.imwrite("reports/discrepancy_map.png", result.discrepancy_map)
    cv2.imwrite("reports/annotated_revision.png", result.annotated_revision)
    cv2.imwrite("reports/side_by_side.png", result.side_by_side)
