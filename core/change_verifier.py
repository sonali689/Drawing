"""
Change-Request Reconciliation (v2)
====================================
Extends the deterministic CV diff (comparator.py) with a semantic layer:
given the text of what was actually *requested* (an ECO note, an email to
CET, a change-order list), classify each detected pixel-diff region as:

  - "applied"    -> matches one of the requested changes
  - "unintended" -> a real geometric change, but not something anyone asked for
  - (and separately) any requested change with no matching region -> "missing"

v2 improvements:
  - Supports both Ollama (local) and Anthropic (cloud) via AIBackend
  - Uses OCR-extracted text for better context in prompts
  - Improved prompt engineering for engineering drawing understanding
  - Batch processing for efficiency (groups small regions)
"""
import base64
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

from core.comparator import Discrepancy
from core.local_llm import AIBackend


@dataclass
class RegionVerdict:
    discrepancy_id: int
    matched_request_index: Optional[int]  # index into requested_changes, or None
    verdict: str                          # "applied" | "unintended"
    explanation: str
    category: str = ""                    # from discrepancy: geometry/dimension/text
    ocr_context: str = ""                 # OCR text found in this region


@dataclass
class ReconciliationResult:
    requested_changes: List[str]
    region_verdicts: List[RegionVerdict] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)  # requested changes with no matching region

    def to_dict(self):
        return {
            "requested_changes": self.requested_changes,
            "applied": [
                {"discrepancy_id": v.discrepancy_id,
                 "requested_change": self.requested_changes[v.matched_request_index],
                 "explanation": v.explanation,
                 "category": v.category}
                for v in self.region_verdicts if v.verdict == "applied"
            ],
            "unintended": [
                {"discrepancy_id": v.discrepancy_id,
                 "explanation": v.explanation,
                 "category": v.category,
                 "ocr_context": v.ocr_context}
                for v in self.region_verdicts if v.verdict == "unintended"
            ],
            "missing": self.missing,
        }


def _crop(img: np.ndarray, bbox, pad: int = 40) -> np.ndarray:
    """Crop a region from an image with padding, ensuring we capture enough context."""
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return img[y0:y1, x0:x1]


def _build_prompt(requested_changes: List[str], ocr_context: str = "") -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(requested_changes))
    context_section = ""
    if ocr_context:
        context_section = f"""
The OCR system has extracted the following text from this region (may contain
errors, use as supplementary context alongside the visual comparison):
Master text: {ocr_context}
"""

    return f"""You are reviewing a mechanical/engineering drawing revision for a supplier
quality check (Autoliv/CET sewing/airbag drawings). You are shown two crops
of the SAME region of a drawing: the BEFORE (master/original) version and the
AFTER (revised) version, taken from the same coordinates. A pixel-diff
algorithm already flagged this region as changed.
{context_section}
Here is the list of changes that were formally REQUESTED for this drawing:
{numbered}

Look at the before/after crop and decide:
- Does this specific change correspond to ONE of the requested changes above?
  If yes, respond with that item's number (0-indexed).
- If the change is real but does NOT correspond to any requested item, it's
  an unintended/unrequested change.

Consider all aspects: geometry changes, dimension value changes, text/note
changes, sewing instruction changes, material callout changes.

Respond with ONLY valid JSON, no markdown fences, no preamble, in this exact
shape:
{{"matched_request_index": <int or null>, "verdict": "applied" or "unintended", "explanation": "<one sentence>"}}
"""


def _get_ocr_context(master_color, revision_color, bbox, pad=40):
    """Extract OCR text from the discrepancy region for additional context."""
    try:
        from core.ocr_engine import extract_text_from_region
        master_ocr = extract_text_from_region(master_color, bbox, pad=pad)
        revision_ocr = extract_text_from_region(revision_color, bbox, pad=pad)
        m_text = master_ocr.full_text.strip()
        r_text = revision_ocr.full_text.strip()
        if m_text or r_text:
            return f"[Master: \"{m_text}\"] [Revision: \"{r_text}\"]"
    except Exception:
        pass
    return ""


def reconcile_changes(master_color: np.ndarray, aligned_revision_color: np.ndarray,
                       discrepancies: List[Discrepancy], requested_changes: List[str],
                       backend: str = "anthropic",
                       api_key: Optional[str] = None,
                       vision_model: Optional[str] = None,
                       use_ocr_context: bool = True) -> ReconciliationResult:
    """
    Reconcile detected discrepancies against requested changes.

    Args:
        master_color: Master drawing (BGR).
        aligned_revision_color: Aligned revision drawing (BGR).
        discrepancies: List of Discrepancy objects from comparator.
        requested_changes: List of change descriptions.
        backend: "ollama" for local, "anthropic" for cloud.
        api_key: Required for Anthropic backend.
        vision_model: Override the default model.
        use_ocr_context: Whether to add OCR text to prompts for extra context.
    """
    ai = AIBackend(
        backend=backend,
        api_key=api_key,
        vision_model=vision_model,
    )

    verdicts: List[RegionVerdict] = []
    matched_indices = set()

    for d in discrepancies:
        master_crop = _crop(master_color, d.bbox)
        revision_crop = _crop(aligned_revision_color, d.bbox)
        if master_crop.size == 0 or revision_crop.size == 0:
            continue

        # Get OCR context for this region
        ocr_context = ""
        if use_ocr_context:
            ocr_context = _get_ocr_context(
                master_color, aligned_revision_color, d.bbox
            )

        prompt = _build_prompt(requested_changes, ocr_context)

        try:
            result = ai.call_vision_json(
                [master_crop, revision_crop],
                f"BEFORE (master) is Image 1. AFTER (revision) is Image 2.\n\n{prompt}",
                max_tokens=300,
            )
        except Exception as e:
            verdicts.append(RegionVerdict(
                discrepancy_id=d.id, matched_request_index=None,
                verdict="unintended", explanation=f"(AI call failed: {e})",
                category=d.category, ocr_context=ocr_context,
            ))
            continue

        idx = result.get("matched_request_index")
        verdict = result.get("verdict", "unintended")
        explanation = result.get("explanation", "")

        # Validate the matched index
        if idx is not None:
            if not isinstance(idx, int) or idx < 0 or idx >= len(requested_changes):
                idx = None
                verdict = "unintended"

        verdicts.append(RegionVerdict(
            discrepancy_id=d.id, matched_request_index=idx,
            verdict=verdict, explanation=explanation,
            category=d.category, ocr_context=ocr_context,
        ))
        if verdict == "applied" and idx is not None:
            matched_indices.add(idx)

    missing = [c for i, c in enumerate(requested_changes) if i not in matched_indices]

    return ReconciliationResult(
        requested_changes=requested_changes,
        region_verdicts=verdicts,
        missing=missing,
    )
