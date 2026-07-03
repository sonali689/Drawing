"""
Typo & Cross-View Consistency Checker (v2)
============================================
Checks for:
  (a) typos / spelling mistakes in drawing text (notes, callouts, labels)
  (b) mismatches between the main drawing and its views/sections — same
      panel/feature described or dimensioned differently in different places

v2 improvements:
  - OCR-first approach: extracts text locally before calling LLM
  - Local typo detection via pyspellchecker (no LLM needed for basic typos)
  - LLM used only for semantic reasoning (cross-view mismatch detection)
  - Supports both Ollama (local) and Anthropic (cloud) via AIBackend
  - Domain terms loaded from YAML config
  - Much more accurate: OCR text fed to LLM as context alongside the image
"""
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Set

import cv2
import numpy as np
import yaml

from core.local_llm import AIBackend


# ---------------------------------------------------------------------------
# Domain terms loader
# ---------------------------------------------------------------------------

def load_domain_terms(config_path: str = None) -> List[str]:
    """Load domain terms from YAML config."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "domain_terms.yaml"
        )

    terms = []
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data:
            for category in data.values():
                if isinstance(category, list):
                    terms.extend(category)

    if not terms:
        terms = DEFAULT_DOMAIN_TERMS

    return terms


DEFAULT_DOMAIN_TERMS = [
    "Nylon 420D", "PU coated", "Pantone", "lock-stitch", "CET", "ALV",
    "Autoliv", "OPW", "silicone", "bartack", "overlock", "selvage",
    "grainline", "inflator", "tether", "cushion", "diffuser", "retainer",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TypoFinding:
    location: str      # e.g. "Section A-A note"
    found_text: str
    suggested_fix: str
    source: str = ""   # "ocr_local" or "ai"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "found_text": self.found_text,
            "suggested_fix": self.suggested_fix,
            "source": self.source,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class MismatchFinding:
    subject: str         # e.g. "Side panel width"
    location_a: str
    value_a: str
    location_b: str
    value_b: str
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ConsistencyResult:
    typos: List[TypoFinding] = field(default_factory=list)
    mismatches: List[MismatchFinding] = field(default_factory=list)
    ocr_text_found: str = ""

    def to_dict(self):
        return {
            "typos": [t.to_dict() for t in self.typos],
            "mismatches": [m.to_dict() for m in self.mismatches],
            "typo_count": len(self.typos),
            "mismatch_count": len(self.mismatches),
        }


# ---------------------------------------------------------------------------
# Local typo detection (no LLM needed)
# ---------------------------------------------------------------------------

def _local_typo_check(drawing_color: np.ndarray,
                      domain_terms: List[str],
                      pdf_bytes: Optional[bytes] = None,
                      page_number: int = 1) -> List[TypoFinding]:
    """
    Run local OCR + spellcheck to find typos without any LLM call.
    This is fast, free, and catches obvious spelling mistakes.
    """
    try:
        from core.pdf_handler import get_pdf_or_ocr_text
        from core.text_analyzer import find_typos, load_domain_terms as load_terms_set

        ocr_result = get_pdf_or_ocr_text(drawing_color, pdf_bytes, page_number)
        if not ocr_result.text_blocks:
            return []

        # Build domain terms set
        terms_set = set(t.lower() for t in domain_terms)
        terms_set.update(load_terms_set())

        typo_candidates = find_typos(ocr_result.text_blocks, terms_set)

        return [
            TypoFinding(
                location=f"Region ({tc.location_bbox[0]},{tc.location_bbox[1]})" if tc.location_bbox else "Unknown",
                found_text=tc.original,
                suggested_fix=tc.suggested,
                source="ocr_local",
                confidence=tc.confidence,
            )
            for tc in typo_candidates
            if tc.confidence > 0.3  # only report reasonably confident findings
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# AI-based consistency check
# ---------------------------------------------------------------------------

def _build_prompt(domain_terms: List[str], ocr_text: str = "") -> str:
    terms = ", ".join(domain_terms) if domain_terms else "(none provided)"
    ocr_section = ""
    if ocr_text:
        ocr_section = f"""
The OCR system has pre-extracted the following text from this drawing (may
contain OCR errors — use as supplementary context alongside your visual
analysis of the image):
---
{ocr_text[:3000]}
---
"""

    return f"""You are proofreading an engineering/sewing drawing sheet that may contain
multiple views (main view, sections, details, title block). This is from an
automotive airbag/cushion supplier (Autoliv / CET Thailand).

Do NOT flag these as errors — they are known-correct domain terms/jargon:
{terms}
{ocr_section}
Do two things:

1. TYPOS: Find genuine spelling mistakes or typos in the drawing's text
   (notes, callouts, labels, dimensions text) — things like "seem" instead
   of "seam", transposed letters, obviously wrong words. Do not flag:
   - Abbreviations, part numbers, or the domain terms listed above
   - OCR artifacts (if you can see the text clearly in the image and it's
     correct, don't flag it even if the OCR text above shows garbled text)

2. CROSS-VIEW MISMATCHES: Find cases where the SAME panel, feature, or
   dimension is described differently in different parts of the drawing:
   - A dimension value given differently between views (e.g., 120mm main
     view vs 125mm section view)
   - A sewing method described one way in the notes but differently in a
     detail view
   - Panel shapes or configurations that don't match between views
   - Material callouts that contradict each other

   Only flag things that refer to the SAME real-world feature. Don't flag
   two different features that happen to look similar.

Respond with ONLY valid JSON, no markdown fences, no preamble, in this exact
shape:
{{
  "typos": [{{"location": "<where on the sheet>", "found_text": "<the misspelled text>", "suggested_fix": "<corrected text>"}}],
  "mismatches": [{{"subject": "<what feature/dimension>", "location_a": "<where>", "value_a": "<value/description>", "location_b": "<where>", "value_b": "<value/description>", "note": "<why this is a mismatch>"}}]
}}

If there are no typos or no mismatches, return empty arrays. Do not invent issues.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_consistency(drawing_color: np.ndarray, domain_terms: List[str] = None,
                      backend: str = "anthropic",
                      api_key: Optional[str] = None,
                      vision_model: Optional[str] = None,
                      include_local_typos: bool = True,
                      pdf_bytes: Optional[bytes] = None,
                      page_number: int = 1) -> ConsistencyResult:
    """
    Check a drawing for typos and cross-view mismatches.

    Uses a hybrid approach:
    1. Local OCR + spellcheck for fast typo detection (no AI cost)
    2. AI vision call for semantic cross-view mismatch detection

    Args:
        drawing_color: Drawing image (BGR).
        domain_terms: Words that should NOT be flagged as typos.
        backend: "ollama" for local, "anthropic" for cloud.
        api_key: Required for Anthropic backend.
        vision_model: Override the default model.
        include_local_typos: Whether to run local OCR-based typo check too.
        pdf_bytes: Optional raw PDF bytes for vector text parsing.
        page_number: The PDF page number.
    """
    if domain_terms is None:
        domain_terms = load_domain_terms()

    # Phase 1: Local typo detection (fast, free)
    local_typos = []
    ocr_text = ""
    if include_local_typos:
        local_typos = _local_typo_check(drawing_color, domain_terms, pdf_bytes, page_number)
        try:
            from core.pdf_handler import get_pdf_or_ocr_text
            ocr_result = get_pdf_or_ocr_text(drawing_color, pdf_bytes, page_number)
            ocr_text = ocr_result.full_text
        except Exception:
            pass

    # Phase 2: AI-based analysis for cross-view mismatches and complex typos
    ai = AIBackend(
        backend=backend,
        api_key=api_key,
        vision_model=vision_model,
    )

    prompt = _build_prompt(domain_terms, ocr_text)
    raw_response = ai.call_vision([drawing_color], prompt, max_tokens=2000)

    from core.local_llm import parse_json_response
    parsed = parse_json_response(raw_response)

    # Parse AI typos
    ai_typos = []
    for t in parsed.get("typos", []):
        ai_typos.append(TypoFinding(
            location=t.get("location", ""),
            found_text=t.get("found_text", ""),
            suggested_fix=t.get("suggested_fix", ""),
            source="ai",
            confidence=0.8,
        ))

    # Parse mismatches
    mismatches = []
    for m in parsed.get("mismatches", []):
        mismatches.append(MismatchFinding(
            subject=m.get("subject", ""),
            location_a=m.get("location_a", ""),
            value_a=m.get("value_a", ""),
            location_b=m.get("location_b", ""),
            value_b=m.get("value_b", ""),
            note=m.get("note", ""),
        ))

    # Merge typos: combine local and AI findings, deduplicate
    all_typos = _merge_typos(local_typos, ai_typos)

    return ConsistencyResult(
        typos=all_typos,
        mismatches=mismatches,
        ocr_text_found=ocr_text,
    )


def _merge_typos(local: List[TypoFinding],
                  ai: List[TypoFinding]) -> List[TypoFinding]:
    """
    Merge typos from local OCR check and AI check, deduplicating.
    If both found the same typo, keep the AI version (usually has better context).
    """
    merged = []
    ai_texts = {t.found_text.lower() for t in ai}

    # Add local typos that weren't also found by AI
    for lt in local:
        if lt.found_text.lower() not in ai_texts:
            merged.append(lt)

    # Add all AI typos
    merged.extend(ai)

    # Sort: AI findings first (more reliable), then local
    merged.sort(key=lambda t: (0 if t.source == "ai" else 1, -t.confidence))
    return merged
