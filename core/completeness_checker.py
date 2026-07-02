"""
Prototype Instruction Completeness Checker (v2)
=================================================
For prototype requests: confirms whether the drawing is missing, or has
ambiguous/incorrect, required instructions -- e.g. base fabric type, sewing
method, panel positioning method.

v2 improvements:
  - Loads checklist from YAML config (editable without touching code)
  - Supports both Ollama (local) and Anthropic (cloud) via AIBackend
  - Uses OCR to pre-extract text, providing both visual and textual evidence
  - Expanded default checklist with Autoliv/CET-specific items
  - Better prompt engineering for sewing/airbag drawings
"""
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import yaml

from core.local_llm import AIBackend


# ---------------------------------------------------------------------------
# Checklist loader
# ---------------------------------------------------------------------------

def load_checklist(config_path: str = None) -> List[str]:
    """
    Load the prototype instruction checklist from the YAML config.
    Falls back to a hardcoded default if the file isn't found.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "autoliv_checklist.yaml"
        )

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and "prototype_checklist" in data:
            return data["prototype_checklist"]

    # Fallback defaults
    return DEFAULT_CHECKLIST


DEFAULT_CHECKLIST = [
    "Base fabric type / material specification",
    "Sewing method / stitch type (e.g., lock-stitch, chain-stitch, overlock)",
    "Seam allowance dimensions",
    "Panel positioning method / orientation marks",
    "Tether routing and attachment method",
    "Inflator pocket specification",
    "Vent hole size, shape, and placement",
    "Fold pattern / packing instructions",
    "Part number and revision in title block",
    "Drawing scale and units",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChecklistItemResult:
    item: str
    status: str            # "present" | "missing" | "ambiguous"
    evidence: str           # quoted/paraphrased text found, or "" if missing
    note: str = ""
    location: str = ""     # where on the drawing the evidence was found

    def to_dict(self) -> dict:
        return {
            "item": self.item,
            "status": self.status,
            "evidence": self.evidence,
            "note": self.note,
            "location": self.location,
        }


@dataclass
class CompletenessResult:
    items: List[ChecklistItemResult] = field(default_factory=list)
    ocr_text_found: str = ""  # all text the OCR found (for debugging)

    def to_dict(self):
        return {
            "items": [i.to_dict() for i in self.items],
            "missing_count": sum(1 for i in self.items if i.status == "missing"),
            "ambiguous_count": sum(1 for i in self.items if i.status == "ambiguous"),
            "present_count": sum(1 for i in self.items if i.status == "present"),
            "total_count": len(self.items),
        }


# ---------------------------------------------------------------------------
# Pre-check with OCR (local, no AI needed)
# ---------------------------------------------------------------------------

def _ocr_precheck(drawing_color: np.ndarray, checklist: List[str]) -> dict:
    """
    Quick local check using OCR — looks for keyword matches before calling
    the LLM. This gives the LLM better context and catches obvious present/missing
    items without needing expensive AI calls.
    """
    try:
        from core.ocr_engine import extract_text
        ocr_result = extract_text(drawing_color)
        full_text = ocr_result.full_text.lower()

        precheck = {}
        for item in checklist:
            # Extract key terms from the checklist item
            keywords = _extract_keywords(item)
            matches = [kw for kw in keywords if kw.lower() in full_text]
            precheck[item] = {
                "keywords": keywords,
                "matches": matches,
                "likely_present": len(matches) >= len(keywords) * 0.5,
                "full_text": ocr_result.full_text,
            }
        return precheck
    except Exception:
        return {}


def _extract_keywords(item: str) -> List[str]:
    """Extract searchable keywords from a checklist item description."""
    import re
    # Remove parenthetical examples
    clean = re.sub(r'\(.*?\)', '', item)
    # Remove common filler words
    fillers = {'the', 'a', 'an', 'or', 'and', 'of', 'in', 'for', 'with', 'e.g.', '/'}
    words = [w.strip().lower() for w in re.split(r'[\s,/]+', clean)]
    return [w for w in words if w and w not in fillers and len(w) > 2]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_prompt(checklist: List[str], ocr_context: str = "") -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(checklist))
    ocr_section = ""
    if ocr_context:
        ocr_section = f"""
The OCR system has extracted the following text from this drawing (may contain
errors, use as supplementary context alongside the visual analysis):
---
{ocr_context[:3000]}
---
"""

    return f"""You are reviewing an engineering/sewing/airbag prototype drawing for a
manufacturing request at an automotive safety supplier. This drawing specifies
how to manufacture a prototype airbag cushion, panel, or similar textile component.

Check whether the drawing contains clear, unambiguous instructions for each of
the following required items:

{numbered}

{ocr_section}

For EACH item, look across the entire drawing — all views, sections, detail
callouts, notes areas, specification blocks, title block, and any legends or
tables. Engineering drawings often place critical info in notes sections
outside the views, or in the title block.

For each item, decide:
- "present": the instruction is clearly and unambiguously given. Quote or
  closely paraphrase the actual text/annotation you found (max 20 words).
  Also state WHERE on the drawing you found it (e.g., "notes section",
  "title block", "Section A-A", "main view callout").
- "missing": there is no instruction anywhere on the drawing for this item.
  Suggest what should be added in the "note" field.
- "ambiguous": something is there but it's unclear, contradictory, or
  incomplete. Explain what's unclear and what needs clarification.

Respond with ONLY valid JSON, no markdown fences, no preamble, as a JSON
array in this exact shape:
[{{"item": "<exact item text>", "status": "present"|"missing"|"ambiguous", "evidence": "<short quote>", "location": "<where found>", "note": "<clarification if needed>"}}]

Return exactly {len(checklist)} objects, one per checklist item, in the same order.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_completeness(drawing_color: np.ndarray, checklist: List[str] = None,
                        backend: str = "anthropic",
                        api_key: Optional[str] = None,
                        vision_model: Optional[str] = None) -> CompletenessResult:
    """
    Check a prototype drawing against the instruction checklist.

    Args:
        drawing_color: Drawing image (BGR).
        checklist: List of required instructions. Loaded from config if None.
        backend: "ollama" for local, "anthropic" for cloud.
        api_key: Required for Anthropic backend.
        vision_model: Override the default model.
    """
    if checklist is None:
        checklist = load_checklist()

    ai = AIBackend(
        backend=backend,
        api_key=api_key,
        vision_model=vision_model,
    )

    # Run OCR pre-check for context
    ocr_context = ""
    try:
        from core.ocr_engine import extract_text
        ocr_result = extract_text(drawing_color)
        ocr_context = ocr_result.full_text
    except Exception:
        pass

    prompt = _build_prompt(checklist, ocr_context)

    raw_response = ai.call_vision([drawing_color], prompt, max_tokens=2000)

    # Parse the response
    from core.local_llm import parse_json_response
    parsed = parse_json_response(raw_response)

    # Handle both list and dict responses
    if isinstance(parsed, dict) and "items" in parsed:
        parsed = parsed["items"]
    elif not isinstance(parsed, list):
        parsed = [parsed]

    items = []
    for i, entry in enumerate(parsed):
        items.append(ChecklistItemResult(
            item=entry.get("item", checklist[i] if i < len(checklist) else "?"),
            status=entry.get("status", "ambiguous"),
            evidence=entry.get("evidence", ""),
            note=entry.get("note", ""),
            location=entry.get("location", ""),
        ))

    # Fill in any checklist items that the LLM didn't return
    if len(items) < len(checklist):
        returned_items = {item.item.lower() for item in items}
        for ci in checklist:
            if ci.lower() not in returned_items:
                items.append(ChecklistItemResult(
                    item=ci, status="ambiguous",
                    evidence="", note="LLM did not evaluate this item",
                ))

    return CompletenessResult(items=items, ocr_text_found=ocr_context)
