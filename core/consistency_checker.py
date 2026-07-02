"""
Typo & Cross-View Consistency Checker
========================================
Checks for:
  (a) typos / spelling mistakes in drawing text (notes, callouts, labels)
  (b) mismatches between the main drawing and its views/sections -- same
      panel/feature described or dimensioned differently in different places

Domain terms (fabric names, part-specific jargon, abbreviations) are passed
in so real terminology isn't flagged as a "typo" -- a generic spellchecker
would misfire constantly on engineering shorthand, so this relies on the
vision-LLM's judgment plus an explicit allow-list.
"""
import base64
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

MODEL = "claude-sonnet-5"

DEFAULT_DOMAIN_TERMS = [
    "Nylon 420D", "PU coated", "Pantone", "lock-stitch", "CET", "ALV",
]


@dataclass
class TypoFinding:
    location: str      # e.g. "Section A-A note"
    found_text: str
    suggested_fix: str


@dataclass
class MismatchFinding:
    subject: str         # e.g. "Side panel width"
    location_a: str
    value_a: str
    location_b: str
    value_b: str
    note: str = ""


@dataclass
class ConsistencyResult:
    typos: List[TypoFinding] = field(default_factory=list)
    mismatches: List[MismatchFinding] = field(default_factory=list)

    def to_dict(self):
        return {
            "typos": [t.__dict__ for t in self.typos],
            "mismatches": [m.__dict__ for m in self.mismatches],
        }


def _to_b64_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _build_prompt(domain_terms: List[str]) -> str:
    terms = ", ".join(domain_terms) if domain_terms else "(none provided)"
    return f"""You are proofreading an engineering/sewing drawing sheet that may contain
multiple views (main view, sections, details).

Do NOT flag these as errors -- they are known-correct domain terms/jargon:
{terms}

Do two things:

1. TYPOS: Find genuine spelling mistakes or typos in the drawing's text
   (notes, callouts, labels, dimensions text) -- things like "seem" instead
   of "seam", transposed letters, obviously wrong words. Do not flag
   abbreviations, part numbers, or the domain terms listed above.

2. CROSS-VIEW MISMATCHES: Find cases where the SAME panel, feature, or
   dimension is described differently in different parts of the drawing --
   e.g. a panel width given as one value in the main view and a different
   value in a section view, or a shape/method described inconsistently
   between views. Only flag things that refer to the same real-world
   feature; don't flag two different features that happen to look similar.

Respond with ONLY valid JSON, no markdown fences, no preamble, in this exact
shape:
{{
  "typos": [{{"location": "<where on the sheet>", "found_text": "<the misspelled text>", "suggested_fix": "<corrected text>"}}],
  "mismatches": [{{"subject": "<what feature/dimension>", "location_a": "<where>", "value_a": "<value/description there>", "location_b": "<where>", "value_b": "<value/description there>", "note": "<why this looks like a real mismatch>"}}]
}}

If there are no typos or no mismatches, return empty arrays for those keys. Do not invent issues that aren't there.
"""


def check_consistency(drawing_color: np.ndarray, domain_terms: List[str] = None,
                       api_key: Optional[str] = None) -> ConsistencyResult:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    domain_terms = domain_terms or DEFAULT_DOMAIN_TERMS
    img_b64 = _to_b64_png(drawing_color)
    prompt = _build_prompt(domain_terms)

    message = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]

    parsed = json.loads(text)
    typos = [TypoFinding(**t) for t in parsed.get("typos", [])]
    mismatches = [MismatchFinding(**m) for m in parsed.get("mismatches", [])]
    return ConsistencyResult(typos=typos, mismatches=mismatches)
