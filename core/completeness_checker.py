"""
Prototype Instruction Completeness Checker
=============================================
For prototype requests: confirms whether the drawing is missing, or has
ambiguous/incorrect, required instructions -- e.g. base fabric type, sewing
method, panel positioning method.

This is a single-drawing information-extraction task, not a diff. The
checklist itself is domain knowledge that should live with whoever owns the
drawing standard -- it's exposed here as a plain editable list so it can be
tuned per drawing type without touching code.
"""
import base64
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

MODEL = "claude-sonnet-5"

DEFAULT_CHECKLIST = [
    "Base fabric type / material callout",
    "Sewing method (stitch type, seam allowance)",
    "Panel positioning method",
]


@dataclass
class ChecklistItemResult:
    item: str
    status: str            # "present" | "missing" | "ambiguous"
    evidence: str           # quoted/paraphrased text found, or "" if missing
    note: str = ""


@dataclass
class CompletenessResult:
    items: List[ChecklistItemResult] = field(default_factory=list)

    def to_dict(self):
        return {
            "items": [
                {"item": i.item, "status": i.status, "evidence": i.evidence, "note": i.note}
                for i in self.items
            ],
            "missing_count": sum(1 for i in self.items if i.status == "missing"),
            "ambiguous_count": sum(1 for i in self.items if i.status == "ambiguous"),
        }


def _to_b64_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _build_prompt(checklist: List[str]) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(checklist))
    return f"""You are reviewing an engineering/prototype drawing for a manufacturing
request. Check whether the drawing contains clear instructions for each of
the following required items:

{numbered}

For EACH item, look across the entire drawing (all views, notes, callouts,
title block) and decide:
- "present": the instruction is clearly given. Quote or closely paraphrase
  the actual text you found (keep it under 15 words).
- "missing": there is no instruction anywhere on the drawing for this item.
- "ambiguous": something is there but it's unclear, contradictory, or
  incomplete (e.g., a fabric callout with no material name, or a sewing note
  that doesn't specify stitch type). Explain what's unclear in "note".

Respond with ONLY valid JSON, no markdown fences, no preamble, as a JSON
array in this exact shape:
[{{"item": "<exact item text from the list above>", "status": "present"|"missing"|"ambiguous", "evidence": "<short quote or paraphrase, empty if missing>", "note": "<short clarification, empty if present>"}}]

Return exactly {len(checklist)} objects, one per checklist item, in the same order.
"""


def check_completeness(drawing_color: np.ndarray, checklist: List[str],
                        api_key: Optional[str] = None) -> CompletenessResult:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    img_b64 = _to_b64_png(drawing_color)
    prompt = _build_prompt(checklist)

    message = client.messages.create(
        model=MODEL,
        max_tokens=1200,
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
    items = [
        ChecklistItemResult(
            item=entry.get("item", checklist[i] if i < len(checklist) else "?"),
            status=entry.get("status", "ambiguous"),
            evidence=entry.get("evidence", ""),
            note=entry.get("note", ""),
        )
        for i, entry in enumerate(parsed)
    ]
    return CompletenessResult(items=items)
