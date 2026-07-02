"""
Change-Request Reconciliation
==============================
Extends the deterministic CV diff (comparator.py) with a semantic layer:
given the text of what was actually *requested* (an ECO note, an email to
CET, a change-order list), classify each detected pixel-diff region as:

  - "applied"    -> matches one of the requested changes
  - "unintended" -> a real geometric change, but not something anyone asked for
  - (and separately) any requested change with no matching region -> "missing"

This uses Claude's vision capability to look at a before/after crop of each
region alongside the requested-change list, because that judgment call
("does this region correspond to 'move the top-right bolt hole 15mm right'")
needs semantic understanding, not just pixel math.

Requires an Anthropic API key (ANTHROPIC_API_KEY env var, or passed in).
"""
import base64
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

from core.comparator import Discrepancy

MODEL = "claude-sonnet-5"


@dataclass
class RegionVerdict:
    discrepancy_id: int
    matched_request_index: Optional[int]  # index into requested_changes, or None
    verdict: str                          # "applied" | "unintended"
    explanation: str


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
                 "explanation": v.explanation}
                for v in self.region_verdicts if v.verdict == "applied"
            ],
            "unintended": [
                {"discrepancy_id": v.discrepancy_id, "explanation": v.explanation}
                for v in self.region_verdicts if v.verdict == "unintended"
            ],
            "missing": self.missing,
        }


def _crop(img: np.ndarray, bbox, pad: int = 30) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return img[y0:y1, x0:x1]


def _to_b64_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode crop as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _build_prompt(requested_changes: List[str]) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(requested_changes))
    return f"""You are reviewing a mechanical/engineering drawing revision for a supplier
quality check. You are shown two crops of the SAME region of a drawing: the
BEFORE (master/original) version and the AFTER (revised) version, taken from
the same coordinates. A pixel-diff algorithm already flagged this region as
changed.

Here is the list of changes that were formally REQUESTED for this drawing:
{numbered}

Look at the before/after crop and decide:
- Does this specific change correspond to ONE of the requested changes above?
  If yes, respond with that item's number.
- If the change is real but does NOT correspond to any requested item, it's
  an unintended/unrequested change.

Respond with ONLY valid JSON, no markdown fences, no preamble, in this exact
shape:
{{"matched_request_index": <int or null>, "verdict": "applied" or "unintended", "explanation": "<one sentence>"}}
"""


def _call_claude(client, master_crop_b64: str, revision_crop_b64: str, prompt: str) -> dict:
    message = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "BEFORE (master):"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": master_crop_b64}},
                    {"type": "text", "text": "AFTER (revision):"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": revision_crop_b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def reconcile_changes(master_color: np.ndarray, aligned_revision_color: np.ndarray,
                       discrepancies: List[Discrepancy], requested_changes: List[str],
                       api_key: Optional[str] = None) -> ReconciliationResult:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    prompt = _build_prompt(requested_changes)
    verdicts: List[RegionVerdict] = []
    matched_indices = set()

    for d in discrepancies:
        master_crop = _crop(master_color, d.bbox)
        revision_crop = _crop(aligned_revision_color, d.bbox)
        if master_crop.size == 0 or revision_crop.size == 0:
            continue
        master_b64 = _to_b64_png(master_crop)
        revision_b64 = _to_b64_png(revision_crop)
        try:
            result = _call_claude(client, master_b64, revision_b64, prompt)
        except Exception as e:
            verdicts.append(RegionVerdict(
                discrepancy_id=d.id, matched_request_index=None,
                verdict="unintended", explanation=f"(AI call failed: {e})",
            ))
            continue

        idx = result.get("matched_request_index")
        verdict = result.get("verdict", "unintended")
        explanation = result.get("explanation", "")
        verdicts.append(RegionVerdict(
            discrepancy_id=d.id, matched_request_index=idx,
            verdict=verdict, explanation=explanation,
        ))
        if verdict == "applied" and idx is not None:
            matched_indices.add(idx)

    missing = [c for i, c in enumerate(requested_changes) if i not in matched_indices]

    return ReconciliationResult(
        requested_changes=requested_changes,
        region_verdicts=verdicts,
        missing=missing,
    )
