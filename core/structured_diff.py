"""
Structured Diff — Text/Dimension-Level Comparison Between Drawings
====================================================================
After OCR extracts text blocks from both master and revision drawings,
this module performs structured comparison:

- Spatial matching: pairs text blocks by location (nearest-neighbor)
- Character-level diff: highlights exact changes (e.g., "25.4" → "25.5")
- Categories: "added", "removed", "modified", "moved", "unchanged"

This catches subtle changes that pixel-diffing misses — like a dimension
value changing from "25.4" to "25.5" where the pixels barely shift.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextChange:
    category: str           # "added" | "removed" | "modified" | "moved" | "unchanged"
    master_text: str        # text in master (empty if added)
    revision_text: str      # text in revision (empty if removed)
    master_bbox: Optional[Tuple[int, int, int, int]] = None
    revision_bbox: Optional[Tuple[int, int, int, int]] = None
    change_detail: str = ""  # human-readable description of the change
    severity: str = "info"   # "info" | "warning" | "error"

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "master_text": self.master_text,
            "revision_text": self.revision_text,
            "change_detail": self.change_detail,
            "severity": self.severity,
        }


@dataclass
class StructuredDiffResult:
    changes: List[TextChange] = field(default_factory=list)
    total_master_blocks: int = 0
    total_revision_blocks: int = 0

    @property
    def added(self) -> List[TextChange]:
        return [c for c in self.changes if c.category == "added"]

    @property
    def removed(self) -> List[TextChange]:
        return [c for c in self.changes if c.category == "removed"]

    @property
    def modified(self) -> List[TextChange]:
        return [c for c in self.changes if c.category == "modified"]

    @property
    def moved(self) -> List[TextChange]:
        return [c for c in self.changes if c.category == "moved"]

    @property
    def unchanged(self) -> List[TextChange]:
        return [c for c in self.changes if c.category == "unchanged"]

    def summary(self) -> dict:
        return {
            "total_master_blocks": self.total_master_blocks,
            "total_revision_blocks": self.total_revision_blocks,
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
            "moved": len(self.moved),
            "unchanged": len(self.unchanged),
        }

    def to_report(self) -> dict:
        return {
            "summary": self.summary(),
            "changes": [c.to_dict() for c in self.changes if c.category != "unchanged"],
        }


# ---------------------------------------------------------------------------
# Spatial matching
# ---------------------------------------------------------------------------

def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _distance(bbox1: Tuple[int, int, int, int],
              bbox2: Tuple[int, int, int, int]) -> float:
    c1 = _center(bbox1)
    c2 = _center(bbox2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, collapse whitespace)."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def _is_dimension(text: str) -> bool:
    """Check if text looks like a numeric dimension."""
    return bool(re.search(r'\d+[.,]?\d*\s*(?:mm|cm|in|°|"|±|Ø|R\d)?', text))


def _classify_severity(master_text: str, revision_text: str) -> str:
    """
    Classify the severity of a text change.
    Dimension changes are 'error' (could be a design mistake).
    Other text changes are 'warning'.
    """
    if _is_dimension(master_text) or _is_dimension(revision_text):
        # Check if the numeric values actually changed
        master_nums = re.findall(r'\d+[.,]?\d*', master_text)
        revision_nums = re.findall(r'\d+[.,]?\d*', revision_text)
        if master_nums != revision_nums:
            return "error"
    return "warning"


# ---------------------------------------------------------------------------
# Core diff algorithm
# ---------------------------------------------------------------------------

def compute_structured_diff(master_blocks, revision_blocks,
                             max_match_distance: float = 100.0,
                             text_similarity_threshold: float = 0.3) -> StructuredDiffResult:
    """
    Compare OCR text blocks from master and revision drawings.

    Algorithm:
    1. For each master block, find the nearest revision block by position.
    2. If the text matches exactly → "unchanged"
    3. If the text is similar but different → "modified"
    4. If the nearest block has very different text → master block is "removed"
       and revision block is "added"
    5. Unmatched revision blocks → "added"
    6. Unmatched master blocks → "removed"

    Args:
        master_blocks: List of TextBlock from OCR on the master drawing.
        revision_blocks: List of TextBlock from OCR on the revision.
        max_match_distance: Max pixel distance to consider two blocks as
                            potentially the same text element.
        text_similarity_threshold: Min ratio (0–1) of matching chars to
                                    consider texts as modified (vs. unrelated).
    """
    changes = []
    used_revision = set()

    for m_block in master_blocks:
        # Find nearest revision block by spatial position
        best_idx = None
        best_dist = float("inf")

        for j, r_block in enumerate(revision_blocks):
            if j in used_revision:
                continue
            dist = _distance(m_block.bbox, r_block.bbox)
            if dist < best_dist:
                best_dist = dist
                best_idx = j

        if best_idx is not None and best_dist <= max_match_distance:
            r_block = revision_blocks[best_idx]
            m_norm = _normalize_text(m_block.text)
            r_norm = _normalize_text(r_block.text)

            if m_norm == r_norm:
                changes.append(TextChange(
                    category="unchanged",
                    master_text=m_block.text,
                    revision_text=r_block.text,
                    master_bbox=m_block.bbox,
                    revision_bbox=r_block.bbox,
                ))
            else:
                # Check text similarity to distinguish "modified" from "unrelated"
                similarity = _text_similarity(m_norm, r_norm)

                if similarity >= text_similarity_threshold:
                    detail = _describe_change(m_block.text, r_block.text)
                    severity = _classify_severity(m_block.text, r_block.text)
                    changes.append(TextChange(
                        category="modified",
                        master_text=m_block.text,
                        revision_text=r_block.text,
                        master_bbox=m_block.bbox,
                        revision_bbox=r_block.bbox,
                        change_detail=detail,
                        severity=severity,
                    ))
                else:
                    # Too different — treat as remove + add
                    changes.append(TextChange(
                        category="removed",
                        master_text=m_block.text,
                        revision_text="",
                        master_bbox=m_block.bbox,
                        change_detail=f"Text removed: \"{m_block.text}\"",
                        severity="warning",
                    ))
                    changes.append(TextChange(
                        category="added",
                        master_text="",
                        revision_text=r_block.text,
                        revision_bbox=r_block.bbox,
                        change_detail=f"New text: \"{r_block.text}\"",
                        severity="warning",
                    ))

            used_revision.add(best_idx)
        else:
            # No nearby match — master block was removed
            changes.append(TextChange(
                category="removed",
                master_text=m_block.text,
                revision_text="",
                master_bbox=m_block.bbox,
                change_detail=f"Text removed: \"{m_block.text}\"",
                severity="warning",
            ))

    # Any revision blocks not matched to a master block are new additions
    for j, r_block in enumerate(revision_blocks):
        if j not in used_revision:
            changes.append(TextChange(
                category="added",
                master_text="",
                revision_text=r_block.text,
                revision_bbox=r_block.bbox,
                change_detail=f"New text: \"{r_block.text}\"",
                severity="warning",
            ))

    # Check for blocks that moved (same text, different position)
    _detect_moves(changes, max_match_distance)

    return StructuredDiffResult(
        changes=changes,
        total_master_blocks=len(master_blocks),
        total_revision_blocks=len(revision_blocks),
    )


def _text_similarity(s1: str, s2: str) -> float:
    """Simple character-level similarity ratio (Jaccard-ish)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Use longest common subsequence ratio
    lcs_len = _lcs_length(s1, s2)
    return (2.0 * lcs_len) / (len(s1) + len(s2))


def _lcs_length(s1: str, s2: str) -> int:
    """Length of the longest common subsequence."""
    m, n = len(s1), len(s2)
    # Space-optimized DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _describe_change(master_text: str, revision_text: str) -> str:
    """Generate a human-readable description of what changed."""
    # Find specific numeric changes
    m_nums = re.findall(r'\d+[.,]\d+|\d+', master_text)
    r_nums = re.findall(r'\d+[.,]\d+|\d+', revision_text)

    if m_nums and r_nums and m_nums != r_nums:
        changed_pairs = []
        for i, (m, r) in enumerate(zip(m_nums, r_nums)):
            if m != r:
                changed_pairs.append(f"{m} → {r}")
        if changed_pairs:
            return f"Value changed: {', '.join(changed_pairs)}"

    return f"\"{master_text}\" → \"{revision_text}\""


def _detect_moves(changes: List[TextChange], threshold: float):
    """
    Post-process: if a 'removed' text and an 'added' text have the same
    content but different positions, reclassify both as a single 'moved'.
    """
    removed = [(i, c) for i, c in enumerate(changes) if c.category == "removed"]
    added = [(i, c) for i, c in enumerate(changes) if c.category == "added"]

    to_remove_indices = set()

    for ri, rc in removed:
        r_norm = _normalize_text(rc.master_text)
        for ai, ac in added:
            if ai in to_remove_indices:
                continue
            a_norm = _normalize_text(ac.revision_text)
            if r_norm == a_norm and r_norm:
                # Same text, different position — it's a move
                rc.category = "moved"
                rc.revision_text = ac.revision_text
                rc.revision_bbox = ac.revision_bbox
                rc.change_detail = f"Text moved: \"{rc.master_text}\""
                rc.severity = "info"
                to_remove_indices.add(ai)
                break

    # Remove the 'added' entries that were reclassified as moves
    for idx in sorted(to_remove_indices, reverse=True):
        changes.pop(idx)
