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
                             max_match_distance: float = 200.0,
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

    # Check for components that moved using relative anchor offsets
    _match_moved_components_by_anchor(changes)

    # Check for blocks that moved (same text, different position)
    _detect_moves(changes, max_match_distance)

    # Content-first matching: catch relocated components that spatial matching missed
    _match_by_content(changes, text_similarity_threshold)

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

    # Check for cushion drawing specific changes
    m_lower = master_text.lower()
    r_lower = revision_text.lower()
    
    # Material changes
    material_terms = ['nylon', 'polyester', 'fabric', 'cloth', 'thread', 'yarn',
                      'silicone', 'coating', 'pa66', 'pa6', 'pet', 'opa', 'material',
                      'denier', 'dtex', 'tex']
    if any(t in m_lower or t in r_lower for t in material_terms):
        return f"Material change: \"{master_text}\" → \"{revision_text}\""
    
    # Sewing / stitch changes
    sewing_terms = ['stitch', 'sew', 'seam', 'hem', 'tack', 'bartack', 'zigzag',
                    'lockstitch', 'chainstitch', 'binding', 'fold', 'pattern',
                    'needle', 'bobbin', 'spi']
    if any(t in m_lower or t in r_lower for t in sewing_terms):
        return f"Sewing change: \"{master_text}\" → \"{revision_text}\""
    
    # Revision / drawing metadata changes
    meta_terms = ['rev', 'revision', 'date', 'drawn', 'checked', 'approved',
                  'issue', 'ecn', 'eco', 'dcn', 'released']
    if any(t in m_lower or t in r_lower for t in meta_terms):
        return f"Revision metadata change: \"{master_text}\" → \"{revision_text}\""
    
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


def _match_moved_components_by_anchor(changes: List[TextChange], max_anchor_distance: float = 400.0, match_tolerance: float = 60.0):
    """
    Find components that moved globally on the drawing sheet.
    
    1. Identify 'removed' and 'added' blocks that are anchor texts (non-numeric, length > 4).
    2. Pair matching anchors (similarity >= 0.8) which are far apart.
    3. For each anchor pair, calculate translation vector (dx, dy).
    4. For all other 'removed' blocks within max_anchor_distance of the master anchor,
       translate their position by (dx, dy).
    5. Search for 'added' blocks in the revision close to the translated position (within match_tolerance).
    6. Reclassify them as 'moved' or 'modified' (with a 'moved' flag in change_detail).
    """
    # 1. Gather unmatched (removed/added) items
    removed_items = [c for c in changes if c.category == "removed"]
    added_items = [c for c in changes if c.category == "added"]
    
    # Heuristic to check if a block is an anchor (text label)
    def is_anchor(text: str) -> bool:
        if not text:
            return False
        # Must contain letters, not just numbers or dimensions
        if not any(char.isalpha() for char in text):
            return False
        # Exclude short strings or things that look like dimensions
        if len(text.strip()) < 5:
            return False
        if _is_dimension(text):
            return False
        return True

    # Find anchor candidates
    master_anchors = [c for c in removed_items if is_anchor(c.master_text)]
    revision_anchors = [c for c in added_items if is_anchor(c.revision_text)]
    
    paired_anchors = []
    used_added_anchors = set()
    
    for m_anch in master_anchors:
        best_r_anch = None
        best_sim = 0.0
        for r_anch in revision_anchors:
            if id(r_anch) in used_added_anchors:
                continue
            sim = _text_similarity(_normalize_text(m_anch.master_text), _normalize_text(r_anch.revision_text))
            if sim >= 0.8 and sim > best_sim:
                best_sim = sim
                best_r_anch = r_anch
        
        if best_r_anch is not None:
            paired_anchors.append((m_anch, best_r_anch))
            used_added_anchors.add(id(best_r_anch))
            
    # Process translation for each anchor pair
    to_remove_added = set()
    
    for m_anch, r_anch in paired_anchors:
        # Calculate displacement vector (dx, dy)
        mx, my = _center(m_anch.master_bbox)
        rx, ry = _center(r_anch.revision_bbox)
        dx = rx - mx
        dy = ry - my
        
        # Reclassify the anchor itself as moved or modified
        m_anch.category = "moved"
        m_anch.revision_text = r_anch.revision_text
        m_anch.revision_bbox = r_anch.revision_bbox
        m_anch.change_detail = f"Anchor moved: \"{m_anch.master_text}\""
        m_anch.severity = "info"
        to_remove_added.add(id(r_anch))
        
        # Find other removed blocks near the master anchor
        for rc in removed_items:
            if rc.category != "removed":  # might have been reclassified already
                continue
            dist_to_anchor = _distance(rc.master_bbox, m_anch.master_bbox)
            if dist_to_anchor > max_anchor_distance:
                continue
                
            # Expected position of this block in revision
            mcx, mcy = _center(rc.master_bbox)
            expected_rx = mcx + dx
            expected_ry = mcy + dy
            
            # Find the closest added block in revision to the expected position
            best_ac = None
            best_ac_dist = float("inf")
            for ac in added_items:
                if id(ac) in to_remove_added:
                    continue
                acx, acy = _center(ac.revision_bbox)
                dist = ((acx - expected_rx) ** 2 + (acy - expected_ry) ** 2) ** 0.5
                if dist < best_ac_dist:
                    best_ac_dist = dist
                    best_ac = ac
            
            if best_ac is not None and best_ac_dist <= match_tolerance:
                # Match found! Decide if it's a move (same text) or modification
                m_norm = _normalize_text(rc.master_text)
                r_norm = _normalize_text(best_ac.revision_text)
                
                rc.revision_text = best_ac.revision_text
                rc.revision_bbox = best_ac.revision_bbox
                to_remove_added.add(id(best_ac))
                
                if m_norm == r_norm:
                    rc.category = "moved"
                    rc.change_detail = f"Text moved: \"{rc.master_text}\""
                    rc.severity = "info"
                else:
                    rc.category = "modified"
                    rc.change_detail = f"Value/text changed: \"{rc.master_text}\" → \"{best_ac.revision_text}\" (moved)"
                    rc.severity = _classify_severity(rc.master_text, best_ac.revision_text)
                    
    # Clean up the reclassified 'added' entries from changes list
    changes[:] = [c for c in changes if id(c) not in to_remove_added]


def _match_by_content(changes: List[TextChange], similarity_threshold: float = 0.3):
    """
    Content-first matching: pair remaining 'removed' and 'added' entries by
    text content similarity, regardless of spatial position.

    This catches cases where a component drawing (e.g., a section detail view)
    is repositioned to a completely different area of the sheet between revisions.
    Spatial matching fails for these, but the text content is the same or similar.
    """
    removed = [(i, c) for i, c in enumerate(changes) if c.category == "removed"]
    added = [(i, c) for i, c in enumerate(changes) if c.category == "added"]

    to_remove_indices = set()

    for ri, rc in removed:
        if ri in to_remove_indices:
            continue
        r_norm = _normalize_text(rc.master_text)
        if not r_norm or len(r_norm) < 3:
            continue

        best_ai = None
        best_sim = 0.0

        for ai, ac in added:
            if ai in to_remove_indices:
                continue
            a_norm = _normalize_text(ac.revision_text)
            if not a_norm:
                continue
            sim = _text_similarity(r_norm, a_norm)
            if sim > best_sim:
                best_sim = sim
                best_ai = (ai, ac)

        if best_ai is not None and best_sim >= 0.6:
            ai, ac = best_ai
            if best_sim >= 0.95:
                # Nearly identical text — it's a move
                rc.category = "moved"
                rc.revision_text = ac.revision_text
                rc.revision_bbox = ac.revision_bbox
                rc.change_detail = f"Component relocated: \"{rc.master_text}\""
                rc.severity = "info"
            else:
                # Similar but changed text at a different location
                rc.category = "modified"
                rc.revision_text = ac.revision_text
                rc.revision_bbox = ac.revision_bbox
                rc.change_detail = _describe_change(rc.master_text, ac.revision_text) + " (relocated)"
                rc.severity = _classify_severity(rc.master_text, ac.revision_text)
            to_remove_indices.add(ai)

    # Remove matched 'added' entries
    for idx in sorted(to_remove_indices, reverse=True):
        changes.pop(idx)

