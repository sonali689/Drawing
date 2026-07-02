"""
Text Analyzer
==============
Local typo detection and dimension parsing for engineering drawing text.
Works entirely offline using pyspellchecker + custom domain dictionaries.

No LLM required — this is pure rule-based analysis that runs instantly.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import yaml

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TypoCandidate:
    original: str              # the word as found
    suggested: str             # closest correct word
    context: str               # surrounding text for reference
    location_bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0    # how likely this is a real typo (0–1)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "suggested": self.suggested,
            "context": self.context,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class ParsedDimension:
    raw_text: str               # e.g., "Ø25.4±0.1"
    value: float                # 25.4
    unit: str                   # "mm", "in", etc.
    tolerance_plus: Optional[float] = None
    tolerance_minus: Optional[float] = None
    prefix: str = ""            # "Ø", "R", "M", etc.
    bbox: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw_text,
            "value": self.value,
            "unit": self.unit,
            "prefix": self.prefix,
            "tolerance": f"+{self.tolerance_plus}/-{self.tolerance_minus}"
            if self.tolerance_plus is not None else None,
        }


@dataclass
class DimensionMismatch:
    subject: str
    location_a: str
    value_a: str
    location_b: str
    value_b: str
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


# ---------------------------------------------------------------------------
# Domain dictionary loader
# ---------------------------------------------------------------------------

def load_domain_terms(config_path: str = None) -> Set[str]:
    """
    Load domain terms from the YAML config file.
    Returns a set of lowercase terms that should NOT be flagged as typos.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "domain_terms.yaml"
        )

    terms = set()

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data:
            for category in data.values():
                if isinstance(category, list):
                    for term in category:
                        terms.add(term.lower())
                        # Also add without hyphens and with common variations
                        terms.add(term.lower().replace("-", ""))
                        terms.add(term.lower().replace(" ", ""))

    # Always add common engineering abbreviations
    engineering_abbrevs = {
        "typ", "ref", "nts", "cl", "max", "min", "nom", "qty",
        "dia", "thru", "dp", "eq", "sp", "rad", "sq", "hex",
        "csk", "cbore", "tol", "ga", "awg", "pc", "pcs",
        "assy", "asm", "mtl", "mat", "spec", "dwg", "rev",
        "sht", "chk", "apd", "drn", "engr", "mfg",
    }
    terms.update(engineering_abbrevs)

    return terms


# ---------------------------------------------------------------------------
# Typo detection
# ---------------------------------------------------------------------------

# Pattern for things that are NOT regular words and should never be spell-checked
_SKIP_PATTERNS = [
    re.compile(r'^\d+[.,]?\d*$'),                     # pure numbers
    re.compile(r'^[A-Z]{1,2}-[A-Z]{1,2}$'),           # section labels: A-A, B-B
    re.compile(r'^\d+[.,]\d+\s*[±+\-]'),              # dimensions with tolerances
    re.compile(r'^[ØRM]\d'),                           # dimension prefixes
    re.compile(r'^[A-Z0-9]{2,}-\d+'),                 # part numbers
    re.compile(r'^\d+x\d+', re.IGNORECASE),           # multiplied dims: 2x5
    re.compile(r'^[\d.]+\s*(?:mm|cm|m|in|°|deg)$', re.IGNORECASE),  # dims with units
    re.compile(r'^[A-Z]{1,3}\d{3,}'),                 # codes like ALV1234
    re.compile(r'^[^a-zA-Z]*$'),                      # no letters at all
]


def _should_skip_word(word: str) -> bool:
    """Return True if this word should not be spell-checked."""
    if len(word) <= 1:
        return True
    for pattern in _SKIP_PATTERNS:
        if pattern.match(word):
            return True
    return False


def find_typos(text_blocks, domain_terms: Set[str] = None,
               min_confidence: float = 0.5) -> List[TypoCandidate]:
    """
    Find likely typos in OCR-extracted text blocks.

    Args:
        text_blocks: List of TextBlock objects from ocr_engine.
        domain_terms: Set of lowercase domain terms to exclude.
        min_confidence: Minimum OCR confidence to consider a block
                        (below this, it's more likely an OCR error than a typo).

    Returns:
        List of TypoCandidate objects.
    """
    if SpellChecker is None:
        raise ImportError(
            "pyspellchecker is required for typo detection. "
            "Install with: pip install pyspellchecker"
        )

    if domain_terms is None:
        domain_terms = load_domain_terms()

    spell = SpellChecker()
    # Add domain terms to the spell checker's dictionary
    spell.word_frequency.load_words(list(domain_terms))

    typos = []

    for block in text_blocks:
        if block.confidence < min_confidence:
            continue

        # Split into words, keeping track of context
        words = re.findall(r'[a-zA-Z]+(?:[-\'][a-zA-Z]+)*', block.text)

        for word in words:
            if _should_skip_word(word):
                continue
            if word.lower() in domain_terms:
                continue

            # Check spelling
            if spell.unknown([word.lower()]):
                correction = spell.correction(word.lower())
                if correction and correction != word.lower():
                    # Estimate confidence that this is a real typo vs OCR noise
                    edit_distance = _levenshtein(word.lower(), correction)
                    conf = max(0.0, 1.0 - (edit_distance - 1) * 0.3)
                    conf *= block.confidence  # weight by OCR confidence

                    typos.append(TypoCandidate(
                        original=word,
                        suggested=correction,
                        context=block.text,
                        location_bbox=block.bbox,
                        confidence=conf,
                    ))

    # Sort by confidence descending — most likely real typos first
    typos.sort(key=lambda t: t.confidence, reverse=True)
    return typos


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ---------------------------------------------------------------------------
# Dimension parsing
# ---------------------------------------------------------------------------

_DIM_PATTERN = re.compile(
    r'(?P<prefix>[ØRM]?)'                          # optional prefix
    r'(?P<value>\d+[.,]\d+|\d+)'                   # numeric value
    r'(?:\s*[±]\s*(?P<tol>\d+[.,]?\d*))?'           # optional ± tolerance
    r'(?:\s*(?P<unit>mm|cm|m|in|"|°|deg))?',        # optional unit
    re.IGNORECASE
)


def parse_dimensions(text_blocks) -> List[ParsedDimension]:
    """
    Extract and parse numeric dimensions from OCR text blocks.
    Returns structured dimension data with values, units, tolerances.
    """
    dimensions = []

    for block in text_blocks:
        for match in _DIM_PATTERN.finditer(block.text):
            value_str = match.group("value").replace(",", ".")
            try:
                value = float(value_str)
            except ValueError:
                continue

            prefix = match.group("prefix") or ""
            unit = match.group("unit") or ""
            tol_str = match.group("tol")

            tol = None
            if tol_str:
                try:
                    tol = float(tol_str.replace(",", "."))
                except ValueError:
                    tol = None

            dimensions.append(ParsedDimension(
                raw_text=match.group(0).strip(),
                value=value,
                unit=unit,
                prefix=prefix,
                tolerance_plus=tol,
                tolerance_minus=tol,
                bbox=block.bbox,
            ))

    return dimensions


# ---------------------------------------------------------------------------
# Cross-reference checking
# ---------------------------------------------------------------------------

def find_dimension_mismatches(dims_a: List[ParsedDimension],
                               dims_b: List[ParsedDimension],
                               label_a: str = "View A",
                               label_b: str = "View B",
                               spatial_tolerance_px: int = 50) -> List[DimensionMismatch]:
    """
    Find cases where the same dimension appears in two sets (e.g., two views)
    with different values. Matches by spatial proximity of bounding boxes.
    """
    mismatches = []
    used_b = set()

    for da in dims_a:
        if da.bbox is None:
            continue
        ca = (da.bbox[0] + da.bbox[2] // 2, da.bbox[1] + da.bbox[3] // 2)

        best_match = None
        best_dist = float("inf")

        for j, db in enumerate(dims_b):
            if j in used_b or db.bbox is None:
                continue
            if da.prefix != db.prefix:
                continue

            cb = (db.bbox[0] + db.bbox[2] // 2, db.bbox[1] + db.bbox[3] // 2)
            dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_match = (j, db)

        if best_match is not None:
            j, db = best_match
            if abs(da.value - db.value) > 0.001:
                mismatches.append(DimensionMismatch(
                    subject=f"{da.prefix}{da.value}{da.unit} vs {db.prefix}{db.value}{db.unit}",
                    location_a=label_a,
                    value_a=da.raw_text,
                    location_b=label_b,
                    value_b=db.raw_text,
                    note=f"Value difference: {abs(da.value - db.value):.3f}",
                ))
                used_b.add(j)

    return mismatches
