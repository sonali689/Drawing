"""
Layout Analyzer
================
Identifies and parses engineering drawing layout structures:
1. Title Block extraction (metadata like part number, revision, title, scale).
2. Bill of Materials (BOM) extraction.

Uses a hybrid approach:
- Local heuristics/regex to extract text blocks in the bottom-right (Title Block)
  and top-right/right-side (BOM).
- Semantic Vision-Language Model (VLM) fallback/enhancement to accurately structure
  and parse tables from image crops.
"""
import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.ocr_engine import TextBlock
from core.local_llm import AIBackend, parse_json_response


def get_title_block_bbox(img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Get the bottom-right bounding box where the title block is typically located."""
    h, w = img_shape[:2]
    # Bottom 25% height, right 35% width
    x = int(0.65 * w)
    y = int(0.70 * h)
    return (x, y, w - x, h - y)


def get_bom_bbox(img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Get the bounding box where the Bill of Materials (BOM) is typically located."""
    h, w = img_shape[:2]
    # Top-right/middle-right region: right 35% width, top 70% height (above title block)
    x = int(0.65 * w)
    y = int(0.05 * h)
    return (x, y, w - x, int(0.65 * h))


def parse_title_block_heuristics(text_blocks: List[TextBlock], img_shape: Tuple[int, int]) -> dict:
    """
    Extract title block metadata locally using regex and coordinate heuristics.
    """
    tx, ty, tw, th = get_title_block_bbox(img_shape)
    
    # Filter text blocks in the title block area
    tb_blocks = []
    for b in text_blocks:
        bx, by, bw, bh = b.bbox
        # Check center point
        cx, cy = bx + bw // 2, by + bh // 2
        if tx <= cx <= tx + tw and ty <= cy <= ty + th:
            tb_blocks.append(b)
            
    # Sort top-to-bottom, left-to-right
    tb_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    
    metadata = {
        "part_number": None,
        "revision": None,
        "title": None,
        "scale": None,
        "units": None,
        "designer": None
    }
    
    # Simple regex patterns
    rev_pattern = re.compile(r'\b(?:rev|revision|level)[:\-\s]*([a-zA-Z0-9]{1,2})\b', re.IGNORECASE)
    scale_pattern = re.compile(r'\b(?:scale)[:\-\s]*([1-9]\d*\s*:\s*[1-9]\d*)\b', re.IGNORECASE)
    unit_pattern = re.compile(r'\b(mm|inch|inches|metric|inch-lbs)\b', re.IGNORECASE)
    # General part number patterns: e.g. ALV-123456 or CET-9988-A
    part_pattern = re.compile(r'\b([a-zA-Z]{2,4}[-_\s]\d{4,}[-_\s]?[a-zA-Z0-9]?)\b')
    
    all_texts = [b.text for b in tb_blocks]
    full_text_tb = " \n ".join(all_texts)
    
    # Apply regex searches
    rev_match = rev_pattern.search(full_text_tb)
    if rev_match:
        metadata["revision"] = rev_match.group(1).upper()
        
    scale_match = scale_pattern.search(full_text_tb)
    if scale_match:
        metadata["scale"] = scale_match.group(1).replace(" ", "")
        
    unit_match = unit_pattern.search(full_text_tb)
    if unit_match:
        metadata["units"] = unit_match.group(1).lower()
        
    part_match = part_pattern.search(full_text_tb)
    if part_match:
        metadata["part_number"] = part_match.group(1).upper()
        
    # Attempt to extract title (heuristically find capitalized multi-word phrases)
    potential_titles = []
    for text in all_texts:
        clean = text.strip()
        if len(clean) > 5 and clean.isupper() and not any(k in clean.lower() for k in ["rev", "scale", "date", "sheet", "dwg", "alv", "cet"]):
            # Avoid matching part numbers
            if not part_pattern.search(clean) and not re.search(r'\d{3,}', clean):
                potential_titles.append(clean)
                
    if potential_titles:
        # Keep the longest or first capitalized multi-word line
        metadata["title"] = " ".join(potential_titles[:2])
        
    return metadata


def parse_title_block_vlm(img: np.ndarray,
                           backend: str = "anthropic",
                           api_key: Optional[str] = None,
                           vision_model: Optional[str] = None) -> dict:
    """
    Crop the title block area and analyze it with a vision model for exact structured metadata.
    """
    tx, ty, tw, th = get_title_block_bbox(img.shape)
    title_block_crop = img[ty:ty+th, tx:tx+tw]
    
    ai = AIBackend(backend=backend, api_key=api_key, vision_model=vision_model)
    
    prompt = """You are an expert engineering drawing metadata extractor.
Analyze the provided crop of a drawing's title block (typically the bottom-right info panel).
Extract the following fields and return them as a clean JSON object. 

Required fields:
- "part_number": drawing number or part code (e.g. 6234567-A, ALV-8991)
- "revision": revision code or level (e.g. A, B, 0, 1, 2)
- "title": drawing description or part title (e.g. DRIVER AIRBAG CUSHION)
- "scale": sheet scale (e.g. 1:1, 1:2)
- "units": measurement units (e.g. mm, inches)
- "designer": designer or approver name

Return format:
{
  "part_number": "extracted value or null",
  "revision": "extracted value or null",
  "title": "extracted value or null",
  "scale": "extracted value or null",
  "units": "extracted value or null",
  "designer": "extracted value or null"
}

Do not write markdown formatting or explanations other than the JSON object. Keep the output clean.
"""
    try:
        raw_response = ai.call_vision([title_block_crop], prompt, max_tokens=1000)
        parsed = parse_json_response(raw_response)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        print(f"VLM Title Block parse failed: {e}")
    return {}


def parse_bom_heuristics(text_blocks: List[TextBlock], img_shape: Tuple[int, int]) -> List[dict]:
    """
    Extract Bill of Materials (BOM) items locally using coordinates.
    Looks for tabular columns in the BOM region.
    """
    bx, by, bw, bh = get_bom_bbox(img_shape)
    
    # Filter text blocks in BOM area
    bom_blocks = []
    for b in text_blocks:
        x, y, w, h = b.bbox
        cx, cy = x + w // 2, y + h // 2
        if bx <= cx <= bx + bw and by <= cy <= by + bh:
            bom_blocks.append(b)
            
    if not bom_blocks:
        return []
        
    # Group text blocks by rows (similar y-coordinate within tolerance)
    row_tolerance = 15
    rows_grouped = []
    
    # Sort blocks top-to-bottom
    bom_blocks.sort(key=lambda b: b.bbox[1])
    
    for block in bom_blocks:
        placed = False
        for r in rows_grouped:
            # Check if this block fits in an existing row based on center-y
            row_y_mean = sum(b.bbox[1] for b in r) / len(r)
            if abs(block.bbox[1] - row_y_mean) < row_tolerance:
                r.append(block)
                placed = True
                break
        if not placed:
            rows_grouped.append([block])
            
    bom_items = []
    for r in rows_grouped:
        # Sort items in each row left-to-right
        r.sort(key=lambda b: b.bbox[0])
        row_text = [b.text for b in r]
        
        # Check if the row contains an item number (typically first element is numeric)
        if row_text:
            first_val = row_text[0].strip()
            # If it looks like an item index (e.g., 1, 2, 01, etc.)
            if re.match(r'^\d{1,2}$', first_val):
                bom_items.append({
                    "item": first_val,
                    "raw_row": " | ".join(row_text)
                })
                
    return bom_items


def parse_bom_vlm(img: np.ndarray,
                  backend: str = "anthropic",
                  api_key: Optional[str] = None,
                  vision_model: Optional[str] = None) -> List[dict]:
    """
    Crop the BOM area and analyze it with a vision model to extract structured table rows.
    """
    bx, by, bw, bh = get_bom_bbox(img.shape)
    bom_crop = img[by:by+bh, bx:bx+bw]
    
    ai = AIBackend(backend=backend, api_key=api_key, vision_model=vision_model)
    
    prompt = """You are an expert engineering drawing inspector.
Analyze this crop of a Bill of Materials (BOM) or parts list table.
Extract the rows of this table and return them as a JSON list.

Each item in the list must represent a row in the table, structured as follows:
{
  "item": "item index/number (e.g. 1, 2)",
  "part_number": "part number or code (if present)",
  "description": "description/name of component (e.g. BASE FABRIC, TETHER)",
  "qty": "quantity (e.g. 1, 2, 0.5m)",
  "material": "material details (if present)",
  "remarks": "remarks or comments (if present)"
}

Format the response as:
{
  "bom": [
    ...rows...
  ]
}

Only return the JSON. If a column is missing in the table, set its value to null.
"""
    try:
        raw_response = ai.call_vision([bom_crop], prompt, max_tokens=1500)
        parsed = parse_json_response(raw_response)
        if isinstance(parsed, dict) and "bom" in parsed:
            return parsed["bom"]
    except Exception as e:
        print(f"VLM BOM parse failed: {e}")
    return []
