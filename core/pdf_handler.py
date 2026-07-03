"""
PDF Handler
============
Converts PDF pages to high-resolution images for the drawing analysis pipeline.
Also extracts embedded text from searchable PDFs for better accuracy than pure OCR.

Uses PyMuPDF (fitz) which is fast and does not require external system dependencies.
"""
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


@dataclass
class PDFPage:
    page_number: int          # 1-indexed
    image: np.ndarray         # BGR OpenCV image
    width_px: int
    height_px: int
    embedded_text: str        # text layer extracted directly from PDF (may be empty)


def is_pdf(file_bytes: bytes) -> bool:
    """Check if the given bytes look like a PDF."""
    return file_bytes[:5] == b"%PDF-"


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300,
                  pages: Optional[List[int]] = None) -> List[PDFPage]:
    """
    Render PDF pages to high-resolution BGR images.

    Args:
        pdf_bytes: Raw PDF file content.
        dpi: Rendering resolution. 300 is good for OCR; 150 is faster for preview.
        pages: 1-indexed page numbers to render. None = all pages.

    Returns:
        List of PDFPage objects with image data and embedded text.
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF is required for PDF support. Install with: pip install PyMuPDF"
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []

    page_indices = range(len(doc))
    if pages is not None:
        page_indices = [p - 1 for p in pages if 0 < p <= len(doc)]

    for idx in page_indices:
        page = doc[idx]

        # Render to image at the specified DPI
        # fitz default is 72 DPI, so scale factor = dpi / 72
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert pixmap to numpy array (RGB)
        img_data = np.frombuffer(pixmap.samples, dtype=np.uint8)
        img_rgb = img_data.reshape(pixmap.height, pixmap.width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Extract embedded text (from the PDF text layer, not OCR)
        embedded_text = page.get_text("text").strip()

        results.append(PDFPage(
            page_number=idx + 1,
            image=img_bgr,
            width_px=pixmap.width,
            height_px=pixmap.height,
            embedded_text=embedded_text,
        ))

    doc.close()
    return results


def get_page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages in a PDF."""
    if fitz is None:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


def load_drawing_file(file_obj) -> Tuple[List[np.ndarray], str]:
    """
    Universal loader: accepts a Streamlit UploadedFile (PDF, PNG, JPG, TIFF).
    Returns (list_of_bgr_images, file_type).

    For images, returns a single-element list.
    For PDFs, returns one image per page.
    """
    from PIL import Image

    raw = file_obj.read()
    file_obj.seek(0)

    if is_pdf(raw):
        pages = pdf_to_images(raw)
        images = [p.image for p in pages]
        return images, "pdf"
    else:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return [bgr], "image"


def extract_vector_text(pdf_bytes: bytes, page_number: int, dpi: int = 300) -> List:
    """
    Extract text search blocks from vector PDF directly, scaling their bounding boxes
    to match the high-resolution rendered image coordinates.
    """
    if fitz is None:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_number < 1 or page_number > len(doc):
        doc.close()
        return []

    page = doc[page_number - 1]
    
    # Scale factor from PDF points (72 DPI) to output image DPI
    scale = dpi / 72.0
    
    # Get text blocks
    # Each block is a tuple: (x0, y0, x1, y1, "text", block_no, block_type)
    blocks_raw = page.get_text("blocks")
    
    from core.ocr_engine import TextBlock
    
    text_blocks = []
    for b in blocks_raw:
        x0, y0, x1, y1, text, block_no, block_type = b
        text = text.strip()
        if not text:
            continue
            
        # Scale bounding box to image coordinate space
        x = int(x0 * scale)
        y = int(y0 * scale)
        w = int((x1 - x0) * scale)
        h = int((y1 - y0) * scale)
        
        # We assign confidence=1.0 for vector text
        text_blocks.append(TextBlock(
            text=text,
            bbox=(x, y, max(w, 1), max(h, 1)),
            confidence=1.0,
            source="pdf_vector"
        ))
        
    doc.close()
    return text_blocks


def warp_text_blocks(text_blocks: List, H: np.ndarray) -> List:
    """
    Warp the bounding boxes of a list of TextBlock objects using homography H.
    """
    if H is None:
        return text_blocks

    import cv2
    
    warped_blocks = []
    for b in text_blocks:
        x, y, w, h = b.bbox
        # Get the 4 corners
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Warp points
        warped_pts = cv2.perspectiveTransform(pts, H)
        warped_pts = warped_pts.reshape(-1, 2)
        
        # Compute new bounding box
        xs = warped_pts[:, 0]
        ys = warped_pts[:, 1]
        x0, y0 = int(np.min(xs)), int(np.min(ys))
        x1, y1 = int(np.max(xs)), int(np.max(ys))
        
        # Create a new TextBlock with warped bbox
        from core.ocr_engine import TextBlock
        warped_blocks.append(TextBlock(
            text=b.text,
            bbox=(x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)),
            confidence=b.confidence,
            source=b.source,
            angle=b.angle
        ))
    return warped_blocks


def get_pdf_or_ocr_text(img: np.ndarray, pdf_bytes: Optional[bytes] = None, page_number: int = 1, dpi: int = 300) -> 'OCRResult':
    from core.ocr_engine import extract_text, OCRResult
    
    if pdf_bytes is not None:
        try:
            vector_blocks = extract_vector_text(pdf_bytes, page_number, dpi)
            if vector_blocks:
                full_text = " ".join(b.text for b in vector_blocks)
                return OCRResult(
                    text_blocks=vector_blocks,
                    full_text=full_text,
                    engine_used="pdf_vector",
                    image_shape=img.shape[:2]
                )
        except Exception:
            pass # fall back to OCR
            
    return extract_text(img)


