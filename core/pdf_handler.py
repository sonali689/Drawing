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
