"""
PDF Loader — Upgrade 1: Table-Aware Parsing via PyMuPDF

Strategy:
- Use PyMuPDF (fitz) to open each PDF page.
- On each page, call `page.find_tables()` to detect all table regions.
- Extract each table as a structured markdown-style block with its caption.
- Extract remaining non-table text as paragraph blocks.
- Fall back to pdfplumber text extraction if fitz fails.
- Fall back to OCR (pytesseract) if extracted text is too sparse.
"""
import os
from typing import List, Dict, Any

from app.core.logging import logger


def _extract_with_pymupdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts structured page content from a PDF using PyMuPDF.

    Returns a list of page dicts, each containing:
    - 'page': page number (1-indexed)
    - 'blocks': list of content blocks with type ('table' or 'text'),
                text content, and bounding box
    """
    import fitz  # PyMuPDF

    pages_data = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        blocks = []

        # --- Step 1: Detect and extract tables ---
        # find_tables() returns a TableFinder with a .tables list
        table_finder = page.find_tables()
        table_bboxes = []

        for table in table_finder.tables:
            table_bboxes.append(table.bbox)

            # Extract header + rows as readable text
            # header: list of cell strings, rows: list of lists
            extracted = table.extract()
            if not extracted:
                continue

            # Build a clean pipe-separated markdown table string
            lines = []
            for row_idx, row in enumerate(extracted):
                # Replace None cells with empty string
                cells = [str(cell).strip() if cell is not None else "" for cell in row]
                lines.append(" | ".join(cells))
                # Insert a separator after the header row
                if row_idx == 0:
                    lines.append("-" * len(lines[0]))

            table_text = "\n".join(lines)
            if table_text.strip():
                blocks.append({
                    "type": "table",
                    "text": table_text,
                    "bbox": table.bbox,
                })

        # --- Step 2: Extract non-table text blocks ---
        # Get all text blocks on the page as (x0,y0,x1,y1,text,block_no,block_type)
        raw_blocks = page.get_text("blocks", sort=True)
        for rb in raw_blocks:
            # rb[6] == 0 means text block (1 = image block)
            if len(rb) < 7 or rb[6] != 0:
                continue

            block_bbox = rb[:4]
            block_text = rb[4].strip()
            if not block_text:
                continue

            # Skip text blocks that fall inside a detected table's area
            is_inside_table = False
            for tb_bbox in table_bboxes:
                # Check significant overlap: block center point inside table rect
                cx = (block_bbox[0] + block_bbox[2]) / 2
                cy = (block_bbox[1] + block_bbox[3]) / 2
                if (tb_bbox[0] <= cx <= tb_bbox[2]) and (tb_bbox[1] <= cy <= tb_bbox[3]):
                    is_inside_table = True
                    break

            if not is_inside_table:
                blocks.append({
                    "type": "text",
                    "text": block_text,
                    "bbox": block_bbox,
                })

        if blocks:
            pages_data.append({"page": page_num, "blocks": blocks})

    doc.close()
    return pages_data


def _extract_with_pdfplumber_fallback(pdf_path: str) -> List[Dict[str, Any]]:
    """Fallback: extracts raw text via pdfplumber, wrapping each page as a single text block."""
    import pdfplumber

    pages_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_data.append({
                        "page": i + 1,
                        "blocks": [{"type": "text", "text": text.strip(), "bbox": None}]
                    })
    except Exception as e:
        logger.error(f"pdfplumber fallback failed: {e}")
    return pages_data


def _extract_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    """Last-resort OCR via pytesseract + pdf2image."""
    pages_data = []
    try:
        from pdf2image import convert_from_path
        import pytesseract

        logger.info("OCR fallback triggered (sparse text document)...")
        images = convert_from_path(pdf_path)
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img) or ""
            if text.strip():
                pages_data.append({
                    "page": i + 1,
                    "blocks": [{"type": "text", "text": text.strip(), "bbox": None}]
                })
    except Exception as e:
        logger.warning(
            f"OCR fallback failed. Ensure tesseract and poppler binaries are installed. Details: {e}"
        )
    return pages_data


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Main entry point. Extracts structured page content from a PDF.

    Priority order:
    1. PyMuPDF table-aware extraction (primary)
    2. pdfplumber plain text extraction (fallback)
    3. pytesseract OCR (last resort for scanned/image PDFs)

    Returns:
        List of page dicts, each with keys: 'page', 'blocks'
        Each block has: 'type' ('text' or 'table'), 'text', 'bbox'
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return []

    # --- Primary: PyMuPDF ---
    try:
        pages_data = _extract_with_pymupdf(pdf_path)
        total_chars = sum(
            len(b["text"]) for p in pages_data for b in p["blocks"]
        )
        if total_chars >= 100:
            table_count = sum(
                1 for p in pages_data for b in p["blocks"] if b["type"] == "table"
            )
            text_count = sum(
                1 for p in pages_data for b in p["blocks"] if b["type"] == "text"
            )
            logger.info(
                f"PyMuPDF extracted {total_chars} chars from {len(pages_data)} pages "
                f"({table_count} table blocks, {text_count} text blocks)"
            )
            return pages_data
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed, trying pdfplumber: {e}")

    # --- Fallback: pdfplumber ---
    pages_data = _extract_with_pdfplumber_fallback(pdf_path)
    total_chars = sum(len(b["text"]) for p in pages_data for b in p["blocks"])
    if total_chars >= 100:
        logger.info(f"pdfplumber extracted {total_chars} chars as fallback")
        return pages_data

    # --- Last resort: OCR ---
    logger.warning("Both PyMuPDF and pdfplumber returned sparse text, falling back to OCR")
    return _extract_with_ocr(pdf_path)
