"""
Text Chunker — Upgrades 2 & 3: Dual-Type Chunking with Rich Metadata

Strategy:
- TABLE chunks: each detected table is emitted as ONE atomic chunk.
  The table text preserves row-column structure. We prepend the nearest
  preceding heading as the caption so embeddings understand the table's topic.
- TEXT chunks: split on paragraph boundaries (double-newlines / block separators).
  If a paragraph exceeds MAX_PARAGRAPH_WORDS, we apply a sliding word window
  as a secondary split. This avoids destroying sentence context.
- Metadata carried per chunk:
    {
      "text":          str  — the chunk content (ready to embed)
      "page":          int  — source page number
      "chunk_type":    str  — "table" | "text"
      "section_title": str  — nearest section heading seen before this chunk
    }
"""
import re
from typing import List, Dict, Any

from app.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# A paragraph text block beyond this word count gets a secondary sliding split
MAX_PARAGRAPH_WORDS = DEFAULT_CHUNK_SIZE  # 450 words
SECONDARY_OVERLAP = DEFAULT_CHUNK_OVERLAP  # 50 words

# Simple heuristic: a line is a "heading" if it is short + title-cased or ALL-CAPS
_HEADING_MAX_WORDS = 12


def _is_heading(text: str) -> bool:
    """Returns True if the text looks like a section heading."""
    words = text.split()
    if not words or len(words) > _HEADING_MAX_WORDS:
        return False
    # All-caps heading (common in financial KIMs)
    if text.isupper():
        return True
    # Title-case with no trailing period (section titles rarely end with '.')
    if text.istitle() and not text.endswith("."):
        return True
    # Numbered heading patterns: "1.", "1.1", "(i)", "A."
    if re.match(r"^(\d+\.|\(\w+\)|[A-Z]\.)[\s]", text):
        return True
    return False


def _sliding_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fallback sliding word-window split for very long paragraphs."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i: i + chunk_size])
        if chunk_text.strip():
            chunks.append(chunk_text)
    return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = MAX_PARAGRAPH_WORDS,
    overlap: int = SECONDARY_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Dual-type chunker with rich metadata.

    Args:
        pages: Output from pdf_loader.extract_text_from_pdf()
               Each item: {'page': int, 'blocks': [{'type', 'text', 'bbox'}]}
        chunk_size: Max words for secondary sliding split on long paragraphs
        overlap: Word overlap for secondary sliding split

    Returns:
        List of chunk dicts with keys: text, page, chunk_type, section_title
    """
    chunks: List[Dict[str, Any]] = []
    current_section_title = "Document"

    for page in pages:
        page_num = page.get("page", 1)
        blocks = page.get("blocks", [])

        for block in blocks:
            block_type = block.get("type", "text")
            block_text = block.get("text", "").strip()

            if not block_text:
                continue

            # --- Track section headings ---
            if block_type == "text" and _is_heading(block_text):
                current_section_title = block_text
                # Do NOT skip the heading — include it as a tiny text chunk so
                # queries about section names can still retrieve the context
                chunks.append({
                    "text": block_text,
                    "page": page_num,
                    "chunk_type": "text",
                    "section_title": current_section_title,
                })
                continue

            # --- TABLE BLOCK: emit as one atomic chunk (Upgrade 2) ---
            if block_type == "table":
                # Prepend the current section title as caption for better embedding
                captioned = f"[{current_section_title}]\n{block_text}"
                chunks.append({
                    "text": captioned,
                    "page": page_num,
                    "chunk_type": "table",
                    "section_title": current_section_title,
                })
                continue

            # --- TEXT BLOCK: split on paragraph boundaries first ---
            # PyMuPDF blocks are already single paragraphs, but pdfplumber
            # fallback may return multi-paragraph page dumps
            paragraphs = re.split(r"\n{2,}", block_text)

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                word_count = len(para.split())

                if word_count <= chunk_size:
                    # Short enough — emit as one chunk
                    chunks.append({
                        "text": para,
                        "page": page_num,
                        "chunk_type": "text",
                        "section_title": current_section_title,
                    })
                else:
                    # Too long — apply secondary sliding window split
                    sub_chunks = _sliding_split(para, chunk_size, overlap)
                    for sub in sub_chunks:
                        chunks.append({
                            "text": sub,
                            "page": page_num,
                            "chunk_type": "text",
                            "section_title": current_section_title,
                        })

    return chunks
