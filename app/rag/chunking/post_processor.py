"""
Chunk Post-Processor — Repair/Merge Pass before Embedding

Fixes five classes of problems that raw PyMuPDF block extraction produces
on financial slide-deck PDFs (e.g. Adani Credit Update, HDFC KIM):

Problem 1 — Noise chunks
    Single page numbers ("3", "01"), watermarks ("STRICTLY CONFIDENTIAL"),
    dates ("November 2025"), and single-word labels become useless chunks.
    Fix: filter them out before they reach the vector DB.

Problem 2 — Context-less numbers
    "6.81x", "3.00x", "92,943" have no metric/entity/period attached.
    Fix: merge short (<60 char) text fragments into the enclosing table chunk
    on the same page as a "Page labels" suffix.

Problem 3 — Wrong section_title
    section_title is set from the previous block, not a real heading.
    Fix: recompute section_title per page using a stronger heading heuristic.

Problem 4 — Footnotes disconnected from tables
    EBITDA definition, "Data as on..." notes land in separate chunks.
    Fix: detect footnote blocks per page and attach to all table chunks on that page.

Problem 5 — Table of contents / TOC noise
    Early pages (typically 1–3) in slide decks are pure navigation; embedding them
    adds noise. Fix: detect TOC pages and demote to a single lightweight chunk.
"""
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional

# ─── Constants ────────────────────────────────────────────────────────────────

# Chunks shorter than this are treated as "fragments" and merged into tables
FRAGMENT_CHAR_LIMIT = 70

# Noise patterns — these add zero retrieval value (completely generic)
_NOISE_PATTERNS = [
    r"^\d{1,3}$",                        # bare page numbers
    r"^[0-9][0-9a-z]{0,3}$",             # section indices ("01", "1.1", "02b")
    r"^(?:strictly\s+)?confidential\s*(?:strictly\s+confidential)?$", # watermarks
    r"^(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}$", # generic Month YYYY
    r"^[a-zA-Z]{3,9}\s+\d{4}$",          # fallback generic Month YYYY
    r"^\d{4}$",                           # bare year
    r"^(?:q[1-4]|h[12])\s*(?:fy)?\d{2,4}$", # generic quarter/half-year
    r"^(?:fy|cy)\d{2,4}$",                  # generic fiscal/calendar year
    r"^page\s+\d+$",
    r"^\s*[-–—]+\s*$",                   # separators
    r"^[•·▪◦]\s*$",                      # bullet character
    r"^none$",
    r"^(?:n\.?a\.?|nil)$",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

# Heading heuristics — detects structural financial sections dynamically
_HEADING_KEYWORDS = [
    "portfolio", "leverage", "liquidity", "borrowing", "rating", "risk", "credit",
    "revenue", "ebitda", "debt", "equity", "fund", "scheme", "performance", "overview",
    "summary", "appendix", "section", "update", "profile", "limited", "capital", "asset",
    "liability", "management", "governance", "operation", "annual", "quarterly", "report",
    "disclosure", "key", "information", "policy", "strategy", "outlook", "prospectus",
    "statement", "sheet", "cash", "flow", "income", "expense", "fee", "cost", "allocation"
]
_HEADING_RE = re.compile(
    rf"(?:{'|'.join(_HEADING_KEYWORDS)}).{{0,80}}",
    re.IGNORECASE,
)

# Footnote / definition patterns (generic symbols, note prefixes, definitions, and currency indicators)
_FOOTNOTE_RE = re.compile(
    r"(?:^[\s*†‡§#•-]*\b(?:note|source|definition|disclaimer|legend|reference|figures? in|data as (?:of|on))\b|"
    r"^\s*[*†‡§#]\s+|"
    r"\b(?:calculated as|defined as|refer to|excludes?|includes?|net of|equal to)\b|"
    r"\b(?:USD|INR|EUR|GBP|Rs\.?|₹|\$)\b|"
    r"^[0-9*†‡§#\s-]*\b(?:ebitda|pat|nav|idcw|expense|ratio|fee)\b)",
    re.IGNORECASE,
)

# Table of contents detection
_TOC_SIGNALS = ["contents", "table of contents", "index"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_noise(text: str) -> bool:
    """Returns True for chunks that carry no retrieval value."""
    t = text.strip()
    if not t or len(t) < 2:
        return True
    # Repeated identical lines (watermarks like "STRICTLY CONFIDENTIAL\nSTRICTLY CONFIDENTIAL")
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if len(lines) >= 2 and len(set(lines)) == 1:
        return True
    return bool(_NOISE_RE.match(t))


def _is_footnote(text: str) -> bool:
    return bool(_FOOTNOTE_RE.search(text))


def _is_toc_page(chunks: List[Dict[str, Any]]) -> bool:
    """Returns True if the chunks on a page look like a table of contents."""
    combined = " ".join(c["text"].lower() for c in chunks)
    return any(sig in combined for sig in _TOC_SIGNALS) and len(chunks) > 5


def _best_heading(chunks: List[Dict[str, Any]], page: int) -> str:
    """
    Pick the most informative heading from the page's text chunks.
    Priority: longest match against _HEADING_RE, else first non-noise line.
    """
    candidates = []
    for c in chunks:
        if c.get("chunk_type") == "table":
            continue
        text = c["text"].strip()
        first_line = text.splitlines()[0].strip() if text else ""
        if _HEADING_RE.search(first_line):
            candidates.append(first_line)

    if candidates:
        # Prefer longer, more descriptive headings
        return max(candidates, key=len)

    # Fallback: first non-noise text block's first line
    for c in chunks:
        if c.get("chunk_type") != "table":
            line = c["text"].strip().splitlines()[0].strip()
            if line and not _is_noise(line):
                return line

    return f"Page {page}"


# ─── Main Repair Pass ─────────────────────────────────────────────────────────

def repair_chunks(raw_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Post-process raw chunker output with a page-level merge/repair pass.

    Pipeline:
      1. Group chunks by page
      2. Detect & skip TOC pages (emit a single lightweight TOC chunk)
      3. Filter noise chunks
      4. Recompute section_title using heading heuristics
      5. Collect footnote/definition text per page
      6. Collect short fragment labels per page
      7. Emit enriched table chunks (with fragments + footnotes attached)
      8. Merge long prose blocks into one chunk per page

    Args:
        raw_chunks: Output of text_chunker.chunk_pages()

    Returns:
        Repaired, enriched chunk list ready for embedding.
    """
    by_page: Dict[int, List[Dict]] = defaultdict(list)
    for chunk in raw_chunks:
        by_page[chunk["page"]].append(chunk)

    repaired: List[Dict[str, Any]] = []

    for page_num in sorted(by_page.keys()):
        page_chunks = by_page[page_num]

        # ── TOC pages: emit a single summary chunk ──────────────────────────
        if _is_toc_page(page_chunks):
            toc_text = "\n".join(
                c["text"] for c in page_chunks if not _is_noise(c["text"])
            )
            if toc_text.strip():
                repaired.append({
                    "text": f"[Table of Contents | Page {page_num}]\n{toc_text}",
                    "page": page_num,
                    "chunk_type": "text",
                    "section_title": "Table of Contents",
                })
            continue

        # ── Recompute real section heading for this page ─────────────────────
        page_heading = _best_heading(page_chunks, page_num)

        # ── Separate chunk types ─────────────────────────────────────────────
        table_chunks = [c for c in page_chunks if c.get("chunk_type") == "table"]
        text_chunks  = [c for c in page_chunks if c.get("chunk_type") != "table"]

        # ── Collect footnotes (definition blocks) ────────────────────────────
        footnotes = [
            c["text"] for c in text_chunks
            if _is_footnote(c["text"]) and not _is_noise(c["text"])
        ]
        footnote_block = "\n".join(footnotes) if footnotes else None

        # ── Collect short label/number fragments ─────────────────────────────
        fragments = [
            c["text"].replace("\n", " ").strip()
            for c in text_chunks
            if (
                len(c["text"].strip()) < FRAGMENT_CHAR_LIMIT
                and not _is_noise(c["text"])
                and not _is_footnote(c["text"])
            )
        ]
        fragment_blob = " | ".join(fragments) if fragments else None

        # ── Emit enriched table chunks ────────────────────────────────────────
        for tbl in table_chunks:
            parts = [
                f"[TABLE | Section: {page_heading} | Page {page_num}]",
                tbl["text"],
            ]
            if fragment_blob:
                parts.append(f"Page context (labels/figures): {fragment_blob}")
            if footnote_block:
                parts.append(f"Definitions: {footnote_block}")

            repaired.append({
                "text": "\n".join(parts),
                "page": page_num,
                "chunk_type": "table",
                "section_title": page_heading,
            })

        # ── Emit prose chunks (long text blocks only) ─────────────────────────
        prose_blocks = [
            c["text"].strip()
            for c in text_chunks
            if (
                len(c["text"].strip()) >= FRAGMENT_CHAR_LIMIT
                and not _is_noise(c["text"])
            )
        ]

        if prose_blocks:
            # Merge all long prose on the page into one coherent chunk
            merged_prose = f"[TEXT | Section: {page_heading} | Page {page_num}]\n"
            merged_prose += "\n\n".join(prose_blocks)

            repaired.append({
                "text": merged_prose,
                "page": page_num,
                "chunk_type": "text",
                "section_title": page_heading,
            })

        # ── Pages with no tables and only fragments: emit a fragment chunk ────
        elif not table_chunks and fragments:
            repaired.append({
                "text": (
                    f"[TEXT | Section: {page_heading} | Page {page_num}]\n"
                    + " | ".join(fragments)
                ),
                "page": page_num,
                "chunk_type": "text",
                "section_title": page_heading,
            })

    return repaired
