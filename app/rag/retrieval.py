"""
Hybrid Retrieval — Reciprocal Rank Fusion (RRF) of Dense + Sparse results.

Why hybrid?
- Dense (FAISS semantic): catches paraphrases, synonyms, conceptual similarity
  e.g. "return on equity" ↔ "how profitable is the company"
- Sparse (BM25 keyword): catches exact matches, tickers, ratios, clause numbers
  e.g. "Net Debt / EBITDA 3.00x", "Risk Factor (iv)", "AGEL", "TTM"
- Together via RRF: best of both worlds without needing a heavy cross-encoder

RRF Formula:
    score(d) = Σ_r  1 / (k + rank_r(d))
where k=60 is a constant that dampens the impact of high ranks.

How ranking works:
1. Dense search returns ranked list of chunks by vector distance
2. BM25 returns ranked list of chunks by keyword score
3. Each chunk gets an RRF score contribution from each list it appears in
4. Final list is sorted by total RRF score descending
"""
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
import faiss

from app.core.logging import logger
from app.rag.vectorstore import faiss_store, bm25_store

# RRF constant — 60 is the standard from the original paper (Cormack et al. 2009)
RRF_K = 60


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-indexed)."""
    return 1.0 / (k + rank)


def hybrid_search(
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    chunks: List[Dict[str, Any]],
    query: str,
    query_embedding: np.ndarray,
    top_k: int = 6,
    dense_candidates: int = 20,
    sparse_candidates: int = 20,
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most relevant chunks using RRF-fused hybrid search.

    Args:
        faiss_index: Loaded FAISS index (dense).
        bm25_index: Loaded BM25Okapi index (sparse).
        chunks: Shared chunk metadata list (ordered, same for both indexes).
        query: Raw user query string (for BM25 tokenisation).
        query_embedding: Pre-computed query vector (for FAISS search).
        top_k: Final number of chunks to return after fusion.
        dense_candidates: How many candidates to fetch from FAISS before fusion.
        sparse_candidates: How many candidates to fetch from BM25 before fusion.

    Returns:
        List of chunk dicts with an extra 'retrieval_score' key (RRF score).
    """
    rrf_scores: Dict[int, float] = {}  # chunk_index → accumulated RRF score

    # --- Dense (Semantic) Search via FAISS ---
    q_emb = np.array(query_embedding).astype("float32")
    if len(q_emb.shape) == 1:
        q_emb = np.expand_dims(q_emb, axis=0)

    distances, indices = faiss_index.search(q_emb, dense_candidates)
    dense_hits = [int(i) for i in indices[0] if 0 <= int(i) < len(chunks)]

    for rank, chunk_idx in enumerate(dense_hits, start=1):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + _rrf_score(rank)

    logger.debug(f"Dense search returned {len(dense_hits)} candidates")

    # --- Sparse (Keyword) Search via BM25 ---
    sparse_hits = bm25_store.query_bm25(bm25_index, chunks, query, top_k=sparse_candidates)

    for rank, (chunk_idx, bm25_score) in enumerate(sparse_hits, start=1):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + _rrf_score(rank)

    logger.debug(f"BM25 search returned {len(sparse_hits)} candidates")

    # --- RRF Fusion: sort by accumulated score descending ---
    sorted_hits = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_idx, score in sorted_hits[:top_k]:
        chunk = dict(chunks[chunk_idx])  # copy to avoid mutating original
        chunk["retrieval_score"] = round(score, 4)
        results.append(chunk)

    # Log retrieval mix for visibility
    n_table = sum(1 for c in results if c.get("chunk_type") == "table")
    n_text = sum(1 for c in results if c.get("chunk_type") == "text")
    logger.info(
        f"Hybrid retrieval: {len(results)} chunks returned "
        f"(table={n_table}, text={n_text}) — top RRF score: "
        f"{results[0]['retrieval_score'] if results else 0}"
    )

    return results


def load_hybrid_indexes(base_path: str):
    """
    Convenience loader: loads both FAISS and BM25 indexes from a base path.

    Returns:
        (faiss_index, bm25_index, chunks) — ready for hybrid_search()

    Raises:
        FileNotFoundError if either index is missing.
    """
    faiss_index, chunks = faiss_store.load_index_and_metadata(base_path)
    bm25_index = bm25_store.load_bm25_index(base_path)
    return faiss_index, bm25_index, chunks
