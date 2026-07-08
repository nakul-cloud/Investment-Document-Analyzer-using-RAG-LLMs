"""
BM25 Keyword Index — Sparse retrieval component for hybrid search.

BM25 (Best Match 25) is a probabilistic keyword scoring algorithm.
It scores documents based on:
- Term frequency (TF): how often the query word appears in the chunk
- Inverse document frequency (IDF): how rare the word is across all chunks
- Document length normalization: penalizes very long documents

We serialize the BM25 index object to disk using pickle alongside the FAISS index.
Both share the same underlying chunks metadata (.json sidecar).
"""
import os
import pickle
import re
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi

from app.core.logging import logger

# File extension for the pickled BM25 index
BM25_EXT = ".bm25.pkl"


def _tokenize(text: str) -> List[str]:
    """
    Lowercase tokenizer with punctuation stripping.
    Financial text has lots of acronyms (EBITDA, TTM, NAV) —
    we keep them intact; only split on non-alphanumeric characters.
    """
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    """
    Builds a BM25Okapi index from chunk metadata.

    Args:
        chunks: List of chunk dicts with at least a 'text' key.

    Returns:
        A fitted BM25Okapi instance.
    """
    tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def save_bm25_index(index: BM25Okapi, base_path: str) -> str:
    """Serializes the BM25 index to disk via pickle."""
    pkl_path = f"{base_path}{BM25_EXT}"
    with open(pkl_path, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"BM25 index saved to {pkl_path}")
    return pkl_path


def load_bm25_index(base_path: str) -> BM25Okapi:
    """Loads the BM25 index from disk. Raises FileNotFoundError if absent."""
    pkl_path = f"{base_path}{BM25_EXT}"
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"BM25 index not found at: {pkl_path}")
    with open(pkl_path, "rb") as f:
        index = pickle.load(f)
    logger.info(f"BM25 index loaded from {pkl_path}")
    return index


def query_bm25(
    index: BM25Okapi,
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Performs BM25 keyword search.

    Args:
        index: Fitted BM25Okapi instance.
        chunks: The same ordered chunk list used to build the index.
        query: Raw query string (tokenized internally).
        top_k: Number of top results to return.

    Returns:
        List of (chunk_index, bm25_score) tuples, sorted descending by score.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = index.get_scores(tokens)  # ndarray of shape (num_chunks,)

    # Build (index, score) pairs and sort descending
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # Filter zero-score results (no keyword overlap at all)
    ranked = [(idx, score) for idx, score in ranked if score > 0.0]

    return ranked[:top_k]
