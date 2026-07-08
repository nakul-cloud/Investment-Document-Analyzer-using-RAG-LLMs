import json
import os
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from app.core.logging import logger


def save_index_and_metadata(
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    base_path: str
) -> Tuple[str, str]:
    """Saves FAISS index to a binary file and chunks metadata to a JSON file."""
    faiss_path = f"{base_path}.faiss"
    json_path = f"{base_path}.json"

    # Save FAISS binary index
    faiss.write_index(index, faiss_path)

    # Save chunks metadata
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"FAISS index and metadata saved to {base_path}.*")
    return faiss_path, json_path


def load_index_and_metadata(
    base_path: str
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Loads FAISS index and chunks metadata from disk."""
    faiss_path = f"{base_path}.faiss"
    json_path = f"{base_path}.json"

    if not os.path.exists(faiss_path) or not os.path.exists(json_path):
        raise FileNotFoundError(f"FAISS index components not found at base path: {base_path}")

    index = faiss.read_index(faiss_path)
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"FAISS index loaded successfully with {len(chunks)} chunks.")
    return index, chunks


def query_vector_store(
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    query_embedding: np.ndarray,
    top_k: int = 6
) -> List[Dict[str, Any]]:
    """Queries FAISS index and returns matching metadata chunk records."""
    # Ensure float32 and correct dimensions
    q_emb = np.array(query_embedding).astype("float32")
    if len(q_emb.shape) == 1:
        q_emb = np.expand_dims(q_emb, axis=0)

    distances, indices = index.search(q_emb, top_k)
    
    results = []
    for i in indices[0]:
        idx = int(i)
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
            
    return results
