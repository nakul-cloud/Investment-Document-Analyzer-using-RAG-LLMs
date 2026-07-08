import os
from typing import BinaryIO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import faiss
import numpy as np

from app.core.config import settings
from app.core.exceptions import DocumentIngestionError, ValidationError
from app.core.logging import logger
from app.models.document import Document
from app.utils.file_utils import calculate_file_sha256, save_file_safely
from app.rag.loaders.pdf_loader import extract_text_from_pdf
from app.rag.chunking.text_chunker import chunk_pages
from app.rag.chunking.post_processor import repair_chunks
from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import faiss_store, bm25_store


async def ingest_document(
    db: AsyncSession,
    file_obj: BinaryIO,
    filename: str,
    owner_id: int
) -> Document:
    """Orchestrates PDF text extraction, overlapping chunking, FAISS index construction,

    disk serialization, and database metadata recording.
    """
    # 1. Compute hash to check for duplicate uploads
    file_hash = calculate_file_sha256(file_obj)
    
    # Query for existing document with the same hash for this user
    result = await db.execute(
        select(Document).where(
            Document.file_hash == file_hash,
            Document.owner_id == owner_id
        )
    )
    existing_doc = result.scalars().first()
    if existing_doc:
        logger.info(f"Document duplicate found! Returning existing document ID {existing_doc.id}")
        return existing_doc

    # 2. Save file safely to uploads directory
    safe_filename = f"{file_hash}_{filename}"
    filepath = os.path.join(settings.UPLOAD_DIR, safe_filename)
    save_file_safely(file_obj, filepath)

    # 3. Extract text page-by-page
    try:
        pages = extract_text_from_pdf(filepath)
        if not pages:
            raise DocumentIngestionError(message="Unable to extract text content from the PDF file")
            
        pages_count = len(pages)
        logger.info(f"Extracted {pages_count} pages from {filename}")

        # 4. Chunk text pages (raw pass)
        raw_chunks = chunk_pages(pages)
        if not raw_chunks:
            raise DocumentIngestionError(message="Parsed text could not be successfully split into chunks")
        logger.info(f"Raw chunker produced {len(raw_chunks)} chunks")

        # 5. Repair pass — filter noise, merge fragments into tables,
        #    recompute headings, attach footnotes, collapse TOC pages
        chunks_metadata = repair_chunks(raw_chunks)
        if not chunks_metadata:
            raise DocumentIngestionError(message="Post-processing produced no usable chunks")

        table_count = sum(1 for c in chunks_metadata if c["chunk_type"] == "table")
        text_count  = sum(1 for c in chunks_metadata if c["chunk_type"] != "table")
        logger.info(
            f"After repair: {len(chunks_metadata)} chunks "
            f"(table={table_count}, text={text_count}) "
            f"from {len(raw_chunks)} raw blocks"
        )

        # 5. Load embedding model and encode chunks
        embed_model = get_embedding_model()
        chunk_texts = [chunk["text"] for chunk in chunks_metadata]
        logger.info("Generating embeddings for chunks...")
        embeddings = embed_model.encode(chunk_texts)
        
        # 6. Build FAISS index
        embeddings_np = np.array(embeddings).astype("float32")
        dim = embeddings_np.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(embeddings_np)
        logger.info(f"Constructed FAISS L2 index with dim {dim}")

        # 7. Serialize FAISS index + chunk metadata to files
        index_base = os.path.join(settings.INDEX_DIR, f"{file_hash}")
        faiss_store.save_index_and_metadata(
            index=faiss_index,
            chunks=chunks_metadata,
            base_path=index_base
        )

        # 8. Build and serialize BM25 keyword index
        bm25_index = bm25_store.build_bm25_index(chunks_metadata)
        bm25_store.save_bm25_index(bm25_index, base_path=index_base)
        logger.info(f"BM25 keyword index built with {len(chunks_metadata)} entries")

        # 9. Commit details to SQL Database
        new_doc = Document(
            filename=filename,
            filepath=filepath,
            file_hash=file_hash,
            index_path=index_base,  # Base path (without suffix)
            pages_count=pages_count,
            owner_id=owner_id
        )
        db.add(new_doc)
        await db.flush()
        
        logger.info(f"Document metadata recorded in database with ID {new_doc.id}")
        return new_doc

    except Exception as e:
        # Cleanup uploaded file on error
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        if not isinstance(e, DocumentIngestionError):
            logger.exception("Ingestion failed due to an unhandled system error")
            raise DocumentIngestionError(message=f"Ingestion failed: {e}")
        raise e
