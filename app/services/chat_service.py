from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from groq import Groq
import numpy as np

from app.core.config import settings
from app.core.constants import RETRIEVAL_LIMIT
from app.core.exceptions import ValidationError, NotFoundError
from app.core.prompts import get_finance_rag_prompt
from app.core.logging import logger
from app.models.document import Document
from app.models.conversation import Conversation, Message
from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import faiss_store, bm25_store
from app.rag import retrieval

_groq_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    """Singleton getter for the Groq API client."""
    global _groq_client
    if _groq_client is None:
        if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
            logger.warning("GROQ_API_KEY is not configured in environment settings.")
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


async def get_latest_document(db: AsyncSession, user_id: int) -> Optional[Document]:
    """Retrieves the latest processed document for a given user."""
    result = await db.execute(
        select(Document)
        .where(Document.owner_id == user_id)
        .order_by(desc(Document.created_at))
        .limit(1)
    )
    return result.scalars().first()


async def execute_rag_query(
    db: AsyncSession,
    question: str,
    user_id: int,
    document_id: Optional[int] = None,
    conversation_id: Optional[int] = None
) -> Tuple[str, List[int]]:
    """Retrieves document chunks from FAISS, constructs prompts, calls Groq,

    and writes message histories into database logs.
    """
    # 1. Resolve Document context
    doc: Optional[Document] = None
    if document_id:
        result = await db.execute(
            select(Document).where(Document.id == document_id, Document.owner_id == user_id)
        )
        doc = result.scalars().first()
        if not doc:
            raise NotFoundError(message="Requested document could not be found or access is denied")
    else:
        doc = await get_latest_document(db, user_id)

    # If no document is indexed, return early
    if not doc or not doc.index_path:
        logger.warning(f"RAG query failed: User {user_id} has no indexed documents")
        return "No document indexed. Please upload a PDF document first.", []

    # 2. Load FAISS index and chunk metadata from disk
    try:
        faiss_index, chunks_metadata = faiss_store.load_index_and_metadata(doc.index_path)
    except Exception as e:
        logger.error(f"Failed to load FAISS store for document {doc.id}: {e}")
        raise ValidationError(message="Failed to load document indices. Try re-uploading the file.")

    # 3. Embed the query
    embed_model = get_embedding_model()
    q_emb = embed_model.encode([question])

    # 4. Hybrid retrieval — RRF(dense FAISS + sparse BM25)
    #    Falls back to dense-only for legacy documents that predate BM25 indexing.
    try:
        bm25_index = bm25_store.load_bm25_index(doc.index_path)
        matched_chunks = retrieval.hybrid_search(
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            chunks=chunks_metadata,
            query=question,
            query_embedding=q_emb,
            top_k=RETRIEVAL_LIMIT,
        )
        logger.info(f"Hybrid search used for document {doc.id}")
    except FileNotFoundError:
        # Legacy document — BM25 index doesn't exist yet; fall back to FAISS only
        logger.warning(
            f"BM25 index missing for document {doc.id}. "
            "Falling back to dense-only search. Re-upload to enable hybrid search."
        )
        matched_chunks = faiss_store.query_vector_store(
            faiss_index, chunks_metadata, q_emb, top_k=RETRIEVAL_LIMIT
        )

    # Build labelled context blocks using chunk_type and section_title metadata
    context_blocks = []
    used_pages = set()
    for chunk in matched_chunks:
        chunk_type = chunk.get("chunk_type", "text")
        section = chunk.get("section_title", "")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        used_pages.add(page)

        # Format: label what kind of content this is so the LLM has context
        if chunk_type == "table":
            header = f"[TABLE | Section: {section} | Page {page}]"
        else:
            header = f"[TEXT | Section: {section} | Page {page}]"

        context_blocks.append(f"{header}\n{text}")

    context = "\n\n---\n\n".join(context_blocks)
    sorted_pages = sorted(list(used_pages))

    # 4. Construct RAG Prompt using the detailed financial analyst prompt template
    prompt = get_finance_rag_prompt(context, question)

    # 5. Invoke Groq LLM API
    client = get_groq_client()
    answer_text = ""
    try:
        # Standard parameters matching Flask application
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_completion_tokens=1024,
            top_p=1.0,
            stream=True,
        )
        # Collect stream
        answer_parts = []
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                answer_parts.append(delta)
        answer_text = "".join(answer_parts).strip()
    except Exception as e:
        logger.exception(f"Groq API call failed: {e}")
        answer_text = f"Error generating answer from Groq: {e}"

    # 6. Persist Q&A turn to conversation history
    #    Auto-create a conversation if the frontend didn't send one
    #    (ensures messages ALWAYS land in SQLite).
    if not conversation_id:
        new_conv = Conversation(
            title=question[:80],          # Use question as conversation title
            user_id=user_id,
            document_id=doc.id if doc else None,
        )
        db.add(new_conv)
        await db.flush()                  # Assign the ID without committing yet
        conversation_id = new_conv.id
        logger.info(f"Auto-created conversation {conversation_id} for user {user_id}")
    else:
        # Verify the conversation belongs to this user
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
        )
        if not result.scalars().first():
            logger.warning(
                f"conversation_id {conversation_id} not found for user {user_id}; "
                "auto-creating a new one"
            )
            fallback = Conversation(
                title=question[:80],
                user_id=user_id,
                document_id=doc.id if doc else None,
            )
            db.add(fallback)
            await db.flush()
            conversation_id = fallback.id

    # Save user question and assistant answer as messages
    user_msg = Message(
        role="user",
        content=question,
        conversation_id=conversation_id,
    )
    assistant_msg = Message(
        role="assistant",
        content=answer_text,
        pages=",".join(map(str, sorted_pages)) if sorted_pages else None,
        conversation_id=conversation_id,
    )
    db.add_all([user_msg, assistant_msg])
    await db.flush()          # get_db commits the whole transaction after this
    logger.info(f"Saved Q&A to conversation {conversation_id} (user={user_id})")

    return answer_text, sorted_pages
