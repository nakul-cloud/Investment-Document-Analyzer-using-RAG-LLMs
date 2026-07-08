from typing import List, BinaryIO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.document import Document
from app.schemas.document import DocumentResponse
from app.services import ingestion_service
from app.core.exceptions import AppException, ValidationError, NotFoundError
from app.core.logging import logger
from app.utils.file_utils import remove_file_safely


class DocumentController:
    @staticmethod
    async def upload_document(
        db: AsyncSession,
        file_obj: BinaryIO,
        filename: str,
        user_id: int
    ) -> DocumentResponse:
        """Controller orchestrating PDF file ingestion and exception parsing."""
        try:
            if not filename.lower().endswith(".pdf"):
                raise ValidationError(message="Only PDF files are supported for document indexing")
                
            doc = await ingestion_service.ingest_document(
                db=db,
                file_obj=file_obj,
                filename=filename,
                owner_id=user_id
            )
            return DocumentResponse.model_validate(doc)
        except AppException as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during document ingestion: {e}")
            raise ValidationError(message=f"Document indexing failed: {e}")

    @staticmethod
    async def list_documents(db: AsyncSession, user_id: int) -> List[DocumentResponse]:
        """Lists all uploaded documents metadata registered to the user."""
        try:
            result = await db.execute(
                select(Document).where(Document.owner_id == user_id)
            )
            docs = result.scalars().all()
            return [DocumentResponse.model_validate(d) for d in docs]
        except Exception as e:
            logger.error(f"Failed to fetch documents for user {user_id}: {e}")
            raise ValidationError(message="Could not load files list")

    @staticmethod
    async def delete_document(db: AsyncSession, document_id: int, user_id: int) -> None:
        """Removes document records from database and deletes binary storage files."""
        try:
            result = await db.execute(
                select(Document).where(Document.id == document_id, Document.owner_id == user_id)
            )
            doc = result.scalars().first()
            if not doc:
                raise NotFoundError(message="Document not found or access denied")

            # Remove physical uploaded file
            remove_file_safely(doc.filepath)

            # Remove serialized index files (FAISS + metadata JSON + BM25 keyword index)
            if doc.index_path:
                remove_file_safely(f"{doc.index_path}.faiss")
                remove_file_safely(f"{doc.index_path}.json")
                remove_file_safely(f"{doc.index_path}.bm25.pkl")

            await db.delete(doc)
            await db.commit()
            logger.info(f"Document {document_id} and all index files deleted successfully")
        except AppException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise ValidationError(message="Failed to delete document files.")
