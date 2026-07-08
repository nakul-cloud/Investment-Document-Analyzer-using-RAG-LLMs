from typing import List
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.document import DocumentResponse
from app.controllers.document_controller import DocumentController

router = APIRouter()


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    pdf: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> DocumentResponse:
    """Uploads and embeds a new PDF document for the current user."""
    return await DocumentController.upload_document(
        db=db,
        file_obj=pdf.file,
        filename=pdf.filename,
        user_id=current_user.id
    )


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> List[DocumentResponse]:
    """Retrieves all indexed document models for the current user."""
    return await DocumentController.list_documents(db=db, user_id=current_user.id)


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Deletes an indexed PDF file and all associated FAISS store files."""
    await DocumentController.delete_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id
    )
    return {"status": "ok", "msg": "Document deleted successfully"}
