from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.chat import ChatQuestion, ChatResponse, ConversationResponse
from app.controllers.chat_controller import ChatController

router = APIRouter()


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    payload: ChatQuestion,
    conversation_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> ChatResponse:
    """Executes RAG queries against indexed PDFs and queries Groq API."""
    return await ChatController.ask_question(
        db=db,
        payload=payload,
        user_id=current_user.id,
        conversation_id=conversation_id
    )


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    document_id: Optional[int] = None,
    title: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> ConversationResponse:
    """Initializes a new message history session for RAG interactions."""
    return await ChatController.create_conversation(
        db=db,
        user_id=current_user.id,
        document_id=document_id,
        title=title
    )


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> List[ConversationResponse]:
    """Retrieves all chat histories logs for the current user."""
    return await ChatController.list_conversations(db=db, user_id=current_user.id)


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> ConversationResponse:
    """Retrieves a specific chat thread conversation history with all messages."""
    return await ChatController.get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )
