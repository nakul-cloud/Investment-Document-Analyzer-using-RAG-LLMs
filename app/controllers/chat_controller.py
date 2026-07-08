from datetime import datetime
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.conversation import Conversation
from app.schemas.chat import ChatQuestion, ChatResponse, ConversationResponse
from app.services import chat_service
from app.core.exceptions import AppException, ValidationError, NotFoundError
from app.core.logging import logger


class ChatController:
    @staticmethod
    async def ask_question(
        db: AsyncSession,
        payload: ChatQuestion,
        user_id: int,
        conversation_id: Optional[int] = None
    ) -> ChatResponse:
        """Controller orchestrating RAG queries and capturing API/Groq runtime failures."""
        try:
            answer, pages = await chat_service.execute_rag_query(
                db=db,
                question=payload.question,
                user_id=user_id,
                document_id=payload.document_id,
                conversation_id=conversation_id
            )
            return ChatResponse(status="ok", answer=answer, pages=pages)
        except AppException as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error executing chat query: {e}")
            raise ValidationError(message=f"Retrieval process failed: {e}")

    @staticmethod
    async def create_conversation(
        db: AsyncSession,
        user_id: int,
        document_id: Optional[int] = None,
        title: Optional[str] = None
    ) -> ConversationResponse:
        """Creates a new conversation session to bind chat history."""
        try:
            new_conv = Conversation(
                title=title or "New Chat Thread",
                user_id=user_id,
                document_id=document_id
            )
            db.add(new_conv)
            await db.flush()
            
            logger.info(f"Created conversation thread ID {new_conv.id} for user {user_id}")
            return ConversationResponse(
                id=new_conv.id,
                title=new_conv.title,
                document_id=new_conv.document_id,
                created_at=new_conv.created_at or datetime.now(),
                messages=[]
            )
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise ValidationError(message="Failed to initialize chat thread.")

    @staticmethod
    async def list_conversations(db: AsyncSession, user_id: int) -> List[ConversationResponse]:
        """Lists all conversations for a user, eager-loading related messages."""
        try:
            result = await db.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .options(selectinload(Conversation.messages))
            )
            convs = result.scalars().all()
            return [ConversationResponse.model_validate(c) for c in convs]
        except Exception as e:
            logger.error(f"Failed to list conversations for user {user_id}: {e}")
            raise ValidationError(message="Failed to load chat history.")

    @staticmethod
    async def get_conversation(
        db: AsyncSession,
        conversation_id: int,
        user_id: int
    ) -> ConversationResponse:
        """Retrieves a single conversation thread with its messages."""
        try:
            result = await db.execute(
                select(Conversation)
                .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
                .options(selectinload(Conversation.messages))
            )
            conv = result.scalars().first()
            if not conv:
                raise NotFoundError(message="Conversation thread not found")
            return ConversationResponse.model_validate(conv)
        except AppException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            raise ValidationError(message="Failed to load thread details.")
