from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional


class ChatQuestion(BaseModel):
    question: str
    document_id: Optional[int] = None  # Query specifically against this document


class ChatResponse(BaseModel):
    status: str = "ok"
    answer: str
    pages: List[int] = []


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    pages: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    id: int
    title: Optional[str] = None
    document_id: Optional[int] = None
    created_at: datetime
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True
