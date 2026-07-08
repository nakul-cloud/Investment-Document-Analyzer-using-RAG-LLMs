from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class DocumentResponse(BaseModel):
    id: int
    filename: str
    pages_count: int
    created_at: datetime
    owner_id: int

    class Config:
        from_attributes = True
