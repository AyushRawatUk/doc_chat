from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Document Schemas
class DocumentBase(BaseModel):
    filename: str
    file_type: str
    size: int

class DocumentResponse(DocumentBase):
    id: str
    created_at: datetime
    is_processed: bool

    class Config:
        from_attributes = True

# Message Schemas
class MessageBase(BaseModel):
    role: str
    content: str
    citations: Optional[str] = None  # JSON string

class MessageCreate(MessageBase):
    pass

class MessageResponse(MessageBase):
    id: str
    session_id: str
    created_at: datetime

    class Config:
        from_attributes = True

# Chat Session Schemas
class ChatSessionBase(BaseModel):
    title: str

class ChatSessionCreate(BaseModel):
    document_ids: Optional[List[str]] = Field(default_factory=list)

class ChatSessionUpdate(BaseModel):
    title: str

class ChatSessionResponse(ChatSessionBase):
    id: str
    created_at: datetime
    updated_at: datetime
    documents: List[DocumentResponse] = []

    class Config:
        from_attributes = True

# RAG Query Schemas
class QueryRequest(BaseModel):
    prompt: str
    model_provider: str = "ollama"  # "openai" or "ollama"
    model_name: Optional[str] = None  # e.g., "gpt-4o", "llama3"
    document_ids: Optional[List[str]] = None  # overrides default session documents if provided
    openai_api_key: Optional[str] = None  # allows client to supply the key dynamically

class CitationChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: float
    page: Optional[int] = None
