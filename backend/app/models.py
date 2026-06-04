import datetime
import uuid
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey, Table
from sqlalchemy.orm import relationship
from app.database import Base

# Many-to-many association table linking chat sessions to the documents they chat with
chat_session_documents = Table(
    "chat_session_documents",
    Base.metadata,
    Column("session_id", String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
)

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_processed = Column(Boolean, default=False)

    # Relationships
    sessions = relationship(
        "ChatSession", 
        secondary=chat_session_documents, 
        back_populates="documents"
    )

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    documents = relationship(
        "Document", 
        secondary=chat_session_documents, 
        back_populates="sessions"
    )
    messages = relationship(
        "Message", 
        back_populates="session", 
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    citations = Column(Text, nullable=True)  # JSON-encoded array of retrieved citation chunks
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")
