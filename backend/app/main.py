import os
import json
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from app import models, schemas, database, config, parser, rag, llm

# Initialize DB tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="DocuMind AI API", version="1.0.0")

# Enable CORS for React frontend (default local ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper: Extract document metadata mapping for citations
def get_doc_metadata_mapping(db: Session, doc_ids: List[str]):
    docs = db.query(models.Document).filter(models.Document.id.in_(doc_ids)).all()
    return {doc.id: doc.filename for doc in docs}


@app.post("/api/documents/upload", response_model=schemas.DocumentResponse)
async def upload_document(
    file: UploadFile = File(...), 
    db: Session = Depends(database.get_db)
):
    """
    Uploads a document (PDF, DOCX, TXT, MD), parses it, chunks it,
    generates embeddings, and indexes it in FAISS.
    """
    # 1. Save uploaded file to local disk
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    file_path = os.path.join(config.UPLOAD_DIR, f"{asyncio.get_event_loop().time()}_{safe_filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    file_size = os.path.getsize(file_path)
    file_ext = os.path.splitext(file.filename)[1]
    
    # 2. Save metadata to DB (marked as not processed yet)
    db_doc = models.Document(
        filename=file.filename,
        filepath=file_path,
        file_type=file_ext,
        size=file_size,
        is_processed=False
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    
    try:
        # 3. Parse Document Content
        pages_content = parser.parse_document(file_path, file_ext)
        if not pages_content:
            raise HTTPException(status_code=400, detail="Document contains no extractable text.")
            
        # 4. Segment into Text Chunks
        chunks = rag.split_text_into_chunks(pages_content)
        
        # 5. Build FAISS index and save vector DB
        rag.build_and_save_index(db_doc.id, chunks)
        
        # 6. Set processed true in DB
        db_doc.is_processed = True
        db.commit()
        db.refresh(db_doc)
        
    except Exception as e:
        # Clean up files on error
        if os.path.exists(file_path):
            os.remove(file_path)
        db.delete(db_doc)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
    return db_doc


@app.get("/api/documents", response_model=List[schemas.DocumentResponse])
def get_documents(db: Session = Depends(database.get_db)):
    """
    Lists all uploaded and processed documents.
    """
    return db.query(models.Document).filter(models.Document.is_processed == True).all()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(database.get_db)):
    """
    Deletes a document from the system (SQL, raw file, and vector index).
    """
    db_doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not db_doc:
        raise HTTPException(status_code=440, detail="Document not found.")
        
    # Delete local raw file
    if os.path.exists(db_doc.filepath):
        try:
            os.remove(db_doc.filepath)
        except Exception as e:
            print(f"Failed to delete raw file {db_doc.filepath}: {e}")
            
    # Delete FAISS indices
    rag.delete_document_index(db_doc.id)
    
    # Delete SQL references (many-to-many handles cascade, or SQLite CASCADE deletes relations)
    db.delete(db_doc)
    db.commit()
    
    return {"message": f"Successfully deleted document {document_id}"}


@app.post("/api/chats", response_model=schemas.ChatSessionResponse)
def create_chat_session(
    chat_data: schemas.ChatSessionCreate, 
    db: Session = Depends(database.get_db)
):
    """
    Creates a new chat session and optionally binds it to a list of existing documents.
    """
    db_session = models.ChatSession(title="New Chat")
    
    # Bind initial documents if provided
    if chat_data.document_ids:
        docs = db.query(models.Document).filter(models.Document.id.in_(chat_data.document_ids)).all()
        db_session.documents = docs
        
        # Set chat title to document name if there is only 1 document
        if len(docs) == 1:
            db_session.title = f"Chat with {docs[0].filename[:25]}"
            
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


@app.get("/api/chats", response_model=List[schemas.ChatSessionResponse])
def get_chat_sessions(db: Session = Depends(database.get_db)):
    """
    Returns all existing chat sessions, ordered by most recently updated first.
    """
    return db.query(models.ChatSession).order_by(models.ChatSession.updated_at.desc()).all()


@app.get("/api/chats/{session_id}/messages", response_model=List[schemas.MessageResponse])
def get_chat_messages(session_id: str, db: Session = Depends(database.get_db)):
    """
    Retrieves history of messages in a given chat session.
    """
    db_session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return db_session.messages


@app.post("/api/chats/{session_id}/query")
async def chat_with_documents(
    session_id: str,
    payload: schemas.QueryRequest,
    db: Session = Depends(database.get_db)
):
    """
    Performs RAG by pulling matching vector chunks from linked documents,
    compiles a system instruction, feeds conversation memory, and streams
    the answer using Server-Sent Events (SSE).
    """
    # 1. Retrieve the session from SQLite
    db_session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
        
    # 2. Determine which documents to chat with
    # Use overriding payload.document_ids if present; otherwise fall back to session-bound documents
    target_doc_ids = payload.document_ids
    if target_doc_ids is None:
        target_doc_ids = [doc.id for doc in db_session.documents]
        
    if not target_doc_ids:
        # If no documents are bound to the session, we will query against all uploaded documents by default
        all_docs = db.query(models.Document).all()
        target_doc_ids = [doc.id for doc in all_docs]
        
    if not target_doc_ids:
        raise HTTPException(
            status_code=400, 
            detail="No documents available to query. Please upload a document first."
        )
        
    # 3. Retrieve relevant vector database chunks
    doc_mapping = get_doc_metadata_mapping(db, target_doc_ids)
    retrieved_chunks = rag.retrieve_relevant_chunks(
        query=payload.prompt,
        document_ids=target_doc_ids,
        top_k=config.RETRIEVAL_TOP_K,
        doc_metadata_mapping=doc_mapping
    )
    
    # 4. Get previous messages for conversation memory
    history = []
    for msg in db_session.messages:
        history.append({
            "role": msg.role,
            "content": msg.content
        })
        
    # 5. Format standard RAG LLM prompts
    llm_messages = llm.format_rag_prompt(
        query=payload.prompt,
        retrieved_chunks=retrieved_chunks,
        history=history
    )
    
    # 6. Store User query in database
    user_message = models.Message(
        session_id=session_id,
        role="user",
        content=payload.prompt
    )
    db.add(user_message)
    import datetime
    db_session.updated_at = datetime.datetime.utcnow()
    db.commit()
    
    # 7. Compile the citation response payload
    citations_data = [
        {
            "doc_id": c["doc_id"],
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "page": c["page"],
            "score": c["score"]
        }
        for c in retrieved_chunks
    ]
    
    # 8. Create full background task/stream to aggregate tokens and commit Assistant response to SQLite
    async def sse_generator():
        # First SSE packet contains the list of citations/source chunks
        yield f"data: {json.dumps({'citations': citations_data})}\n\n"
        
        # Second, select model provider and stream tokens
        full_response_text = ""
        
        if payload.model_provider == "extractive":
            # Offline mode: return document chunks directly, no model needed
            generator = llm.stream_extractive_response(payload.prompt, retrieved_chunks)

        elif payload.model_provider == "ollama":
            generator = llm.stream_ollama_response(llm_messages, payload.model_name)

        else:
            try:
                generator = llm.stream_huggingface_response(llm_messages, payload.model_name or config.HF_LLM_MODEL)
                # Eagerly trigger the generator to catch load errors early
                async for sse_line in generator:
                    yield sse_line
                    if sse_line.startswith("data: "):
                        data_str = sse_line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        data_obj = json.loads(data_str)
                        if "error" in data_obj:
                            raise Exception(data_obj["error"])
                        if "token" in data_obj:
                            full_response_text += data_obj["token"]
                            
                # Save and return — skip the generic loop below
                if full_response_text.strip():
                    assistant_message = models.Message(
                        session_id=session_id,
                        role="assistant",
                        content=full_response_text,
                        citations=json.dumps(citations_data)
                    )
                    db_generator = database.SessionLocal()
                    try:
                        db_generator.add(assistant_message)
                        db_session_gen = db_generator.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
                        if db_session_gen and db_session_gen.title == "New Chat":
                            smart_title = payload.prompt[:30] + ("..." if len(payload.prompt) > 30 else "")
                            db_session_gen.title = smart_title
                        db_generator.commit()
                    except Exception as commit_err:
                        print(f"Failed to save assistant message: {commit_err}")
                    finally:
                        db_generator.close()
                return

            except Exception as hf_err:
                print(f"HuggingFace model failed ({hf_err}), falling back to extractive mode.")
                yield f"data: {json.dumps({'token': '⚠️ Hugging Face model not available (connection failed or rate limited). Falling back to Offline Extractive mode:\\n\\n'})}\n\n"
                
                # Stream the extractive response chunks
                generator = llm.stream_extractive_response(payload.prompt, retrieved_chunks)
                async for sse_line in generator:
                    yield sse_line
                    if sse_line.startswith("data: "):
                        data_str = sse_line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            data_obj = json.loads(data_str)
                            if "token" in data_obj:
                                full_response_text += data_obj["token"]
                        except Exception:
                            pass
                            
                # Save assistant message to SQLite
                if full_response_text.strip():
                    assistant_message = models.Message(
                        session_id=session_id,
                        role="assistant",
                        content=full_response_text,
                        citations=json.dumps(citations_data)
                    )
                    db_generator = database.SessionLocal()
                    try:
                        db_generator.add(assistant_message)
                        db_session_gen = db_generator.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
                        if db_session_gen and db_session_gen.title == "New Chat":
                            smart_title = payload.prompt[:30] + ("..." if len(payload.prompt) > 30 else "")
                            db_session_gen.title = smart_title
                        db_generator.commit()
                    except Exception as commit_err:
                        print(f"Failed to save assistant message: {commit_err}")
                    finally:
                        db_generator.close()
                return
            
        async for sse_line in generator:
            yield sse_line
            
            # Extract token to accumulate complete text
            if sse_line.startswith("data: "):
                data_str = sse_line[6:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    data_obj = json.loads(data_str)
                    if "token" in data_obj:
                        full_response_text += data_obj["token"]
                except Exception:
                    pass
                    
        # Once complete, save the Assistant's reply with JSON citations to DB
        if full_response_text.strip():
            assistant_message = models.Message(
                session_id=session_id,
                role="assistant",
                content=full_response_text,
                citations=json.dumps(citations_data)
            )
            db_generator = database.SessionLocal()
            try:
                db_generator.add(assistant_message)
                db_session_gen = db_generator.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
                if db_session_gen and db_session_gen.title == "New Chat":
                    smart_title = payload.prompt[:30] + ("..." if len(payload.prompt) > 30 else "")
                    db_session_gen.title = smart_title
                db_generator.commit()
            except Exception as commit_err:
                print(f"Failed to save assistant message: {commit_err}")
            finally:
                db_generator.close()
                
    return StreamingResponse(sse_generator(), media_type="text/event-stream")



@app.put("/api/chats/{session_id}", response_model=schemas.ChatSessionResponse)
def update_chat_session(
    session_id: str,
    payload: schemas.ChatSessionUpdate,
    db: Session = Depends(database.get_db)
):
    """
    Renames the title of a chat session.
    """
    db_session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    db_session.title = payload.title
    db.commit()
    db.refresh(db_session)
    return db_session


@app.delete("/api/chats/{session_id}")
def delete_chat_session(session_id: str, db: Session = Depends(database.get_db)):
    """
    Deletes an entire chat session and its nested message history.
    """
    db_session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    db.delete(db_session)
    db.commit()
    return {"message": f"Successfully deleted chat session {session_id}"}

# Serve React single page client directly from the backend
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h2>DocuMind AI Frontend Static File not found. Make sure backend/app/static/index.html exists.</h2>", status_code=404)
