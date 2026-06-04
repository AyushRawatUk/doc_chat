# DocuMind AI — Chat with Your Documents

**DocuMind AI** is an intelligent, high-fidelity "Chat with Your Documents" assistant (similar to ChatPDF and NotebookLM) designed to run entirely locally, or connect with secure cloud models. 

Users can upload documents in various formats (**PDF, DOCX, TXT, Markdown**), compile them automatically into local vector indexes, and query them with visual citation click-backs, multi-document aggregations, side-by-side text viewing, and full chat session memory.

---

## 🏗️ Project Architecture

DocuMind uses a modular, decoupled **Retrieval-Augmented Generation (RAG)** layout:

```
                  +----------------------------------------------+
                  |               React Web UI                   |
                  |  - Drag & Drop Document Upload               |
                  |  - Multi-Document Active Query Filter        |
                  |  - Split-Screen Side-by-Side Reading Layout  |
                  |  - SSE Streaming Chat Responses              |
                  |  - Collapsible Citation Highlighters         |
                  +----------------------------------------------+
                                         ^
                                         | REST / SSE (Streams)
                                         v
                  +----------------------------------------------+
                  |               FastAPI Server                 |
                  |  - Upload, Session Management, CRUD Routers  |
                  |  - RAG Prompt Assembly (History + Memory)    |
                  |  - Server-Sent Events (SSE) Stream Compiler  |
                  +----------------------------------------------+
                      |                      |                |
                      v                      v                v
            +-------------------+  +------------------+  +-------------+
            |  SQLite Metadata  |  |   Parsers Engine |  |  LLM Layer  |
            |  - Documents      |  |  - pdfplumber    |  |  - OpenAI   |
            |  - Chat Sessions  |  |  - python-docx   |  |  - Ollama   |
            |  - Messages       |  |  - TXT / MD      |  |   (Local)   |
            +-------------------+  +------------------+  +-------------+
                                             |
                                             v
                                   +------------------+
                                   |  Vector DB RAG   |
                                   |  - SentenceTrans |
                                   |  - FAISS CPU     |
                                   +------------------+
```

---

## 🛠️ Tech Stack Spec Sheet

* **Frontend**: React (Single Page Client using Tailwind CSS, Lucide icons, and Babel in-browser compiler for zero-dependency instant launches).
* **Backend**: Python 3.8+ with **FastAPI** & **Uvicorn** for async high-efficiency operations.
* **Vector Index**: **FAISS** (Flat L2 index, saved to disk per-document for dynamic search modularity).
* **Local Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` loaded locally (384-dimensional dense vectors, zero API cost).
* **File Parsers**: `pdfplumber` (PDF layout & tables), `pypdf` (fallback PDF extractor), `python-docx` (Word paragraph grouping).
* **Conversational Cache**: **SQLite** via **SQLAlchemy ORM** (maintains sessions, chat transcripts, and JSON citation logs).

---

## ⚡ Quick Start: Running the App in 2 Minutes!

Since the frontend is bundled directly as a compiled SPA served from the FastAPI backend, you **only need Python** to run the complete, fully operational software suite!

### Step 1: Clone or Open Workspace
Ensure you are in the project folder:
```powershell
cd doc-chat-app/backend
```

### Step 2: Setup Environment and Install Dependencies
Create a Python virtual environment and run the package installer:
```powershell
# Create environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Run the Application!
Start the unified server launcher:
```powershell
python run.py
```

### Step 4: Access in Browser
Open your browser and navigate to:
👉 **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

*The stunning ChatGPT-like dark-themed UI will appear instantly. You can drag and drop a PDF, choose a model, and start chatting with your content immediately!*

---

## 🧠 Using Your Own Trained Embedding Model

If you have trained/fine-tuned your own embedding model, DocuMind allows you to plug it in seamlessly:

1. **Save your trained weights**: Ensure your model is saved in standard Hugging Face/SentenceTransformer format to a local folder (e.g., `C:/models/my-custom-embeddings/`). The folder should contain `config.json`, `pytorch_model.bin` (or safetensors), and vocabulary files.
2. **Configure your `.env` file**: Create a file named `.env` in the `backend/` directory:
   ```env
   EMBEDDING_MODEL_PATH=C:/models/my-custom-embeddings/
   ```
3. **Automatic Hot-Swap**: When the backend starts up, it reads `EMBEDDING_MODEL_PATH` and loads your local trained model directly using `SentenceTransformer("C:/models/my-custom-embeddings/")` instead of downloading standard Hugging Face models.

---

## 🔌 API Endpoint Documentation

| Method | Endpoint | Description | Payload / Response |
| :--- | :--- | :--- | :--- |
| **POST** | `/api/documents/upload` | Upload & parse document, split chunks, embed in FAISS | Multipart File -> Document ID |
| **GET** | `/api/documents` | List all processed documents | Array of processed files metadata |
| **DELETE** | `/api/documents/{id}` | Delete raw file, DB entries, and FAISS index files | confirmation JSON |
| **POST** | `/api/chats` | Create new chat session, optionally bound to document IDs | Session metadata JSON |
| **GET** | `/api/chats` | Get past chat sessions list ordered by recent activity | Array of active sessions |
| **GET** | `/api/chats/{id}/messages` | Retrieve full dialogue transcript and citations | Array of chat messages |
| **POST** | `/api/chats/{id}/query` | Core RAG streaming route using Server-Sent Events | SSE Event Stream |
| **PUT** | `/api/chats/{id}` | Rename chat session title | Update title JSON |

---

## 💾 Database Schema (SQLite ORM)

```sql
CREATE TABLE documents (
    id VARCHAR PRIMARY KEY,
    filename VARCHAR NOT NULL,
    filepath VARCHAR NOT NULL,
    file_type VARCHAR NOT NULL,
    size INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE chat_sessions (
    id VARCHAR PRIMARY KEY,
    title VARCHAR,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME
);

CREATE TABLE chat_session_documents (
    session_id VARCHAR REFERENCES chat_sessions(id) ON DELETE CASCADE,
    document_id VARCHAR REFERENCES documents(id) ON DELETE CASCADE,
    PRIMARY KEY (session_id, document_id)
);

CREATE TABLE messages (
    id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    citations TEXT, -- JSON-serialized array of source segment details
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 📈 Scalability and Performance Best Practices

To scale DocuMind to support hundreds of large documents or high-concurrency enterprise use:

1. **Swap to FAISS GPU or IVF Indexes**:
   - For indexes with millions of chunks, change standard flat FAISS indexes (`IndexFlatL2`) to cell probe indexes like `IndexIVFFlat` inside `rag.py`. This uses clustering to limit comparisons and scales search time logarithmically rather than linearly.
2. **Move Vector Indexes to a Dedicated Vector DB**:
   - If deploying to a multi-container environment, migrate FAISS to **ChromaDB**, **Milvus**, or **Pinecone** to avoid relying on a local shared network file system.
3. **Decouple Document Parsing with Celery**:
   - Document upload, text parsing, and indexing can take 10-30 seconds for 500-page textbooks. Move these heavy computations out of the main FastAPI thread by using a task queue like **Celery** with Redis as a broker.
4. **Sentence-Transformers Inference Optimization**:
   - Use **ONNX Runtime** or **TensorRT** to compile the embedding model. This speeds up semantic searches by 3-5x on CPU, and reduces RAM footprint by up to 60%.
5. **Database Indexing**:
   - Add database indexes on `session_id` in the `messages` table and `session_id`/`document_id` in the association tables to ensure database search performance remains fast as millions of chat messages accumulate.
