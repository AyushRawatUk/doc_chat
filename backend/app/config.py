import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Directories configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indices"

# Create directories if they do not exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/documind.db")

# LLM Providers Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Embeddings Configuration
# For custom trained models:
# You can set EMBEDDING_MODEL_PATH to a local directory path (e.g., "C:/path/to/my/trained/model")
# or a Hugging Face model repository name (e.g., "sentence-transformers/all-MiniLM-L6-v2").
# Our RAG pipeline will automatically load it.
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH", 
    r"C:\Users\Ayush RAwat\.gemini\antigravity\scratch\doc-chat-app\backend\model\best_modern_bert.pth"
)

# RAG Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
