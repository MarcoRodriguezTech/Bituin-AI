"""Central configuration for Bituin AI.

All values can be overridden via environment variables or a local `.env` file.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Storage ---
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "raw_cache.db"))
CHROMA_PATH = os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma_db"))
MOCK_GMAIL_PATH = os.getenv("MOCK_GMAIL_PATH", str(DATA_DIR / "mock_gmail_data.json"))

# --- Ollama / LLM ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "384"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "4"))

# --- Voice ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
# Pre-downloaded local copy (bypasses HuggingFace CDN if blocked)
WHISPER_LOCAL_DIR = Path(
    os.getenv("WHISPER_LOCAL_DIR", str(DATA_DIR / "models" / "faster-whisper-base.en"))
)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

# --- Gmail ---
GMAIL_CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", str(BASE_DIR / "credentials.json"))
GMAIL_TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", str(DATA_DIR / "token.json"))
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
# Which inbox tab to sync: primary | promotions | social | updates | forums | all
GMAIL_CATEGORY = os.getenv("GMAIL_CATEGORY", "primary").strip().lower()
# How many of the latest matching messages to pull per sync (read or unread)
GMAIL_MAX_RESULTS = int(os.getenv("GMAIL_MAX_RESULTS", "100"))
# Only sync messages received within this many days (0 = no age limit)
GMAIL_MAX_AGE_DAYS = int(os.getenv("GMAIL_MAX_AGE_DAYS", "0"))
# Extra Gmail search filter on top, e.g. "-from:notifications@github.com"
# or "is:unread" to go back to unread-only
# (note: the category: operator does not work in q — use GMAIL_CATEGORY)
GMAIL_QUERY = os.getenv("GMAIL_QUERY", "").strip()
