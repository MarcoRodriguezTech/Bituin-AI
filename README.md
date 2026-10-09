# Bituin AI

A privacy-first local AI assistant that answers questions about your Gmail
inbox, triages urgency, extracts to-dos, and writes a daily briefing — all
on-device. Built for the AppBuildersPH Hackathon.

**Everything runs on-device** — LLM inference, speech-to-text, and embeddings never
touch the cloud. A mock-data fallback keeps the demo fully functional with Wi-Fi off.

## Architecture

```
Gmail API (OAuth) ──► Python connector ──► SQLite cache ──► ChromaDB
                                                              │
Mic ──► faster-whisper (CUDA INT8) ──► query text ──► Ollama (Qwen 2.5 3B) ──► Streamlit UI
```

- **LLM**: Ollama `qwen2.5:3b` (~2.2 GB VRAM)
- **STT**: `faster-whisper` `base.en`, CTranslate2 INT8 on CUDA (~0.6 GB VRAM)
- **Embeddings**: `all-MiniLM-L6-v2` via ChromaDB ONNX (~0.3 GB)
- **Context cap**: 8,192 tokens

## Setup

```powershell
git clone https://github.com/MarcoRodriguezTech/Bituin-AI.git
cd Bituin-AI
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt   # exact tested versions

# Local LLM (Ollama desktop app already runs the server)
ollama pull qwen2.5:3b-instruct

# Optional: live Gmail access (mock data used otherwise)
# Google Cloud Console -> enable Gmail API -> OAuth consent screen (External,
# add yourself as test user) -> Credentials -> OAuth client ID (Desktop app)
# -> download JSON and save as credentials.json in this folder.
```

## Run

```powershell
.venv\Scripts\streamlit run app.py
```

Click **Sync Gmail inbox** in the sidebar (first live sync opens a Google
sign-in tab; the token is cached in `data/token.json`), then use the tabs:
**Ask** (chat + mic input, streaming answers), **To-do** (extracted action
items), **Inbox** (urgency-triaged), **Digest** (one-click briefing).
Works fully offline once models are pulled.

> **Note on model downloads:** HuggingFace's CDN may be unreachable on some
> networks. The whisper `base.en` weights are vendored under
> `data/models/faster-whisper-base.en/` (fetched from the ModelScope mirror),
> and embeddings use ChromaDB's ONNX all-MiniLM-L6-v2 (S3-hosted), so no
> HuggingFace access is needed at runtime.

## Project layout

```
config/settings.py          Central env-driven config
data/mock_*.json            Offline fallback inbox data
src/connectors/             Gmail (OAuth + gmail.readonly) connector
src/database/db_manager.py  SQLite cache with SHA-256 dedup
src/database/vector_store.py  ChromaDB index + retrieval
src/rag/embedder.py         Chunking (~384 tok, 50 overlap) + embedding fn
src/rag/llm_chain.py        Ollama RAG chain (strict context grounding)
src/voice/whisper_listener.py  faster-whisper push-to-talk STT
app.py                      Streamlit dashboard
```

## Why local?

Academic communications are sensitive student data. Running inference locally
guarantees zero data leaves the device, eliminates API latency/cost, and keeps
the product fully usable in campus Wi-Fi dead zones.

## Hackathon submission

**Team:** Jhon Marco Rodriguez (@MarcoRodriguezTech) — solo

### What runs locally

- LLM inference (Ollama `qwen2.5:3b-instruct`) — Q&A, urgency triage, task
  extraction, daily briefing
- Speech-to-text (faster-whisper `base.en`, CTranslate2 INT8 on CUDA)
- Embeddings + vector search (`all-MiniLM-L6-v2`, ChromaDB ONNX runtime)
- All storage: SQLite cache + ChromaDB index on-disk

### What requires internet

- Gmail API fetch (optional — the app ships with a mock inbox and remains
  fully functional offline; every AI feature works with Wi-Fi off)
- One-time model downloads (`ollama pull`, whisper weights, ONNX embedder)

### APIs and cloud services

- Google Gmail API (`gmail.readonly` scope) — inbox ingestion only, no cloud
  AI inference anywhere in the pipeline

### Models used

| Purpose | Model | Runtime |
|---|---|---|
| LLM | Qwen 2.5 3B Instruct | Ollama |
| STT | faster-whisper base.en | CTranslate2, CUDA INT8 |
| Embeddings | all-MiniLM-L6-v2 | ChromaDB ONNX |

### Technologies and frameworks

Streamlit · Ollama · ChromaDB · faster-whisper / CTranslate2 ·
sentence-transformers · google-api-python-client + OAuth · sounddevice ·
SQLite

### Existing code and assets

- Application code written during the hackathon
- `data/mock_gmail_data.json` — hand-authored demo inbox (no real email data)
- Nunito font family (`static/fonts/`) — SIL Open Font License
- Logo and avatar images in `assets/` — created for this project

### AI development tools

- Devin (Cognition) — AI-assisted development
