# JAMS — Judicial AI Management System

Split architecture: **Python FastAPI backend** + **Node.js React frontend**.

## Project Structure

```
judicial-ai-demo/
├── app.py              # Backend entry (uvicorn)
├── backend/
│   ├── core.py         # PDF indexing, search, local LLM
│   └── main.py         # FastAPI REST API
├── frontend/           # React + Vite UI
│   ├── src/
│   └── package.json
├── oldcode.py          # Legacy Gradio app (reference)
└── requirements.txt
```

## Setup

### 1. Python Backend

```powershell
cd E:\python\ji\judicial-ai-demo
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

API runs at **http://127.0.0.1:8000**

### 2. Node.js Frontend

```powershell
cd E:\python\ji\judicial-ai-demo\frontend
npm install
npm run dev
```

UI runs at **http://localhost:5173** (proxies `/api` to backend)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Dashboard stats |
| GET | `/api/cases` | List indexed cases |
| POST | `/api/cases/extract-metadata` | Auto-fill from PDF |
| POST | `/api/cases/upload` | Upload & index case |
| POST | `/api/chat` | Chat with optional PDF |
| GET | `/api/scraper/fccp/status` | FCCP scrape manifest & stats |
| POST | `/api/scraper/fccp/sync` | Download & index FCCP judgments |
| GET | `/api/scraper/lhc/status` | LHC scrape manifest & stats |
| POST | `/api/scraper/lhc/sync` | Fetch metadata / download LHC judgments |

## FCCP Judgments Scraper

Imports cases from [Federal Constitutional Court of Pakistan — Judgments](https://fccp.gov.pk/judgments?page=1):

1. Scrapes **case title**, **author judge**, **upload date**, **PDF download link**
2. Saves PDFs to `data/fccp/pdfs/`
3. Indexes text chunks for **AI chat** (same pipeline as manual upload)
4. Persists dataset to `data/jams_store.json` + `data/fccp/manifest.json`

**From UI:** open **FCCP Import** tab → **Sync & Index All** (pages 1–5 ≈ 46 cases)

**From CLI:**
```powershell
python scripts/run_fccp_sync.py --start 1 --end 5
```

## LHC Judgments Scraper

Imports **Judgments Approved for Reporting** from [Lahore High Court](https://data.lhc.gov.pk/reported_judgments/judgments_approved_for_reporting) (~4683 with filter **All Courts**):

1. **Step 1 — Metadata:** one API call lists all judgments (title, judge, citation, PDF URL, tag line)
2. **Step 2 — Batch download:** PDFs saved to `data/lhc/pdfs/` and indexed for chat (50 per batch by default)

**From UI:** **LHC Import** tab → **Fetch All Metadata** → **Download & Index Batch**

**From CLI:**
```powershell
python scripts/run_lhc_sync.py --metadata-only
python scripts/run_lhc_sync.py --limit 50
```

## LLM: Ollama `qwen2.5:1.5b` + LangChain RAG

Chat uses **LangChain** with **Chroma** (persisted at `data/chroma/`) and **MMR retrieval** for multi-case answers. LLM via **Ollama**.

```powershell
ollama pull qwen2.5:1.5b
ollama serve
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embeddings |
| `CHROMA_DIR` | `data/chroma` | Vector index on disk |

## Notes

- Indexed cases persist to `data/jams_store.json` (local; not in git — sync via FCCP on each server).
- Chat-attached PDFs are not added to the indexed case database.
- Legacy Gradio UI is preserved in `oldcode.py` for reference.

## Ubuntu server (production)

```bash
sudo git clone https://github.com/RehanALiBalti/judicial-ai-demo.git /opt/jams
sudo chown -R www-data:www-data /opt/jams
cd /opt/jams
sudo DOMAIN=YOUR_SERVER_IP bash deploy/ubuntu-setup.sh
```

Full guide: [deploy/DEPLOY.md](deploy/DEPLOY.md)
