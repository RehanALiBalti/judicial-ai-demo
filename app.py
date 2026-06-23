"""
JAMS — Judicial AI Management System
Python backend entry point (FastAPI).

Run:
  python app.py

Frontend (Node.js) runs separately:
  cd frontend && npm install && npm run dev
"""

import os

import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
