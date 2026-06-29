"""JAMS FastAPI REST backend."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend import core
from backend.scraper import fccp as fccp_scraper
from backend.scraper import lhc as lhc_scraper

_sync_status: Dict[str, Any] = {"running": False, "last_result": None}
_lhc_sync_status: Dict[str, Any] = {"running": False, "last_result": None}

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
).split(",")


async def save_upload(upload: UploadFile) -> tuple[str, str]:
    """Save upload to a temp file; returns (path, original_filename)."""
    suffix = os.path.splitext(upload.filename or "file.pdf")[1] or ".pdf"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    content = await upload.read()
    with open(path, "wb") as f:
        f.write(content)
    return path, upload.filename or os.path.basename(path)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(
    title="JAMS API",
    description="Judicial AI Management System — Python backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    ollama = core.check_ollama()
    return {
        "status": "ok",
        "service": "jams-api",
        "llm": "ollama",
        "ollama_model": core.OLLAMA_MODEL,
        "ollama": ollama,
    }


@app.get("/api/stats")
def stats():
    return core.get_dashboard_stats()


@app.get("/api/cases")
def get_cases():
    return {"cases": core.list_cases(), "stats": core.get_dashboard_stats()}


@app.post("/api/cases/extract-metadata")
async def extract_metadata(file: UploadFile = File(...)):
    path, name = await save_upload(file)
    try:
        return core.auto_fill_metadata_from_path(path, name)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/api/cases/upload")
async def upload_case(
    file: UploadFile = File(...),
    case_title: str = Form(...),
    court_name: str = Form(...),
    decision_date: str = Form(...),
):
    path, name = await save_upload(file)
    try:
        result = core.upload_case_from_path(path, name, case_title, court_name, decision_date)
        if result.get("success"):
            result["stats"] = core.get_dashboard_stats()
        return result
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _run_fccp_sync(start_page: int, end_page: Optional[int]) -> None:
    global _sync_status
    try:
        result = fccp_scraper.sync_fccp_judgments(
            start_page=start_page,
            end_page=end_page,
            auto_index=True,
            index_callback=core.index_fccp_judgment,
        )
        result["stats"] = core.get_dashboard_stats()
        _sync_status["last_result"] = result
    except Exception as exc:
        _sync_status["last_result"] = {"error": str(exc)}
    finally:
        _sync_status["running"] = False


@app.get("/api/scraper/fccp/status")
def fccp_status():
    status = fccp_scraper.get_fccp_status()
    status["sync_running"] = _sync_status["running"]
    status["last_sync_result"] = _sync_status.get("last_result")
    status["stats"] = core.get_dashboard_stats()
    return status


@app.post("/api/scraper/fccp/sync")
def fccp_sync(
    background_tasks: BackgroundTasks,
    start_page: int = 1,
    end_page: Optional[int] = None,
):
    if _sync_status["running"]:
        return {"success": False, "message": "FCCP sync already running."}
    _sync_status["running"] = True
    _sync_status["last_result"] = None
    background_tasks.add_task(_run_fccp_sync, start_page, end_page)
    return {
        "success": True,
        "message": "FCCP sync started in background.",
        "start_page": start_page,
        "end_page": end_page,
    }


def _run_lhc_sync(
    year: str,
    court_name: str,
    metadata_only: bool,
    download_limit: Optional[int],
) -> None:
    global _lhc_sync_status
    try:
        result = lhc_scraper.sync_lhc_judgments(
            year=year,
            court_name=court_name,
            metadata_only=metadata_only,
            auto_index=not metadata_only,
            download_limit=download_limit,
            index_callback=core.index_lhc_judgment,
        )
        result["stats"] = core.get_dashboard_stats()
        _lhc_sync_status["last_result"] = result
    except Exception as exc:
        _lhc_sync_status["last_result"] = {"error": str(exc)}
    finally:
        _lhc_sync_status["running"] = False


@app.get("/api/scraper/lhc/status")
def lhc_status():
    status = lhc_scraper.get_lhc_status()
    status["sync_running"] = _lhc_sync_status["running"]
    status["last_sync_result"] = _lhc_sync_status.get("last_result")
    status["stats"] = core.get_dashboard_stats()
    return status


@app.post("/api/scraper/lhc/sync")
def lhc_sync(
    background_tasks: BackgroundTasks,
    year: str = "",
    court_name: str = "All Courts",
    metadata_only: bool = False,
    download_limit: Optional[int] = 50,
):
    if _lhc_sync_status["running"]:
        return {"success": False, "message": "LHC sync already running."}
    _lhc_sync_status["running"] = True
    _lhc_sync_status["last_result"] = None
    background_tasks.add_task(_run_lhc_sync, year, court_name, metadata_only, download_limit)
    return {
        "success": True,
        "message": "LHC sync started in background.",
        "metadata_only": metadata_only,
        "download_limit": download_limit,
        "court_name": court_name,
        "year": year or "Any",
    }


@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(""),
    history: str = Form("[]"),
    temp_docs: str = Form("[]"),
    file: Optional[UploadFile] = File(None),
):
    parsed_history: List[Dict[str, str]] = json.loads(history or "[]")
    parsed_temp_docs: List[Dict[str, Any]] = json.loads(temp_docs or "[]")

    pdf_path = None
    pdf_filename = None
    if file and file.filename:
        pdf_path, pdf_filename = await save_upload(file)

    try:
        result = core.chat(
            message=message,
            history=parsed_history,
            temp_docs=parsed_temp_docs,
            pdf_path=pdf_path,
            pdf_filename=pdf_filename,
        )
        result["stats"] = core.get_dashboard_stats()
        return result
    finally:
        if pdf_path:
            try:
                os.remove(pdf_path)
            except OSError:
                pass
