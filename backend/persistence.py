"""Persistent storage for JAMS cases and FCCP scrape manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FCCP_DIR = DATA_DIR / "fccp"
FCCP_PDF_DIR = FCCP_DIR / "pdfs"
MANIFEST_PATH = FCCP_DIR / "manifest.json"

LHC_DIR = DATA_DIR / "lhc"
LHC_PDF_DIR = LHC_DIR / "pdfs"
LHC_MANIFEST_PATH = LHC_DIR / "manifest.json"

STORE_PATH = DATA_DIR / "jams_store.json"

COURT_FCCP = "Federal Constitutional Court of Pakistan"
COURT_LHC = "Lahore High Court, Lahore"


def ensure_data_dirs() -> None:
    FCCP_PDF_DIR.mkdir(parents=True, exist_ok=True)
    LHC_PDF_DIR.mkdir(parents=True, exist_ok=True)


def normalize_case_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def manifest_item_key(item: Dict[str, Any]) -> str:
    """Stable key — FCCP download URLs change every request."""
    if item.get("source_id"):
        return str(item["source_id"])
    title = normalize_case_title(item.get("case_title", ""))
    date = (item.get("upload_date") or "").strip()
    return f"fccp|{title}|{date}"


def dedupe_manifest_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge manifest rows by stable case key (keeps best local record)."""
    merged: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = manifest_item_key(item)
        item = {**item, "source_id": key}
        if key not in merged:
            merged[key] = item
            continue
        prev = merged[key]
        prev_path = prev.get("pdf_path")
        new_path = item.get("pdf_path")
        prev_ok = prev_path and Path(prev_path).is_file()
        new_ok = new_path and Path(new_path).is_file()
        if new_ok and not prev_ok:
            merged[key] = item
        elif item.get("indexed") and not prev.get("indexed"):
            merged[key] = {**prev, **item, "indexed": True, "case_id": item.get("case_id") or prev.get("case_id")}
        else:
            merged[key] = {**prev, **item, "pdf_path": prev.get("pdf_path") or item.get("pdf_path")}
    return list(merged.values())


def load_manifest() -> Dict[str, Any]:
    ensure_data_dirs()
    if not MANIFEST_PATH.exists():
        return {"items": [], "last_sync": None}
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["items"] = dedupe_manifest_items(manifest.get("items", []))
    return manifest


def save_manifest(manifest: Dict[str, Any]) -> None:
    ensure_data_dirs()
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def lhc_manifest_item_key(item: Dict[str, Any]) -> str:
    """Stable key from PDF URL or citation + case number."""
    if item.get("source_id"):
        return str(item["source_id"])
    pdf_url = (item.get("pdf_url") or "").strip()
    if pdf_url:
        stem = Path(pdf_url).name.replace(".pdf", "")
        return f"lhc|{stem}"
    citation = (item.get("lhc_citation") or "").strip()
    case_no = normalize_case_title(item.get("case_number", ""))
    return f"lhc|{citation}|{case_no}"


def load_lhc_manifest() -> Dict[str, Any]:
    ensure_data_dirs()
    if not LHC_MANIFEST_PATH.exists():
        return {"items": [], "last_sync": None, "total_reported": None}
    with LHC_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    merged: Dict[str, Dict[str, Any]] = {}
    for item in manifest.get("items", []):
        key = lhc_manifest_item_key(item)
        item = {**item, "source_id": key}
        if key not in merged:
            merged[key] = item
            continue
        prev = merged[key]
        prev_path = prev.get("pdf_path")
        new_path = item.get("pdf_path")
        prev_ok = prev_path and Path(prev_path).is_file()
        new_ok = new_path and Path(new_path).is_file()
        if new_ok and not prev_ok:
            merged[key] = item
        elif item.get("indexed") and not prev.get("indexed"):
            merged[key] = {**prev, **item, "indexed": True, "case_id": item.get("case_id") or prev.get("case_id")}
        else:
            merged[key] = {**prev, **item, "pdf_path": prev.get("pdf_path") or item.get("pdf_path")}
    manifest["items"] = list(merged.values())
    return manifest


def save_lhc_manifest(manifest: Dict[str, Any]) -> None:
    ensure_data_dirs()
    with LHC_MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_store() -> Dict[str, Any]:
    if not STORE_PATH.exists():
        return {"cases": [], "documents": []}
    with STORE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_store(cases: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump({"cases": cases, "documents": documents}, f, ensure_ascii=False)

