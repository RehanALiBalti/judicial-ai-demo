"""
Scraper for Federal Constitutional Court of Pakistan judgments.
Source: https://fccp.gov.pk/judgments
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from backend.persistence import (
    COURT_FCCP,
    FCCP_PDF_DIR,
    ensure_data_dirs,
    load_manifest,
    manifest_item_key,
    save_manifest,
)

BASE_URL = "https://fccp.gov.pk"
JUDGMENTS_URL = f"{BASE_URL}/judgments"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_HEADERS = {"User-Agent": USER_AGENT}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(REQUEST_HEADERS)
    return s


def _slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip()).strip("-").lower()
    return slug[:max_len] or "judgment"


def stable_pdf_filename(case_title: str) -> str:
    return f"fccp-{_slugify(case_title)}.pdf"


def resolve_local_pdf(record: Dict[str, Any], case_title: str) -> Optional[str]:
    """Find already-downloaded PDF on disk (manifest path or stable filename)."""
    pdf_path = record.get("pdf_path")
    if pdf_path and os.path.isfile(pdf_path):
        return pdf_path

    stable_path = FCCP_PDF_DIR / stable_pdf_filename(case_title)
    if stable_path.is_file():
        return str(stable_path)

    pattern = f"fccp-{_slugify(case_title)}*.pdf"
    matches = sorted(FCCP_PDF_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return str(matches[0])
    return None


def parse_judgments_page(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    items: List[Dict[str, Any]] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        case_title = tds[0].get_text(" ", strip=True)
        if not case_title:
            continue
        author_judge = tds[1].get_text(" ", strip=True)
        upload_date = tds[2].get_text(" ", strip=True)
        link = tr.find("a", href=True)
        if not link:
            continue
        download_url = urljoin(BASE_URL, link["href"])
        source_id = manifest_item_key({"case_title": case_title, "upload_date": upload_date})
        items.append({
            "case_title": case_title,
            "author_judge": author_judge,
            "upload_date": upload_date,
            "download_url": download_url,
            "source": "fccp",
            "source_id": source_id,
            "court": COURT_FCCP,
        })
    return items


def get_total_pages(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    pages = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "judgments?page=" in href:
            m = re.search(r"page=(\d+)", href)
            if m:
                pages.append(int(m.group(1)))
    return max(pages) if pages else 1


def fetch_judgments_page(page: int = 1) -> tuple[List[Dict[str, Any]], int]:
    session = _session()
    url = f"{JUDGMENTS_URL}?page={page}"
    resp = session.get(url, timeout=45)
    resp.raise_for_status()
    items = parse_judgments_page(resp.text)
    total_pages = get_total_pages(resp.text)
    return items, total_pages


def download_pdf(download_url: str, dest_path: str) -> bool:
    session = _session()
    resp = session.get(download_url, timeout=120, stream=True)
    resp.raise_for_status()
    content_type = (resp.headers.get("content-type") or "").lower()
    data = resp.content
    if not data:
        return False
    if "pdf" not in content_type and not data[:4] == b"%PDF":
        return False
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(data)
    return True


def sync_fccp_judgments(
    start_page: int = 1,
    end_page: Optional[int] = None,
    auto_index: bool = True,
    delay_seconds: float = 0.8,
    index_callback: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Scrape FCCP judgments, download PDFs once, store locally, index for chat.
    Re-runs skip cases already on disk / in manifest (FCCP URLs change each visit).
    """
    ensure_data_dirs()
    manifest = load_manifest()
    existing = {manifest_item_key(i): i for i in manifest.get("items", [])}

    session_report = {
        "scraped": 0,
        "downloaded": 0,
        "skipped_existing": 0,
        "skipped_indexed": 0,
        "indexed": 0,
        "failed": [],
        "pages_processed": [],
    }

    _, detected_pages = fetch_judgments_page(1)
    last_page = end_page or detected_pages
    last_page = min(last_page, detected_pages)

    for page in range(start_page, last_page + 1):
        try:
            items, _ = fetch_judgments_page(page)
        except Exception as exc:
            session_report["failed"].append({"page": page, "error": str(exc)})
            continue

        session_report["pages_processed"].append(page)

        for item in items:
            session_report["scraped"] += 1
            key = manifest_item_key(item)
            prior = existing.get(key, {})
            local_pdf = resolve_local_pdf(prior, item["case_title"])

            if local_pdf:
                session_report["skipped_existing"] += 1
                record = {
                    **prior,
                    **item,
                    "source_id": key,
                    "pdf_path": local_pdf,
                    "file_name": os.path.basename(local_pdf),
                }
                if prior.get("indexed") and prior.get("case_id"):
                    session_report["skipped_indexed"] += 1
                    record["indexed"] = True
                    record["case_id"] = prior.get("case_id")
                    existing[key] = record
                    continue

                if auto_index and index_callback:
                    try:
                        result = index_callback(local_pdf, record)
                        if result.get("success"):
                            record["indexed"] = True
                            record["case_id"] = result.get("case_id")
                            if result.get("message", "").lower().startswith("already"):
                                session_report["skipped_indexed"] += 1
                            else:
                                session_report["indexed"] += 1
                    except Exception as exc:
                        session_report["failed"].append({"title": item["case_title"], "error": str(exc)})

                existing[key] = record
                continue

            filename = stable_pdf_filename(item["case_title"])
            pdf_path = str(FCCP_PDF_DIR / filename)

            try:
                ok = download_pdf(item["download_url"], pdf_path)
                if not ok:
                    session_report["failed"].append({
                        "title": item["case_title"],
                        "error": "Download did not return a valid PDF",
                    })
                    continue
                session_report["downloaded"] += 1
            except Exception as exc:
                session_report["failed"].append({"title": item["case_title"], "error": str(exc)})
                continue

            record = {
                **item,
                "source_id": key,
                "pdf_path": pdf_path,
                "file_name": filename,
                "indexed": False,
                "case_id": None,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }

            if auto_index and index_callback:
                try:
                    result = index_callback(pdf_path, record)
                    if result.get("success"):
                        record["indexed"] = True
                        record["case_id"] = result.get("case_id")
                        session_report["indexed"] += 1
                    else:
                        session_report["failed"].append({
                            "title": item["case_title"],
                            "error": result.get("message", "Index failed"),
                        })
                except Exception as exc:
                    session_report["failed"].append({"title": item["case_title"], "error": str(exc)})

            existing[key] = record
            time.sleep(delay_seconds)

    manifest["items"] = list(existing.values())
    manifest["last_sync"] = datetime.now(timezone.utc).isoformat()
    manifest["total_items"] = len(manifest["items"])
    save_manifest(manifest)
    session_report["total_in_manifest"] = len(manifest["items"])
    return session_report


def get_fccp_status() -> Dict[str, Any]:
    manifest = load_manifest()
    items = manifest.get("items", [])
    indexed = sum(1 for i in items if i.get("indexed"))
    downloaded = sum(1 for i in items if resolve_local_pdf(i, i.get("case_title", "")))
    return {
        "source": "fccp",
        "source_url": JUDGMENTS_URL,
        "last_sync": manifest.get("last_sync"),
        "total_items": len(items),
        "downloaded": downloaded,
        "indexed": indexed,
        "items": items,
    }
