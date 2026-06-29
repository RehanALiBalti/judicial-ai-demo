"""
Scraper for Lahore High Court — Judgments Approved for Reporting.
Source: https://data.lhc.gov.pk/reported_judgments/judgments_approved_for_reporting

Uses the site's AJAX endpoint (same as browser filter "All Courts"):
  /dynamic/approved_judgments_result_new.php
Returns all matching rows in one HTML response (~4683 judgments for All Courts).
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from backend.persistence import (
    COURT_LHC,
    LHC_PDF_DIR,
    ensure_data_dirs,
    lhc_manifest_item_key,
    load_lhc_manifest,
    save_lhc_manifest,
)

BASE_URL = "https://data.lhc.gov.pk"
PAGE_URL = f"{BASE_URL}/reported_judgments/judgments_approved_for_reporting"
API_URL = f"{BASE_URL}/dynamic/approved_judgments_result_new.php"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": PAGE_URL,
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(REQUEST_HEADERS)
    try:
        s.get(PAGE_URL, timeout=30)
    except requests.RequestException:
        pass
    return s


def _slugify(text: str, max_len: int = 72) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip()).strip("-").lower()
    return slug[:max_len] or "judgment"


def stable_pdf_filename(pdf_url: str) -> str:
    name = Path(pdf_url).name
    if name.lower().endswith(".pdf"):
        return f"lhc-{name}"
    return f"lhc-{_slugify(pdf_url)}.pdf"


def resolve_local_pdf(record: Dict[str, Any]) -> Optional[str]:
    pdf_path = record.get("pdf_path")
    if pdf_path and os.path.isfile(pdf_path):
        return pdf_path
    pdf_url = record.get("pdf_url")
    if pdf_url:
        stable = LHC_PDF_DIR / stable_pdf_filename(pdf_url)
        if stable.is_file():
            return str(stable)
    return None


def parse_total_count(html: str) -> Optional[int]:
    m = re.search(r"Total Judgments\s*<b>\((\d+)\)</b>", html, re.I)
    return int(m.group(1)) if m else None


def parse_approved_judgments_html(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 1:
            continue
        header = rows[0]
        tds = header.find_all("td")
        if len(tds) < 8:
            continue
        link = header.find("a", href=True)
        if not link or ".pdf" not in (link.get("href") or "").lower():
            continue

        pdf_url = link["href"].strip()
        case_number = tds[1].get_text(" ", strip=True)
        title = tds[2].get_text(" ", strip=True)
        author_judge = tds[3].get_text(" ", strip=True)
        decision_date = tds[4].get_text(" ", strip=True)
        lhc_citation = tds[5].get_text(" ", strip=True)
        other_citations = tds[6].get_text(" ", strip=True)

        tag_line = ""
        if len(rows) >= 2:
            tag_cell = rows[1].find("td")
            if tag_cell:
                tag_text = tag_cell.get_text(" ", strip=True)
                if tag_text.lower().startswith("tag line:"):
                    tag_line = tag_text[9:].strip()

        case_title = f"{case_number} — {title}" if case_number else title
        record = {
            "case_number": case_number,
            "case_title": case_title,
            "title": title,
            "author_judge": author_judge,
            "decision_date": decision_date,
            "lhc_citation": lhc_citation,
            "other_citations": other_citations,
            "tag_line": tag_line,
            "pdf_url": pdf_url,
            "source": "lhc",
            "court": COURT_LHC,
        }
        record["source_id"] = lhc_manifest_item_key(record)
        items.append(record)

    return items


def fetch_judgments(
    year: str = "",
    court_name: str = "All Courts",
    case_number: str = "",
    citation_tag: str = "",
    party_name: str = "",
    decision_date0: str = "",
    decision_date1: str = "",
    upload_date: str = "",
    upload_date1: str = "",
) -> tuple[str, List[Dict[str, Any]], Optional[int]]:
    params = {
        "year": year,
        "debug": "0",
        "courtName": court_name,
        "caseNumber": case_number,
        "citationTag": citation_tag,
        "partyName": party_name,
        "decisionDate0": decision_date0,
        "decisionDate1": decision_date1,
        "uploadDate": upload_date,
        "uploadDate1": upload_date1,
    }
    url = f"{API_URL}?{urlencode(params)}"
    session = _session()
    resp = session.get(url, timeout=(30, 300))
    resp.raise_for_status()
    html = resp.text
    total = parse_total_count(html)
    items = parse_approved_judgments_html(html)
    if not items:
        preview = re.sub(r"\s+", " ", html[:400]).strip()
        raise RuntimeError(
            f"LHC returned no judgments ({len(html)} bytes). "
            f"The server may be blocked from data.lhc.gov.pk. Preview: {preview[:200]}"
        )
    return url, items, total


def download_pdf(pdf_url: str, dest_path: str) -> bool:
    session = _session()
    resp = session.get(pdf_url, timeout=180, stream=True)
    resp.raise_for_status()
    data = resp.content
    if not data:
        return False
    content_type = (resp.headers.get("content-type") or "").lower()
    if "pdf" not in content_type and not data[:4] == b"%PDF":
        return False
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(data)
    return True


def sync_lhc_judgments(
    year: str = "",
    court_name: str = "All Courts",
    metadata_only: bool = False,
    auto_index: bool = True,
    download_limit: Optional[int] = 50,
    delay_seconds: float = 0.5,
    index_callback: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Scrape LHC approved judgments list, optionally download PDFs and index.

    - metadata_only=True: fetch all ~4683 rows into manifest (no PDF download)
    - download_limit: max new PDFs per run (None = all pending in manifest)
    """
    ensure_data_dirs()
    manifest = load_lhc_manifest()
    existing = {lhc_manifest_item_key(i): i for i in manifest.get("items", [])}

    session_report: Dict[str, Any] = {
        "scraped": 0,
        "total_reported": None,
        "downloaded": 0,
        "skipped_existing": 0,
        "skipped_indexed": 0,
        "indexed": 0,
        "metadata_only": metadata_only,
        "failed": [],
        "api_url": None,
    }

    try:
        api_url, items, total = fetch_judgments(year=year, court_name=court_name)
        session_report["api_url"] = api_url
        session_report["total_reported"] = total
    except Exception as exc:
        session_report["failed"].append({"stage": "fetch", "error": str(exc)})
        return session_report

    for item in items:
        session_report["scraped"] += 1
        key = lhc_manifest_item_key(item)
        prior = existing.get(key, {})
        record = {**prior, **item, "source_id": key}

        if metadata_only:
            existing[key] = record
            continue

        local_pdf = resolve_local_pdf(record)
        if local_pdf:
            session_report["skipped_existing"] += 1
            record["pdf_path"] = local_pdf
            record["file_name"] = os.path.basename(local_pdf)
            if prior.get("indexed") and prior.get("case_id"):
                session_report["skipped_indexed"] += 1
                record["indexed"] = True
                record["case_id"] = prior.get("case_id")
                existing[key] = record
                continue
            if auto_index and index_callback and not record.get("indexed"):
                if download_limit is not None and session_report["downloaded"] + session_report["indexed"] >= download_limit:
                    existing[key] = record
                    continue
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
                    session_report["failed"].append({"title": item.get("case_title"), "error": str(exc)})
            existing[key] = record
            continue

        if download_limit is not None and session_report["downloaded"] >= download_limit:
            existing[key] = record
            continue

        filename = stable_pdf_filename(item["pdf_url"])
        pdf_path = str(LHC_PDF_DIR / filename)
        try:
            ok = download_pdf(item["pdf_url"], pdf_path)
            if not ok:
                session_report["failed"].append({
                    "title": item.get("case_title"),
                    "error": "Download did not return a valid PDF",
                })
                existing[key] = record
                continue
            session_report["downloaded"] += 1
        except Exception as exc:
            session_report["failed"].append({"title": item.get("case_title"), "error": str(exc)})
            existing[key] = record
            continue

        record.update({
            "pdf_path": pdf_path,
            "file_name": filename,
            "indexed": False,
            "case_id": None,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        })

        if auto_index and index_callback:
            try:
                result = index_callback(pdf_path, record)
                if result.get("success"):
                    record["indexed"] = True
                    record["case_id"] = result.get("case_id")
                    session_report["indexed"] += 1
                else:
                    session_report["failed"].append({
                        "title": item.get("case_title"),
                        "error": result.get("message", "Index failed"),
                    })
            except Exception as exc:
                session_report["failed"].append({"title": item.get("case_title"), "error": str(exc)})

        existing[key] = record
        time.sleep(delay_seconds)

    manifest["items"] = list(existing.values())
    manifest["last_sync"] = datetime.now(timezone.utc).isoformat()
    manifest["total_items"] = len(manifest["items"])
    manifest["total_reported"] = session_report.get("total_reported")
    save_lhc_manifest(manifest)
    session_report["total_in_manifest"] = len(manifest["items"])
    return session_report


def get_lhc_status() -> Dict[str, Any]:
    manifest = load_lhc_manifest()
    items = manifest.get("items", [])
    indexed = sum(1 for i in items if i.get("indexed"))
    downloaded = sum(1 for i in items if resolve_local_pdf(i))
    return {
        "source": "lhc",
        "source_url": PAGE_URL,
        "api_url": API_URL,
        "last_sync": manifest.get("last_sync"),
        "total_reported": manifest.get("total_reported"),
        "total_items": len(items),
        "downloaded": downloaded,
        "indexed": indexed,
        "items": items,
    }
