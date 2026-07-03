"""PDF text extraction with optional OCR fallback for scanned/image pages."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader

OCR_ENABLED = os.getenv("JAMS_OCR_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
OCR_LANG = os.getenv("JAMS_OCR_LANG", "eng")
OCR_DPI = int(os.getenv("JAMS_OCR_DPI", "200"))
OCR_MAX_PAGES = int(os.getenv("JAMS_OCR_MAX_PAGES", "40"))


def ocr_available() -> bool:
    if not OCR_ENABLED:
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _page_count(file_path: str) -> int:
    try:
        return len(PdfReader(file_path).pages)
    except Exception:
        return 0


def _ocr_page_image(file_path: str, page_no: int) -> str:
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(
        file_path,
        first_page=page_no,
        last_page=page_no,
        dpi=OCR_DPI,
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang=OCR_LANG) or ""


def _ocr_pages(file_path: str, page_numbers: List[int]) -> List[Tuple[int, str]]:
    if not page_numbers or not ocr_available():
        return []

    if OCR_MAX_PAGES > 0:
        page_numbers = page_numbers[:OCR_MAX_PAGES]

    results: List[Tuple[int, str]] = []
    for page_no in page_numbers:
        try:
            text = _ocr_page_image(file_path, page_no)
        except Exception:
            text = ""
        if text.strip():
            results.append((page_no, text.strip()))
    return results


def extract_pdf_text(file_path: str, use_ocr: bool = True) -> List[Dict[str, Any]]:
    """Extract per-page text. Uses Tesseract OCR on pages with no embedded text."""
    reader = PdfReader(file_path)
    pages_text: List[Dict[str, Any]] = []
    ocr_needed: List[int] = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages_text.append({"page": page_no, "text": text, "method": "native"})
        else:
            ocr_needed.append(page_no)

    used_ocr = False
    if ocr_needed and use_ocr and ocr_available():
        for page_no, text in _ocr_pages(file_path, ocr_needed):
            pages_text.append({"page": page_no, "text": text, "method": "ocr"})
            used_ocr = True

    pages_text.sort(key=lambda item: item["page"])
    if used_ocr:
        for item in pages_text:
            item["ocr_used"] = True
    return pages_text


def extract_first_pages_text(file_path: str, max_pages: int = 2) -> str:
    try:
        reader = PdfReader(file_path)
        text_parts: List[str] = []
        total_pages = min(len(reader.pages), max_pages)
        ocr_queue: List[int] = []

        for page_index in range(total_pages):
            page_no = page_index + 1
            text = (reader.pages[page_index].extract_text() or "").strip()
            if text:
                text_parts.append(text)
            else:
                ocr_queue.append(page_no)

        if not text_parts and ocr_queue and ocr_available():
            for page_no, text in _ocr_pages(file_path, ocr_queue[:max_pages]):
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts).strip()
    except Exception:
        return ""


def extraction_status_message(pages: List[Dict[str, Any]]) -> str:
    if not pages:
        if ocr_available():
            return (
                "No text could be extracted from this PDF. "
                "The scan may be low quality or use a language pack not installed on the server."
            )
        return (
            "No extractable text found in attached PDF. "
            "For scanned PDFs, install Tesseract OCR on the server "
            "(tesseract-ocr poppler-utils)."
        )
    ocr_pages = sum(1 for page in pages if page.get("method") == "ocr")
    if ocr_pages:
        return f"OCR extracted text from {ocr_pages} scanned page(s)."
    return ""
