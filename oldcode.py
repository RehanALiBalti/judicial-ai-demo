# -*- coding: utf-8 -*-
"""
JAMS - Judicial AI Management System

Modules:
- Upload Case: PDF upload + auto-fill metadata + index for search/AI/chat
- Chats: GPT-style chat from indexed cases + optional temporary PDF upload
- Indexed Cases: view all uploaded cases
- (Hidden for now) Summarize PDF, Search Cases, Ask AI
"""

import os
import re
import html
from datetime import datetime

import gradio as gr
import faiss
import numpy as np
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Global in-memory data
# -----------------------------
cases = []
documents = []
faiss_index = None

# -----------------------------
# Models
# -----------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
print("Models loaded successfully")

# -----------------------------
# HTML helpers
# -----------------------------
def safe_html(value):
    return html.escape(str(value or ""))


def nl2br(value):
    return safe_html(value).replace("\n", "<br>")


# Font Awesome icon helpers (CDN loaded in page header)
FA = {
    "success": "fa-circle-check",
    "error": "fa-circle-xmark",
    "info": "fa-circle-info",
    "warning": "fa-triangle-exclamation",
    "upload": "fa-cloud-arrow-up",
    "chat": "fa-comments",
    "cases": "fa-folder-tree",
    "attach": "fa-paperclip",
    "pdf": "fa-file-pdf",
    "send": "fa-paper-plane",
    "clear": "fa-trash-can",
    "gavel": "fa-gavel",
    "folder": "fa-folder-open",
    "database": "fa-database",
}


def fa_icon(name, style="solid"):
    css_class = FA.get(name, name)
    return f'<i class="fa-{style} {css_class}"></i>'


def fa_page_head():
    return """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&display=swap" />
    """


ALERT_STYLES = {
    "success": "background:#d1fae5;color:#065f46;border:1px solid #34d399;",
    "error": "background:#fee2e2;color:#991b1b;border:1px solid #f87171;",
    "info": "background:#dbeafe;color:#1e3a8a;border:1px solid #3b82f6;",
    "warning": "background:#fef3c7;color:#92400e;border:1px solid #fbbf24;",
}


def professional_alert(kind, title, message):
    auto_hide_class = "auto-hide-alert" if kind == "success" else ""
    inline_style = ALERT_STYLES.get(kind, ALERT_STYLES["info"])
    return f"""
    <div class="pro-alert pro-alert-{kind} {auto_hide_class}" style="{inline_style}">
        <div class="pro-alert-icon" style="color:inherit;">{fa_icon(kind)}</div>
        <div class="pro-alert-content">
            <div class="pro-alert-title" style="color:inherit;">{safe_html(title)}</div>
            <div class="pro-alert-message" style="color:inherit;">{safe_html(message)}</div>
        </div>
    </div>
    """


def format_chat_reply(text, reply_type="answer"):
    if reply_type == "error":
        return f"**Notice**\n\n{text}\n\n---\n*JAMS only answers from indexed cases or attached PDFs.*"
    if reply_type == "cases":
        return text
    sections = text.strip()
    if not sections.startswith("**") and not sections.startswith("#"):
        sections = f"**JAMS Response**\n\n{sections}"
    return sections


def section_header(title, subtitle, icon_key="folder"):
    return f"""
    <div class="section-card">
        <div class="section-card-inner">
            <div class="section-icon">{fa_icon(icon_key)}</div>
            <div>
                <div class="section-title">{safe_html(title)}</div>
                <div class="section-subtitle">{safe_html(subtitle)}</div>
            </div>
        </div>
    </div>
    """


def panel_header(title, subtitle="", icon_key="folder"):
    return f"""
    <div class="panel-header">
        <div class="panel-header-icon">{fa_icon(icon_key)}</div>
        <div>
            <div class="panel-header-title">{safe_html(title)}</div>
            {f'<div class="panel-header-sub">{safe_html(subtitle)}</div>' if subtitle else ''}
        </div>
    </div>
    """


def app_stats_html():
    return f"""
    <div class="stats-row">
        <div class="stat-chip">
            <span class="stat-num">{len(cases)}</span>
            <span class="stat-label">Indexed Cases</span>
        </div>
        <div class="stat-chip">
            <span class="stat-num">{len(documents)}</span>
            <span class="stat-label">Text Chunks</span>
        </div>
        <div class="stat-chip">
            <span class="stat-num">{sum(c.get('pages', 0) for c in cases)}</span>
            <span class="stat-label">Total Pages</span>
        </div>
    </div>
    """


def dashboard_header_html():
    return f"""
    <div class="app-shell">
        <div class="main-header">
            <div class="header-top">
                <div class="brand-block">
                    <div class="brand-icon">{fa_icon("gavel")}</div>
                    <div>
                        <h1>JAMS</h1>
                        <p class="header-subtitle">Judicial AI Management System</p>
                        <p class="header-tagline">
                            Upload judicial case PDFs, index legal records, and chat with
                            source-grounded AI — answers based only on your case data.
                        </p>
                    </div>
                </div>
                <span class="header-badge">Local Demo</span>
            </div>
            {app_stats_html()}
        </div>
    </div>
    """

# -----------------------------
# PDF extraction + chunking
# -----------------------------
def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    pages_text = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append({"page": page_no, "text": text.strip()})
    return pages_text


def chunk_text(text, chunk_size=1000, overlap=150):
    clean_text = " ".join(str(text or "").split())
    chunks = []
    start = 0
    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def extract_full_pdf_text_for_summary(pdf_file):
    if pdf_file is None:
        return ""
    reader = PdfReader(pdf_file.name)
    parts = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"\n--- Page {page_no} ---\n{text.strip()}")
    return "\n".join(parts).strip()

# -----------------------------
# Auto-fill metadata
# -----------------------------
def normalize_text(value):
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9\s\-:/]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def get_query_tokens(query):
    stopwords = {"court", "name", "case", "title", "date", "decision", "the", "a", "an", "of", "in", "on", "for", "to", "vs", "versus", "and"}
    return [t for t in normalize_text(query).split() if t not in stopwords and len(t) > 1]


def get_uploaded_file_name(pdf_file):
    if pdf_file is None:
        return ""
    original_name = getattr(pdf_file, "orig_name", None)
    return os.path.basename(original_name or pdf_file.name)


def clean_filename_title(file_name):
    name = os.path.splitext(file_name)[0]
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"^case\s*\d+\s*", "", name, flags=re.IGNORECASE).strip()
    return name.title()


def extract_first_pages_text(pdf_file, max_pages=2):
    if pdf_file is None:
        return ""
    try:
        reader = PdfReader(pdf_file.name)
        parts = []
        for page_index in range(min(len(reader.pages), max_pages)):
            text = reader.pages[page_index].extract_text() or ""
            if text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    except Exception:
        return ""


def guess_case_title(text, file_name=""):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    patterns = [
        r"([A-Z][A-Za-z\s\.\-]+)\s+(?:vs\.?|v\.?|versus)\s+([A-Z][A-Za-z\s\.\-]+)",
        r"case\s*title\s*[:\-]\s*(.+)",
        r"title\s*[:\-]\s*(.+)",
        r"matter\s*of\s*[:\-]\s*(.+)",
    ]
    for line in lines[:40]:
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    return f"{match.group(1).strip()} Vs {match.group(2).strip()}".title()
                return match.group(1).strip().title()
    return clean_filename_title(file_name) if file_name else ""


def guess_court_name(text):
    patterns = [
        r"(Supreme Court(?:\s+of\s+[A-Za-z\s]+)?)",
        r"([A-Za-z\s]+High Court)",
        r"([A-Za-z\s]+Sessions Court)",
        r"(Sessions Court)",
        r"([A-Za-z\s]+Civil Court)",
        r"(Civil Court)",
        r"([A-Za-z\s]+Family Court)",
        r"(Family Court)",
        r"([A-Za-z\s]+Service Tribunal)",
        r"(Service Tribunal)",
        r"([A-Za-z\s]+Tribunal)",
        r"(Tribunal)",
        r"court\s*name\s*[:\-]\s*(.+)",
        r"court\s*[:\-]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            court = re.sub(r"\s+", " ", match.group(1).strip())
            if len(court) <= 80:
                return court.title()
    return ""


def normalize_date_value(value):
    value = (value or "").strip()
    if not value:
        return ""
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"]:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return ""


def guess_decision_date(text, file_name=""):
    patterns = [
        r"decision\s*date\s*[:\-]\s*([A-Za-z0-9,\s\/\-]+)",
        r"date\s*of\s*decision\s*[:\-]\s*([A-Za-z0-9,\s\/\-]+)",
        r"decided\s*on\s*[:\-]?\s*([A-Za-z0-9,\s\/\-]+)",
        r"order\s*date\s*[:\-]\s*([A-Za-z0-9,\s\/\-]+)",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b(\d{2}-\d{2}-\d{4})\b",
        r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
        r"\b([A-Za-z]+\s+\d{1,2},\s*\d{4})\b",
    ]
    combined = f"{text}\n{file_name}"
    for pattern in patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            raw_date = match.group(1).strip().split("\n")[0].strip()
            normalized = normalize_date_value(raw_date)
            if normalized:
                return normalized
    return ""


def auto_fill_case_metadata(pdf_file):
    if pdf_file is None:
        return "", "", "", professional_alert("info", "Ready", "Select a PDF file to auto-fill case metadata.")
    file_name = get_uploaded_file_name(pdf_file)
    first_text = extract_first_pages_text(pdf_file, max_pages=2)
    if not first_text:
        return clean_filename_title(file_name), "", "", professional_alert("warning", "Limited Auto-Fill", "No extractable text found. Title guessed from file name. OCR may be required.")
    title = guess_case_title(first_text, file_name)
    court = guess_court_name(first_text)
    date = guess_decision_date(first_text, file_name)
    missing = []
    if not title: missing.append("case title")
    if not court: missing.append("court name")
    if not date: missing.append("decision date")
    if missing:
        alert = professional_alert("warning", "Auto-Fill Partially Completed", "Could not detect: " + ", ".join(missing) + ". Please review manually.")
    else:
        alert = professional_alert("success", "Auto-Fill Completed", "Case title, court name, and decision date were detected from the PDF.")
    return title, court, date, alert

# -----------------------------
# Indexing
# -----------------------------
def build_document_search_text(doc):
    return f"""
Case ID: {doc.get('case_id', '')}
Case Title: {doc.get('title', '')}
Court: {doc.get('court', '')}
Decision Date: {doc.get('decision_date', '')}
Page: {doc.get('page', '')}
Text: {doc.get('text', '')}
""".strip()


def rebuild_faiss_index():
    global faiss_index
    if not documents:
        faiss_index = None
        return
    texts = [build_document_search_text(doc) for doc in documents]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings.astype("float32"))


def validate_case_upload_inputs(pdf_file, title, court, date):
    errors = []
    if pdf_file is None: errors.append("PDF file is required.")
    if not title or not title.strip(): errors.append("Case title is required.")
    if not court or not court.strip(): errors.append("Court name is required.")
    if not date or not date.strip(): errors.append("Decision date is required.")
    if date and date.strip() and not normalize_date_value(date.strip()):
        errors.append("Decision date must be valid. Recommended format: YYYY-MM-DD.")
    return errors


def upload_case(pdf_file, case_title, court_name, decision_date):
    global cases, documents
    pages = extract_pdf_text(pdf_file.name)
    if not pages:
        return "No text found in this PDF. This may be a scanned PDF and OCR is required."
    case_id = f"CASE-{len(cases) + 1:03d}"
    case_record = {
        "case_id": case_id,
        "title": case_title.strip(),
        "court": (court_name or "").strip(),
        "decision_date": (decision_date or "").strip(),
        "file_name": pdf_file.name,
        "pages": len(pages),
    }
    cases.append(case_record)
    added_chunks = 0
    for page in pages:
        for chunk_index, chunk in enumerate(chunk_text(page["text"])):
            documents.append({
                "case_id": case_id,
                "title": case_title.strip(),
                "court": (court_name or "").strip(),
                "decision_date": (decision_date or "").strip(),
                "page": page["page"],
                "chunk_index": chunk_index,
                "text": chunk,
                "source_type": "indexed_case",
            })
            added_chunks += 1
    rebuild_faiss_index()
    return f"Uploaded and indexed successfully: {case_title} | Pages: {len(pages)} | Chunks: {added_chunks}"


def upload_case_professional(pdf_file, case_title, court_name, decision_date):
    errors = validate_case_upload_inputs(pdf_file, case_title, court_name, decision_date)
    if errors:
        return professional_alert("error", "Validation Failed", " ".join(errors)), pdf_file, case_title, court_name, decision_date
    try:
        normalized_date = normalize_date_value(decision_date.strip()) or decision_date.strip()
        result = upload_case(pdf_file, case_title.strip(), court_name.strip(), normalized_date)
        if "successfully" in result.lower():
            return professional_alert("success", "Case Uploaded Successfully", result), None, "", "", ""
        return professional_alert("warning", "Upload Warning", result), pdf_file, case_title, court_name, decision_date
    except Exception as error:
        return professional_alert("error", "Upload Failed", str(error)), pdf_file, case_title, court_name, decision_date

# -----------------------------
# Search functions
# -----------------------------
def search_cases_by_metadata(query):
    query_normalized = normalize_text(query)
    if not query_normalized:
        return []
    query_tokens = get_query_tokens(query)
    matched = []
    for case in cases:
        metadata_text = normalize_text(f"{case.get('case_id', '')} {case.get('title', '')} {case.get('court', '')} {case.get('decision_date', '')}")
        direct = query_normalized in metadata_text
        token_count = sum(1 for token in query_tokens if token in metadata_text)
        token_match = len(query_tokens) >= 2 and token_count >= 2
        if direct or token_match:
            matched.append(case)
    return matched


def deduplicate_results(results, max_results=3):
    out = []
    seen = set()
    for item in results:
        key = (item.get("source_type"), item.get("case_id"), item.get("page"), item.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_results:
            break
    return out


def search_cases(query, top_k=5):
    if faiss_index is None or not documents:
        return []
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    search_k = min(max(top_k * 4, top_k), len(documents))
    distances, indexes = faiss_index.search(query_embedding, search_k)
    results = []
    for rank, doc_index in enumerate(indexes[0], start=1):
        if doc_index < 0:
            continue
        doc = documents[doc_index]
        results.append({"rank": rank, "distance": float(distances[0][rank - 1]), **doc})
    return deduplicate_results(results, max_results=top_k)


def search_temp_docs(query, temp_docs, top_k=3):
    if not temp_docs:
        return []
    texts = [build_document_search_text(doc) for doc in temp_docs]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    temp_index = faiss.IndexFlatL2(embeddings.shape[1])
    temp_index.add(embeddings.astype("float32"))
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indexes = temp_index.search(query_embedding, min(top_k, len(temp_docs)))
    results = []
    for rank, doc_index in enumerate(indexes[0], start=1):
        if doc_index < 0:
            continue
        doc = temp_docs[doc_index]
        results.append({"rank": rank, "distance": float(distances[0][rank - 1]), **doc})
    return results


def should_use_temp_pdf_directly(query):
    query = normalize_text(query)
    triggers = ["this pdf", "uploaded pdf", "temporary pdf", "summarize", "summary", "is pdf", "is document", "current pdf", "file"]
    return any(word in query for word in triggers)

# -----------------------------
# Professional result HTML
# -----------------------------
def metadata_cases_html(matched_cases):
    if not matched_cases:
        return professional_alert("info", "No Record Found", "No case matched your search.")
    cards = ""
    for case in matched_cases:
        cards += f"""
        <div class="case-card">
            <div class="case-card-top">
                <div><div class="case-title">{safe_html(case.get('title'))}</div><div class="case-subtitle">{safe_html(case.get('case_id'))}</div></div>
                <span class="case-badge">Metadata Match</span>
            </div>
            <div class="case-meta-grid">
                <div class="meta-item"><span>Court</span><strong>{safe_html(case.get('court') or 'N/A')}</strong></div>
                <div class="meta-item"><span>Decision Date</span><strong>{safe_html(case.get('decision_date') or 'N/A')}</strong></div>
                <div class="meta-item"><span>Pages</span><strong>{safe_html(case.get('pages') or 'N/A')}</strong></div>
            </div>
        </div>
        """
    return f"<div class='result-panel'><div class='result-header'><div><h3>Matching Case Records</h3><p>Cases matched by metadata.</p></div></div>{cards}</div>"


def search_results_html(results, query):
    if not results:
        return professional_alert("info", "No Results Found", "No matching case content found.")
    cards = ""
    for item in results:
        badge = "Temporary PDF" if item.get("source_type") == "chat_temp_pdf" else "Indexed Case"
        cards += f"""
        <div class="case-card">
            <div class="case-card-top">
                <div><div class="case-title">{safe_html(item.get('title'))}</div><div class="case-subtitle">{safe_html(item.get('case_id'))} · Page {safe_html(item.get('page'))}</div></div>
                <span class="case-badge">{badge}</span>
            </div>
            <div class="case-meta-grid">
                <div class="meta-item"><span>Court</span><strong>{safe_html(item.get('court') or 'N/A')}</strong></div>
                <div class="meta-item"><span>Decision Date</span><strong>{safe_html(item.get('decision_date') or 'N/A')}</strong></div>
                <div class="meta-item"><span>Score</span><strong>{round(float(item.get('distance', 0)), 3)}</strong></div>
            </div>
            <div class="snippet-box">{nl2br(item['text'][:900])}</div>
        </div>
        """
    return f"<div class='result-panel'><div class='result-header'><div><h3>Search Results</h3><p>Showing best matches for: <strong>{safe_html(query)}</strong></p></div></div>{cards}</div>"


def search_ui(query):
    if not query or not query.strip():
        return professional_alert("error", "Search Required", "Please enter a search query.")
    query = query.strip()
    metadata = search_cases_by_metadata(query)
    if metadata:
        return metadata_cases_html(metadata)
    return search_results_html(search_cases(query, top_k=5), query)


def indexed_cases_html():
    if not cases:
        return professional_alert("info", "No Cases Indexed", "Upload case PDFs in the Upload Case tab, then click Refresh below.")

    rows = ""
    for i, case in enumerate(cases, start=1):
        rows += f"""
        <tr>
            <td class="jams-td-center jams-td-muted">{i}</td>
            <td><span class="jams-pill jams-pill-blue">{safe_html(case.get('case_id'))}</span></td>
            <td class="jams-td-bold">{safe_html(case.get('title'))}</td>
            <td>{safe_html(case.get('court') or 'N/A')}</td>
            <td><span class="jams-pill jams-pill-gray">{safe_html(case.get('decision_date') or 'N/A')}</span></td>
            <td class="jams-td-center"><span class="jams-pill jams-pill-green">{safe_html(case.get('pages') or '0')}</span></td>
        </tr>
        """

    total_pages = sum(c.get("pages", 0) for c in cases)
    return f"""
    <div class="jams-indexed-results">
        <div class="jams-data-card">
            <div class="jams-data-card-header">
                <div class="jams-data-card-title">
                    {fa_icon("database")}
                    <span>Indexed Case Library</span>
                </div>
                <span class="jams-count-pill">{len(cases)} Cases</span>
            </div>
            <div class="jams-stats-grid">
                <div class="jams-stat-box">
                    <div class="jams-stat-label">Total Cases</div>
                    <div class="jams-stat-value jams-color-blue">{len(cases)}</div>
                </div>
                <div class="jams-stat-box">
                    <div class="jams-stat-label">Total Pages</div>
                    <div class="jams-stat-value jams-color-green">{total_pages}</div>
                </div>
                <div class="jams-stat-box">
                    <div class="jams-stat-label">Text Chunks</div>
                    <div class="jams-stat-value jams-color-teal">{len(documents)}</div>
                </div>
            </div>
            <div class="jams-table-wrap">
                <table class="jams-table">
                    <thead>
                        <tr>
                            <th class="jams-th-center">#</th>
                            <th>Case ID</th>
                            <th>Case Title</th>
                            <th>Court</th>
                            <th>Decision Date</th>
                            <th class="jams-th-center">Pages</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
    </div>
    """


def ai_answer_html(answer):
    return f"""
    <div class="ai-answer-card">
        <div class="ai-answer-header"><div><h3>AI Answer</h3><p>Generated from provided case sources only.</p></div><span class="case-badge">Source Based</span></div>
        <div class="ai-answer-body">{nl2br(answer)}</div>
    </div>
    """

# -----------------------------
# AI generation
# -----------------------------
def build_sources_text(results, max_chars_per_source=700):
    text = ""
    for idx, item in enumerate(results, start=1):
        label = "Temporary Chat PDF" if item.get("source_type") == "chat_temp_pdf" else "Indexed Case"
        text += f"""
Source {idx}
Source Type: {label}
Case ID: {item['case_id']}
Case Title: {item['title']}
Court: {item['court'] or 'N/A'}
Decision Date: {item['decision_date'] or 'N/A'}
Page: {item['page']}
Text: {item['text'][:max_chars_per_source]}
"""
    return text.strip()


def generate_from_model(prompt, max_new_tokens=350):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    with torch.inference_mode():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_ai_answer(question):
    if not question or not question.strip():
        return "Please enter a question."
    question = question.strip()
    metadata = search_cases_by_metadata(question)
    if metadata:
        return "\n".join([f"{c.get('case_id')} | {c.get('title')} | {c.get('court')} | {c.get('decision_date')}" for c in metadata])
    results = search_cases(question, top_k=3)
    if not results:
        return "No indexed case data found. Please upload case PDFs first."
    prompt = f"""
You are an AI judicial research assistant.

Rules:
- Answer only from the provided case sources.
- Do not invent case names, citations, laws, facts, or decisions.
- Do not give general legal principles unless directly supported by sources.
- If answer is not supported, say: No supported source found.
- Mention source case title, case id, and page number.

User Question:
{question}

Available Case Sources:
{build_sources_text(results, 650)}

Answer format:
1. Short Answer
2. Relevant Case
3. Reasoning
4. Sources
"""
    return generate_from_model(prompt, max_new_tokens=300)


def generate_ai_answer_professional(question):
    if not question or not question.strip():
        return professional_alert("error", "Question Required", "Please enter a question.")
    metadata = search_cases_by_metadata(question.strip())
    if metadata:
        return metadata_cases_html(metadata)
    try:
        return ai_answer_html(generate_ai_answer(question))
    except Exception as error:
        return professional_alert("error", "AI Answer Failed", str(error))


def generate_case_summary(pdf_file, case_title):
    if pdf_file is None:
        return "Please upload a PDF file."
    full_text = extract_full_pdf_text_for_summary(pdf_file)
    if not full_text:
        return "No extractable text found. This may be a scanned/image-based PDF. OCR is required."
    prompt = f"""
You are an AI judicial case summarizer.

Rules:
- Summarize only from provided PDF text.
- Do not invent facts, citations, laws, parties, or decisions.
- If something is not clear, write: Not clearly available in the provided text.

Case Title:
{case_title or 'Not provided'}

PDF Text:
{full_text[:9000]}

Return:
1. Case Overview
2. Important Facts
3. Legal Issues
4. Arguments / Claims
5. Court Reasoning
6. Final Decision / Order
7. Important Points
"""
    return generate_from_model(prompt, max_new_tokens=500)

# -----------------------------
# Chat (Gradio ChatInterface)
# -----------------------------
def build_chat_context_messages(history, max_turns=3):
    """Build context string from Gradio ChatInterface message history."""
    if not history:
        return ""
    lines = []
    for msg in history[-(max_turns * 2):]:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [str(c) for c in content if c]
            content = " ".join(text_parts)
        elif isinstance(content, dict):
            content = content.get("text", str(content))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _file_path_from_upload(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    return getattr(file_obj, "name", None) or str(file_obj)


def parse_chat_message(message):
    """Parse Gradio ChatInterface message (text or multimodal)."""
    if message is None:
        return "", []
    if isinstance(message, str):
        return message.strip(), []
    if isinstance(message, dict):
        return (message.get("text") or "").strip(), message.get("files") or []
    text = getattr(message, "text", "") or ""
    files = getattr(message, "files", None) or []
    return str(text).strip(), list(files)


def process_attached_pdfs(files, existing_docs=None):
    """Process PDF attachments from multimodal chat input (session-only)."""
    if not files:
        return existing_docs or []
    for file_obj in files:
        path = _file_path_from_upload(file_obj)
        if not path or not str(path).lower().endswith(".pdf"):
            continue
        pages = extract_pdf_text(path)
        if not pages:
            continue
        title = clean_filename_title(os.path.basename(path))
        temp_docs = []
        for page in pages:
            for chunk_index, chunk in enumerate(chunk_text(page["text"])):
                temp_docs.append({
                    "case_id": "CHAT-PDF",
                    "title": title,
                    "court": "Temporary Chat PDF",
                    "decision_date": "N/A",
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "source_type": "chat_temp_pdf",
                })
        return temp_docs
    return existing_docs or []


def jams_chat_fn(message, history, chat_temp_docs):
    """
    Gradio ChatInterface handler.
    Supports multimodal PDF attach + indexed case search + local AI answer.
    """
    if chat_temp_docs is None:
        chat_temp_docs = []

    question, files = parse_chat_message(message)
    if files:
        chat_temp_docs = process_attached_pdfs(files, chat_temp_docs)

    if not question and not chat_temp_docs:
        return (
            format_chat_reply("Please enter a message or attach a PDF.", "error"),
            chat_temp_docs,
        )

    if not question and chat_temp_docs:
        question = "Summarize the attached PDF and list key legal points."

    metadata = search_cases_by_metadata(question)
    if metadata and not chat_temp_docs:
        answer = "**Matching Case Records**\n\n"
        for c in metadata:
            answer += (
                f"**{c.get('title')}**\n"
                f"- Case ID: `{c.get('case_id')}`\n"
                f"- Court: {c.get('court') or 'N/A'}\n"
                f"- Decision Date: {c.get('decision_date') or 'N/A'}\n"
                f"- Pages: {c.get('pages') or 'N/A'}\n\n"
            )
        return format_chat_reply(answer, "cases"), chat_temp_docs

    temp_results = []
    if chat_temp_docs:
        if should_use_temp_pdf_directly(question):
            temp_results = chat_temp_docs[:3]
            for rank, item in enumerate(temp_results, start=1):
                item["rank"] = rank
                item["distance"] = 0.0
        else:
            temp_results = search_temp_docs(question, chat_temp_docs, top_k=3)

    indexed_results = search_cases(question, top_k=3)
    results = (temp_results + indexed_results[:1]) if temp_results else indexed_results
    results = deduplicate_results(results, max_results=4)

    if not results:
        return (
            format_chat_reply(
                "No relevant data found.\n\n"
                "• Upload cases in **Upload Case** tab, or\n"
                "• Attach a PDF in the message box.",
                "error",
            ),
            chat_temp_docs,
        )

    prompt = f"""
You are JAMS, an AI judicial research chat assistant.

Rules:
- Answer only from the provided case sources.
- Sources may include indexed cases and/or a temporary chat PDF.
- Temporary chat PDF is for this chat only and is not stored in indexed cases.
- Do not invent case names, citations, laws, facts, or decisions.
- If answer is not supported, say: No supported source found.
- Mention source type, case title, case id, and page number.

Previous Conversation:
{build_chat_context_messages(history) or 'No previous conversation.'}

Current User Question:
{question}

Available Case Sources:
{build_sources_text(results, 650)}

Answer format:
1. Answer
2. Relevant Source
3. Reasoning
4. Source References
"""
    try:
        answer = generate_from_model(prompt, max_new_tokens=350)
        return format_chat_reply(answer), chat_temp_docs
    except Exception as exc:
        return format_chat_reply(f"AI generation failed: {exc}", "error"), chat_temp_docs

# -----------------------------
# CSS
# -----------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

:root {
    --primary: #0f2744;
    --primary-light: #1d4ed8;
    --accent: #c9a227;
    --bg: #d8dee9;
    --surface: #edf1f7;
    --surface-2: #ffffff;
    --border: #8896ab;
    --text: #0b1220;
    --text-soft: #2d3748;
    --muted: #4a5568;
    --shadow: 0 10px 30px rgba(11, 18, 32, 0.12);
    --chat-bg: #eceff4;
    --chat-user: #1e40af;
    --chat-bot: #ffffff;
    --chat-bot-text: #1a202c;
    --input-bg: #ffffff;
    --input-border: #7b8ba3;
    --font: 'Plus Jakarta Sans', system-ui, sans-serif;
}

*, body, .gradio-container, input, textarea, button, label, .markdown, .prose {
    font-family: var(--font) !important;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 20px 18px 28px !important;
}

/* Header */
.app-shell { margin-bottom: 20px; }
.main-header {
    background: linear-gradient(135deg, #0a1638 0%, #122a5f 55%, #1a3d7a 100%);
    color: #fff;
    padding: 28px 32px;
    border-radius: 20px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -8%;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(201,162,39,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.header-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 20px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.brand-block { display: flex; gap: 16px; align-items: flex-start; }
.brand-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%) !important;
    border: none !important;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 14px rgba(29,78,216,0.4);
}
.brand-icon i { font-size: 22px; color: #ffffff !important; }
.main-header h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 0.06em;
    line-height: 1.1;
}
.header-subtitle {
    margin: 4px 0 0;
    color: #c9a227;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.header-tagline {
    margin: 10px 0 0;
    color: #b8c9e8;
    font-size: 14px;
    line-height: 1.65;
    max-width: 620px;
}
.header-badge {
    background: rgba(201,162,39,0.12);
    border: 1px solid rgba(201,162,39,0.4);
    color: #f5e6b8;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* Stats */
.stats-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 20px;
    position: relative;
    z-index: 1;
}
.stat-chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 14px;
    padding: 12px 18px;
    min-width: 120px;
    backdrop-filter: blur(4px);
}
.stat-num {
    display: block;
    font-size: 22px;
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
}
.stat-label {
    display: block;
    margin-top: 4px;
    font-size: 11px;
    font-weight: 600;
    color: #94a8cc;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Tabs — forced high contrast */
.tabs, div.tabs {
    background: #0b1a30 !important;
    border: 2px solid #1e3a5f !important;
    border-radius: 14px !important;
    padding: 0 !important;
    box-shadow: var(--shadow) !important;
    overflow: hidden !important;
}
.tab-nav, .tabs > .tab-nav, div[role="tablist"] {
    background: #0b1a30 !important;
    border-bottom: 2px solid #1e3a5f !important;
    gap: 8px !important;
    padding: 12px 16px !important;
}
.tab-nav button, .tabs button, button[role="tab"] {
    font-family: var(--font) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #8fa4c4 !important;
    border-radius: 10px !important;
    padding: 11px 22px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
    opacity: 1 !important;
}
.tab-nav button:hover, button[role="tab"]:hover {
    color: #e8eef8 !important;
    background: rgba(255,255,255,0.08) !important;
}
.tab-nav button.selected, button[role="tab"][aria-selected="true"], .tabs button.selected {
    color: #ffffff !important;
    background: #1d4ed8 !important;
    border: 1px solid #3b82f6 !important;
    box-shadow: 0 4px 12px rgba(29,78,216,0.45) !important;
}
.tabitem, .tab-content {
    padding: 22px 20px 18px !important;
    background: var(--surface) !important;
}

/* Section cards */
.section-card {
    background: #ffffff !important;
    border: 1px solid #b0bccf !important;
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(15,31,75,0.06);
}
.section-title { color: #0f172a !important; }
.section-subtitle { color: #475569 !important; }

/* Override Gradio white text on HTML blocks */
.gradio-container .contain, .gradio-container .form, .gradio-container .tabitem {
    color: #0f172a !important;
}
.gradio-container .html-container, .gradio-container [class*="html"] {
    color: #0f172a !important;
}
.section-card-inner {
    display: flex;
    gap: 14px;
    align-items: flex-start;
}
.section-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%) !important;
    border: none !important;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(29,78,216,0.35);
}
.section-icon i, .section-icon .fa-solid {
    color: #ffffff !important;
    font-size: 18px !important;
}
.section-title {
    font-size: 18px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}
.section-subtitle {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
}

/* JAMS upload form — single column */
.upload-card {
    background: var(--surface-2) !important;
    border: 1px solid #b0bccf !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 8px 24px rgba(11,18,32,0.08) !important;
    max-width: 720px !important;
    margin: 0 auto !important;
}
/* JAMS form styles */
.jams-form .block { gap: 16px !important; }
.jams-form label span {
    color: #1e293b !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}
.jams-form input, .jams-form textarea, .jams-form .wrap {
    background: #f8fafc !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
}
.jams-form input:focus, .jams-form textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}
.jams-form button.primary {
    width: 100% !important;
    margin-top: 8px !important;
    padding: 12px !important;
    font-size: 15px !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
}
.form-status-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 24px 0 16px;
    padding-top: 20px;
    border-top: 2px solid #e2e8f0;
}
.form-status-divider span {
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    white-space: nowrap;
}
.form-status-divider::before, .form-status-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #cbd5e1;
}
.upload-status-box, .upload-status-box > .wrap, .upload-status-box > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Modern page wrapper (Upload + Indexed) */
.modern-page-card, .indexed-page-card {
    background: #ffffff !important;
    border: 1px solid #b0bccf !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px rgba(11,18,32,0.09) !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}
.indexed-page-card { max-width: 100% !important; }
.page-action-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 14px 16px;
    background: linear-gradient(135deg, #f0f4fa, #e8eef6);
    border: 1px solid #c5d0e0;
    border-radius: 12px;
    margin-bottom: 20px;
}
.page-action-text { font-size: 13px; color: #475569; line-height: 1.5; }
.page-action-text strong { color: #0f172a; display: block; font-size: 14px; margin-bottom: 2px; }
.indexed-results-box, .indexed-results-box > div, .indexed-results-box .wrap {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.jams-refresh-row button.primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    width: 100% !important;
    max-width: 280px !important;
}

/* JAMS indexed results — pure custom CSS */
.jams-indexed-results { color: #1e293b; font-family: var(--font); }
.jams-data-card {
    background: #fff;
    border: 1px solid #c5d0e0;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(11,18,32,0.08);
}
.jams-data-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a);
    color: #fff;
}
.jams-data-card-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 800;
}
.jams-data-card-title i { color: #fff !important; font-size: 16px; }
.jams-count-pill {
    background: #fff;
    color: #1d4ed8;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
}
.jams-stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    padding: 18px 20px;
    background: #f1f5f9;
    border-bottom: 1px solid #e2e8f0;
}
.jams-stat-box {
    background: #fff;
    border: 1px solid #d1dae6;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}
.jams-stat-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 6px;
}
.jams-stat-value { font-size: 26px; font-weight: 800; line-height: 1; }
.jams-color-blue { color: #1d4ed8; }
.jams-color-green { color: #059669; }
.jams-color-teal { color: #0284c7; }
.jams-table-wrap { overflow-x: auto; }
.jams-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.jams-table thead th {
    background: #1e3a5f;
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 14px 16px;
    text-align: left;
    border: none;
}
.jams-th-center, .jams-td-center { text-align: center !important; }
.jams-table tbody td {
    padding: 14px 16px;
    border-bottom: 1px solid #e2e8f0;
    color: #1e293b;
    vertical-align: middle;
}
.jams-table tbody tr:hover { background: #e8f0fe; }
.jams-td-bold { font-weight: 700; color: #0f172a; }
.jams-td-muted { color: #64748b; font-weight: 600; }
.jams-pill {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid transparent;
}
.jams-pill-blue { background: #dbeafe; color: #1d4ed8; border-color: #93c5fd; }
.jams-pill-gray { background: #f1f5f9; color: #334155; border-color: #cbd5e1; }
.jams-pill-green { background: #d1fae5; color: #059669; border-color: #6ee7b7; }

/* Footer */
.jams-footer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 20px;
    padding: 16px 20px;
    background: #fff;
    border: 1px solid #c5d0e0;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(11,18,32,0.05);
}
.jams-footer-badge {
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a);
    color: #fff;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.06em;
}
.jams-footer-text { font-size: 12px; color: #64748b; }

@media (max-width: 768px) {
    .jams-stats-grid { grid-template-columns: 1fr; }
}

.panel-header-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a) !important;
    border: none !important;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 3px 10px rgba(29,78,216,0.3);
}
.panel-header-icon i { color: #ffffff !important; font-size: 16px !important; }

/* Upload split layout (indexed cases) */
.upload-split-row {
    gap: 18px !important;
    align-items: stretch !important;
}
.upload-form-panel, .upload-status-panel {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 16px rgba(11,18,32,0.06) !important;
    min-height: 420px !important;
}
.upload-form-panel { border-top: 3px solid var(--primary-light) !important; }
.upload-status-panel { border-top: 3px solid var(--accent) !important; }
.upload-form-inner { gap: 12px !important; }
.upload-status-panel .pro-alert { margin-top: 0 !important; min-height: 120px; }

.panel-header {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid #cbd5e1;
}
.panel-header-icon {
    width: 40px; height: 40px;
    background: #e8eef8;
    border: 1px solid #b8c5d9;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: var(--primary-light);
    font-size: 16px;
}
.panel-header-title {
    font-size: 15px;
    font-weight: 800;
    color: var(--text);
}
.panel-header-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 2px;
    line-height: 1.4;
}

/* Panels */
.panel-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 14px rgba(15,31,75,0.04);
    height: 100%;
}
.panel-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 14px;
}
.result-panel, .ai-answer-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    margin-top: 4px;
    box-shadow: 0 4px 14px rgba(15,31,75,0.04);
}

/* Alerts — force visible text (override Gradio theme) */
.pro-alert, .pro-alert * {
    box-sizing: border-box;
}
.pro-alert {
    display: flex !important;
    gap: 14px !important;
    align-items: flex-start !important;
    padding: 16px 18px !important;
    border-radius: 12px !important;
    margin: 0 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
.pro-alert-icon { font-size: 18px !important; flex-shrink: 0 !important; }
.pro-alert-title { font-weight: 800 !important; margin-bottom: 4px !important; font-size: 14px !important; }
.pro-alert-message { font-weight: 500 !important; font-size: 13px !important; }
.pro-alert-success, .pro-alert-success .pro-alert-title, .pro-alert-success .pro-alert-message, .pro-alert-success i {
    background-color: #d1fae5 !important; color: #065f46 !important; border-color: #34d399 !important;
}
.pro-alert-error, .pro-alert-error .pro-alert-title, .pro-alert-error .pro-alert-message, .pro-alert-error i {
    background-color: #fee2e2 !important; color: #991b1b !important; border-color: #f87171 !important;
}
.pro-alert-info, .pro-alert-info .pro-alert-title, .pro-alert-info .pro-alert-message, .pro-alert-info i {
    background-color: #dbeafe !important; color: #1e3a8a !important; border-color: #3b82f6 !important;
}
.pro-alert-warning, .pro-alert-warning .pro-alert-title, .pro-alert-warning .pro-alert-message, .pro-alert-warning i {
    background-color: #fef3c7 !important; color: #92400e !important; border-color: #fbbf24 !important;
}
.pro-alert-success { border: 1px solid #34d399 !important; }
.pro-alert-error { border: 1px solid #f87171 !important; }
.pro-alert-info { border: 1px solid #3b82f6 !important; }
.pro-alert-warning { border: 1px solid #fbbf24 !important; }
.pro-alert-icon, .pro-alert-content { background: transparent !important; }
.gradio-container .pro-alert, .contain .pro-alert, [class*="html"] .pro-alert {
    opacity: 1 !important;
}
.gradio-container .pro-alert-info, .contain .pro-alert-info {
    background: #dbeafe !important;
    color: #1e3a8a !important;
}
.gradio-container .pro-alert-info .pro-alert-title,
.gradio-container .pro-alert-info .pro-alert-message {
    color: #1e3a8a !important;
}
.auto-hide-alert { animation: fadeOutAlert .6s ease forwards !important; animation-delay: 4s !important; }
@keyframes fadeOutAlert {
    from { opacity: 1; max-height: 180px; }
    to { opacity: 0; max-height: 0; padding-top: 0; padding-bottom: 0; margin-top: 0; border-width: 0; }
}

/* Results */
.result-header, .ai-answer-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 14px;
    margin-bottom: 16px;
    align-items: flex-start;
}
.result-header h3, .ai-answer-header h3 {
    margin: 0;
    color: var(--text);
    font-size: 18px;
    font-weight: 800;
}
.result-header p, .ai-answer-header p {
    margin: 5px 0 0;
    color: var(--muted);
    font-size: 13px;
}
.case-card {
    background: #fafbfd;
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
    transition: box-shadow 0.2s ease;
}
.case-card:hover { box-shadow: 0 6px 20px rgba(15,31,75,0.07); }
.case-card-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 12px;
}
.case-title { font-size: 16px; font-weight: 800; color: var(--text); }
.case-subtitle { margin-top: 3px; color: var(--muted); font-size: 12px; }
.case-badge {
    background: #eef2ff;
    color: var(--primary);
    border: 1px solid #c7d2fe;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    white-space: nowrap;
}
.case-meta-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 12px;
}
.meta-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
}
.meta-item span { display: block; color: var(--muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 3px; }
.meta-item strong { color: var(--text); font-size: 13px; font-weight: 700; }
.snippet-box, .ai-answer-body {
    background: var(--surface);
    border: 1px solid var(--border);
    color: #334155;
    border-radius: 12px;
    padding: 14px;
    line-height: 1.7;
    font-size: 13px;
}

/* Footer */
.footer-note {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    margin-top: 20px;
    padding: 14px 20px;
    border-radius: 14px;
    line-height: 1.6;
}

/* Form controls */
button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.01em !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button.primary {
    background: linear-gradient(135deg, #0f1f4b, #1e3a8a) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(15,31,75,0.25) !important;
}
button.primary:hover { transform: translateY(-1px) !important; }
button.secondary {
    background: #f8fafc !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
textarea, input[type="text"] {
    border-radius: 10px !important;
    border-color: var(--border) !important;
    font-size: 13px !important;
}
label span { font-weight: 600 !important; font-size: 13px !important; color: var(--text) !important; }
.block { gap: 10px !important; }
input, textarea, select {
    color: var(--text) !important;
    background: var(--input-bg) !important;
}

/* ── JAMS Chat — Gradio ChatInterface ── */
.gradio-container .chatbot {
    font-size: 14px !important;
    line-height: 1.65 !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    background: var(--chat-bg) !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}
.gradio-container .chatbot .message.user {
    background: var(--chat-user) !important;
    color: #ffffff !important;
}
.gradio-container .chatbot .message.bot {
    background: var(--chat-bot) !important;
    color: var(--chat-bot-text) !important;
    border: 1px solid #cbd5e1 !important;
}
.gradio-container .multimodal-textbox {
    max-width: 900px !important;
    margin: 0 auto !important;
    border-radius: 14px !important;
}

@media (max-width: 768px) {
    .case-meta-grid { grid-template-columns: 1fr; }
    .main-header h1 { font-size: 22px; }
    .header-top { flex-direction: column; }
    .stat-chip { min-width: 100px; }
}
"""

# -----------------------------
# UI
# -----------------------------
jams_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
)

with gr.Blocks(title="JAMS — Judicial AI Management System") as demo:
    gr.HTML(fa_page_head())
    header_display = gr.HTML(value=dashboard_header_html())

    with gr.Tab("Upload Case"):
        gr.HTML(section_header(
            "Upload & Index Case PDF",
            "Add a judicial PDF with metadata. Text is extracted, chunked, and indexed for Chat and case records.",
            "upload",
        ))
        with gr.Column(elem_classes=["modern-page-card", "upload-card"]):
            gr.HTML(panel_header("Case Information", "Fill in details or auto-fill from PDF", "folder"))
            with gr.Group(elem_classes=["jams-form"]):
                pdf_file = gr.File(label="PDF Document", file_types=[".pdf"])
                case_title = gr.Textbox(label="Case Title", placeholder="Auto-filled after PDF selection")
                court_name = gr.Textbox(label="Court Name", placeholder="e.g. Lahore High Court")
                decision_date = gr.Textbox(label="Decision Date", placeholder="YYYY-MM-DD")
                upload_btn = gr.Button("Upload & Index Case", variant="primary", size="lg")
            gr.HTML('<div class="form-status-divider"><span>Status & Alerts</span></div>')
            upload_output = gr.HTML(
                elem_classes=["upload-status-box"],
                value=professional_alert(
                    "info", "Ready",
                    "Select a PDF file. Case title, court name, and decision date will be auto-filled when possible.",
                ),
            )
        pdf_file.change(
            auto_fill_case_metadata,
            inputs=pdf_file,
            outputs=[case_title, court_name, decision_date, upload_output],
            show_progress="minimal",
        )
        upload_btn.click(
            upload_case_professional,
            inputs=[pdf_file, case_title, court_name, decision_date],
            outputs=[upload_output, pdf_file, case_title, court_name, decision_date],
            show_progress="minimal",
        ).then(dashboard_header_html, outputs=header_display)

    # Hidden modules — enable later by setting visible=True
    with gr.Tab("Summarize PDF", visible=False):
        gr.HTML(section_header("Summarize PDF Case", "Upload any judicial PDF and generate a structured summary.", "📄"))
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                summary_pdf_file = gr.File(label="Case PDF", file_types=[".pdf"])
                summary_case_title = gr.Textbox(label="Case Title (optional)", placeholder="e.g. Sara Bibi vs Muhammad Imran")
                summary_btn = gr.Button("Generate Summary", variant="primary", size="lg")
            with gr.Column(scale=8):
                summary_output = gr.Textbox(label="Case Summary", lines=28, show_label=False)
        summary_btn.click(generate_case_summary, inputs=[summary_pdf_file, summary_case_title], outputs=summary_output, show_progress="minimal")

    with gr.Tab("Search Cases", visible=False):
        gr.HTML(section_header("Search Indexed Cases", "Search by court, title, date, or legal issue.", "🔍"))
        search_query = gr.Textbox(label="Query", lines=4, show_label=False)
        search_btn = gr.Button("Search Cases", variant="primary", size="lg")
        search_output = gr.HTML(value=professional_alert("info", "Search Ready", "Enter a query."))
        search_btn.click(search_ui, inputs=search_query, outputs=search_output, show_progress="minimal")

    with gr.Tab("Ask AI", visible=False):
        gr.HTML(section_header("Ask AI", "Ask a legal research question from indexed sources.", "🤖"))
        ai_question = gr.Textbox(label="Question", lines=5, show_label=False)
        ai_btn = gr.Button("Ask AI", variant="primary", size="lg")
        ai_output = gr.HTML(value=professional_alert("info", "AI Ready", "Ask a question."))
        ai_btn.click(generate_ai_answer_professional, inputs=ai_question, outputs=ai_output, show_progress="minimal")

    with gr.Tab("Chat"):
        chat_temp_docs = gr.State([])
        gr.HTML(section_header(
            "JAMS Chat",
            "Ask about indexed cases or attach a PDF using the upload button in the message box.",
            "chat",
        ))
        gr.ChatInterface(
            fn=jams_chat_fn,
            multimodal=True,
            textbox=gr.MultimodalTextbox(
                placeholder="Message JAMS...",
                file_types=[".pdf"],
                show_label=False,
            ),
            additional_inputs=[chat_temp_docs],
            additional_outputs=[chat_temp_docs],
            examples=[
                [{"text": "What cases are indexed?"}, []],
                [{"text": "Find bail related cases"}, []],
                [{"text": "Summarize the attached PDF and list key legal points"}, []],
            ],
            cache_examples=False,
            fill_height=True,
            show_progress="minimal",
        )

    with gr.Tab("Indexed Cases"):
        gr.HTML(section_header(
            "Indexed Case Records",
            "View all cases uploaded and indexed in the current session.",
            "database",
        ))
        with gr.Column(elem_classes=["modern-page-card", "indexed-page-card"]):
            gr.HTML(f"""
            <div class="page-action-bar">
                <div class="page-action-text">
                    <strong>{fa_icon("cases")} Case Library</strong>
                    Browse all indexed judicial cases. Click refresh after uploading new PDFs.
                </div>
            </div>
            """)
            with gr.Row(elem_classes=["jams-refresh-row"]):
                refresh_btn = gr.Button("Refresh Case List", variant="primary", size="lg")
            gr.HTML('<div class="form-status-divider"><span>Results</span></div>')
            cases_output = gr.HTML(
                elem_classes=["indexed-results-box"],
                value=professional_alert(
                    "info", "No Records Loaded",
                    "Click 'Refresh Case List' to view indexed cases.",
                ),
            )
        refresh_btn.click(
            indexed_cases_html,
            inputs=[],
            outputs=cases_output,
            show_progress="minimal",
        ).then(dashboard_header_html, outputs=header_display)

    gr.HTML("""
    <div class="jams-footer">
        <span class="jams-footer-badge">JAMS</span>
        <span class="jams-footer-text">Judicial AI Management System · Demo · Session data in memory only</span>
    </div>
    """)

demo.queue(default_concurrency_limit=1)
demo.launch(
    share=True,
    debug=False,
    prevent_thread_lock=True,
    css=custom_css,
    theme=jams_theme,
)
