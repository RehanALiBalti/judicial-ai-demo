"""
Judicial AI Case Assistant - Local demo application.

Upload judicial PDFs, index them with FAISS embeddings, search case content,
and ask questions answered by a local Ollama model using only uploaded sources.
"""

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import gradio as gr
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
STORE_PATH = DATA_DIR / "store.json"
FAISS_PATH = DATA_DIR / "faiss.index"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEARCH_TOP_K = 5
AI_CONTEXT_TOP_K = 5

DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _empty_store() -> dict[str, Any]:
    return {"cases": {}, "chunks": []}


def load_store() -> dict[str, Any]:
    if STORE_PATH.exists():
        with open(STORE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return _empty_store()


def save_store(store: dict[str, Any]) -> None:
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# PDF extraction and chunking
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: str) -> tuple[list[tuple[int, str]], str | None]:
  """
  Extract text page-by-page from a PDF.
  Returns (pages, error_message). error_message is set when OCR is required.
  """
  try:
    reader = PdfReader(pdf_path)
  except Exception as exc:
    return [], f"Failed to read PDF: {exc}"

  pages: list[tuple[int, str]] = []
  for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    pages.append((i, text.strip()))

  combined = "\n".join(t for _, t in pages if t).strip()
  if not combined:
    return pages, (
      "No extractable text found in this PDF. "
      "The document likely contains scanned images and requires OCR before indexing."
    )
  return pages, None


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict[str, Any]]:
  """Split page text into overlapping chunks while tracking source page numbers."""
  chunks: list[dict[str, Any]] = []
  for page_num, page_text in pages:
    if not page_text:
      continue
    start = 0
    while start < len(page_text):
      end = start + CHUNK_SIZE
      chunk_text = page_text[start:end].strip()
      if chunk_text:
        chunks.append({"page": page_num, "text": chunk_text})
      if end >= len(page_text):
        break
      start = max(start + 1, end - CHUNK_OVERLAP)
  return chunks


# ---------------------------------------------------------------------------
# Embedding model and FAISS index
# ---------------------------------------------------------------------------

class CaseIndex:
  """Manages sentence-transformer embeddings and a FAISS vector index."""

  def __init__(self) -> None:
    self.model: SentenceTransformer | None = None
    self.index: faiss.IndexFlatIP | None = None
    self.dimension = 0

  def _ensure_model(self) -> SentenceTransformer:
    if self.model is None:
      self.model = SentenceTransformer(EMBEDDING_MODEL)
      self.dimension = self.model.get_sentence_embedding_dimension()
    return self.model

  def _build_index(self, texts: list[str]) -> None:
    model = self._ensure_model()
    if not texts:
      self.index = None
      return
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    self.index = faiss.IndexFlatIP(self.dimension)
    self.index.add(embeddings)

  def load_or_build(self, store: dict[str, Any]) -> None:
    texts = [c["text"] for c in store["chunks"]]
    if FAISS_PATH.exists() and texts:
      self._ensure_model()
      self.index = faiss.read_index(str(FAISS_PATH))
      if self.index.ntotal != len(texts):
        self._build_index(texts)
        faiss.write_index(self.index, str(FAISS_PATH))
    elif texts:
      self._build_index(texts)
      faiss.write_index(self.index, str(FAISS_PATH))
    else:
      self.index = None

  def add_chunks(self, new_texts: list[str], store: dict[str, Any]) -> None:
    if not new_texts:
      return
    model = self._ensure_model()
    new_embeddings = model.encode(new_texts, convert_to_numpy=True, show_progress_bar=False)
    new_embeddings = new_embeddings.astype("float32")
    faiss.normalize_L2(new_embeddings)

    if self.index is None:
      self.index = faiss.IndexFlatIP(self.dimension)
    self.index.add(new_embeddings)
    faiss.write_index(self.index, str(FAISS_PATH))

  def search(self, query: str, top_k: int = SEARCH_TOP_K) -> list[tuple[int, float]]:
    if self.index is None or self.index.ntotal == 0:
      return []
    model = self._ensure_model()
    query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_vec = query_vec.astype("float32")
    faiss.normalize_L2(query_vec)
    k = min(top_k, self.index.ntotal)
    scores, indices = self.index.search(query_vec, k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]


# Global index instance (loaded once at startup)
case_index = CaseIndex()
store_data = load_store()
case_index.load_or_build(store_data)


# ---------------------------------------------------------------------------
# Case upload
# ---------------------------------------------------------------------------

def upload_case(
  title: str,
  court: str,
  decision_date: str,
  pdf_file: str | None,
) -> str:
  title = (title or "").strip()
  court = (court or "").strip()
  decision_date = (decision_date or "").strip()

  if not title:
    return "Error: Case Title is required."
  if not court:
    return "Error: Court Name is required."
  if not decision_date:
    return "Error: Decision Date is required."
  if not pdf_file:
    return "Error: Please upload a PDF file."

  pages, ocr_error = extract_pdf_text(pdf_file)
  if ocr_error:
    return f"Error: {ocr_error}"

  text_chunks = chunk_pages(pages)
  if not text_chunks:
    return "Error: No text chunks could be created from this PDF."

  case_id = f"case_{uuid.uuid4().hex[:8]}"
  dest_name = f"{case_id}.pdf"
  dest_path = UPLOADS_DIR / dest_name
  shutil.copy2(pdf_file, dest_path)

  global store_data
  store_data = load_store()

  start_idx = len(store_data["chunks"])
  new_texts: list[str] = []
  for chunk in text_chunks:
    store_data["chunks"].append(
      {
        "chunk_id": start_idx + len(new_texts),
        "case_id": case_id,
        "page": chunk["page"],
        "text": chunk["text"],
      }
    )
    new_texts.append(chunk["text"])

  store_data["cases"][case_id] = {
    "case_id": case_id,
    "title": title,
    "court": court,
    "decision_date": decision_date,
    "pdf_filename": dest_name,
    "num_chunks": len(new_texts),
    "uploaded_at": datetime.now().isoformat(timespec="seconds"),
  }

  save_store(store_data)
  case_index.add_chunks(new_texts, store_data)

  return (
    f"Successfully indexed case '{title}'.\n"
    f"Case ID: {case_id}\n"
    f"Chunks created: {len(new_texts)}"
  )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _snippet(text: str, max_len: int = 300) -> str:
  text = re.sub(r"\s+", " ", text).strip()
  if len(text) <= max_len:
    return text
  return text[: max_len - 3] + "..."


def search_cases(query: str, top_k: int = SEARCH_TOP_K) -> str:
  query = (query or "").strip()
  if not query:
    return "Please enter a search query."

  store_data_local = load_store()
  if not store_data_local["chunks"]:
    return "No cases indexed yet. Upload a PDF case first."

  results = case_index.search(query, top_k=top_k)
  if not results:
    return "No matching results found."

  lines = [f"Top {len(results)} result(s) for: \"{query}\"\n"]
  for rank, (chunk_idx, score) in enumerate(results, start=1):
    chunk = store_data_local["chunks"][chunk_idx]
    case = store_data_local["cases"].get(chunk["case_id"], {})
    lines.append(f"--- Result {rank} (score: {score:.3f}) ---")
    lines.append(f"Case Title: {case.get('title', 'N/A')}")
    lines.append(f"Case ID: {case.get('case_id', 'N/A')}")
    lines.append(f"Court: {case.get('court', 'N/A')}")
    lines.append(f"Decision Date: {case.get('decision_date', 'N/A')}")
    lines.append(f"Page: {chunk.get('page', 'N/A')}")
    lines.append(f"Snippet: {_snippet(chunk['text'])}")
    lines.append("")

  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama AI Q&A
# ---------------------------------------------------------------------------

def _build_context_chunks(query: str, top_k: int = AI_CONTEXT_TOP_K) -> list[dict[str, Any]]:
  store_data_local = load_store()
  results = case_index.search(query, top_k=top_k)
  context: list[dict[str, Any]] = []
  for chunk_idx, _score in results:
    chunk = store_data_local["chunks"][chunk_idx]
    case = store_data_local["cases"].get(chunk["case_id"], {})
    context.append(
      {
        "case_id": case.get("case_id", "unknown"),
        "title": case.get("title", "Unknown"),
        "court": case.get("court", "Unknown"),
        "decision_date": case.get("decision_date", "Unknown"),
        "page": chunk.get("page"),
        "text": chunk["text"],
      }
    )
  return context


def _format_sources_for_prompt(sources: list[dict[str, Any]]) -> str:
  if not sources:
    return "(No sources available)"
  parts = []
  for i, src in enumerate(sources, start=1):
    parts.append(
      f"[Source {i}]\n"
      f"Case ID: {src['case_id']}\n"
      f"Title: {src['title']}\n"
      f"Court: {src['court']}\n"
      f"Decision Date: {src['decision_date']}\n"
      f"Page: {src.get('page', 'N/A')}\n"
      f"Text: {src['text']}\n"
    )
  return "\n".join(parts)


def _call_ollama(prompt: str) -> str:
  try:
    response = requests.post(
      OLLAMA_URL,
      json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
      },
      timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()
  except requests.ConnectionError:
    return (
      "Error: Cannot connect to Ollama at http://localhost:11434. "
      "Please ensure Ollama is running and the model is pulled:\n"
      f"  ollama pull {OLLAMA_MODEL}"
    )
  except requests.RequestException as exc:
    return f"Error calling Ollama API: {exc}"


def ask_ai(question: str) -> str:
  question = (question or "").strip()
  if not question:
    return "Please enter a question."

  store_data_local = load_store()
  if not store_data_local["chunks"]:
    return "No cases indexed yet. Upload a PDF case before asking questions."

  sources = _build_context_chunks(question)
  if not sources:
    return "No supported source found."

  sources_text = _format_sources_for_prompt(sources)

  prompt = f"""You are a judicial case assistant. Answer ONLY using the provided source excerpts below.
Do NOT invent case names, citations, facts, laws, or details not present in the sources.
If the sources do not contain enough information to answer the question, respond with exactly:
No supported source found.

Format your answer with these four sections:

1. Short Answer
2. Relevant Cases
3. Reasoning
4. Source References

--- PROVIDED SOURCES ---
{sources_text}
--- END SOURCES ---

Question: {question}

Answer:"""

  answer = _call_ollama(prompt)
  if not answer:
    return "No supported source found."
  return answer


# ---------------------------------------------------------------------------
# Indexed cases list
# ---------------------------------------------------------------------------

def list_indexed_cases() -> str:
  store_data_local = load_store()
  cases = store_data_local.get("cases", {})
  if not cases:
    return "No cases indexed yet."

  lines = [f"Total cases: {len(cases)}\n"]
  for case_id in sorted(cases.keys()):
    case = cases[case_id]
    lines.append(f"Case ID: {case['case_id']}")
    lines.append(f"Title: {case['title']}")
    lines.append(f"Court: {case['court']}")
    lines.append(f"Decision Date: {case['decision_date']}")
    lines.append(f"Chunks: {case['num_chunks']}")
    lines.append(f"Uploaded: {case.get('uploaded_at', 'N/A')}")
    lines.append("-" * 40)

  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
  with gr.Blocks(title="Judicial AI Case Assistant") as demo:
    gr.Markdown(
      "# Judicial AI Case Assistant\n"
      "Upload judicial PDF cases, search indexed content, and ask AI questions "
      "using **local Ollama** — answers are grounded only in your uploaded sources."
    )

    with gr.Tab("1. Upload PDF Case"):
      gr.Markdown("Upload a judicial PDF and provide case metadata to index it locally.")
      with gr.Row():
        upload_title = gr.Textbox(label="Case Title", placeholder="e.g. Smith v. Jones")
        upload_court = gr.Textbox(label="Court Name", placeholder="e.g. Supreme Court")
      upload_date = gr.Textbox(label="Decision Date", placeholder="e.g. 2024-03-15")
      upload_pdf = gr.File(label="PDF File", file_types=[".pdf"], type="filepath")
      upload_btn = gr.Button("Upload & Index Case", variant="primary")
      upload_output = gr.Textbox(label="Result", lines=6)

      upload_btn.click(
        upload_case,
        inputs=[upload_title, upload_court, upload_date, upload_pdf],
        outputs=upload_output,
      )

    with gr.Tab("2. Search Cases"):
      gr.Markdown("Search indexed case chunks using semantic vector search (FAISS).")
      search_query = gr.Textbox(
        label="Search Query",
        placeholder="e.g. breach of contract damages",
        lines=2,
      )
      search_btn = gr.Button("Search", variant="primary")
      search_output = gr.Textbox(label="Search Results", lines=20)

      search_btn.click(search_cases, inputs=search_query, outputs=search_output)

    with gr.Tab("3. Ask AI"):
      gr.Markdown(
        f"Ask a question answered by local Ollama (`{OLLAMA_MODEL}`). "
        "The AI uses only retrieved source chunks — no external knowledge."
      )
      ai_question = gr.Textbox(
        label="Your Question",
        placeholder="e.g. What was the court's ruling on damages?",
        lines=3,
      )
      ai_btn = gr.Button("Ask AI", variant="primary")
      ai_output = gr.Textbox(label="AI Answer", lines=25)

      ai_btn.click(ask_ai, inputs=ai_question, outputs=ai_output)

    with gr.Tab("4. Indexed Cases"):
      gr.Markdown("View all locally indexed cases.")
      list_btn = gr.Button("Refresh List", variant="secondary")
      list_output = gr.Textbox(label="Indexed Cases", lines=20)

      list_btn.click(list_indexed_cases, outputs=list_output)
      demo.load(list_indexed_cases, outputs=list_output)

  return demo


if __name__ == "__main__":
  print("Starting Judicial AI Case Assistant...")
  print(f"Data directory: {DATA_DIR}")
  print(f"Ollama model: {OLLAMA_MODEL}")
  app = build_ui()
  app.launch()
