const API_BASE = import.meta.env.VITE_API_URL || "";

async function handleResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || data.message || `Request failed (${res.status})`);
  }
  return data;
}

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/api/stats`);
  return handleResponse(res);
}

export async function fetchCases() {
  const res = await fetch(`${API_BASE}/api/cases`);
  return handleResponse(res);
}

export async function extractMetadata(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/cases/extract-metadata`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function uploadCase({ file, caseTitle, courtName, decisionDate }) {
  const form = new FormData();
  form.append("file", file);
  form.append("case_title", caseTitle);
  form.append("court_name", courtName);
  form.append("decision_date", decisionDate);
  const res = await fetch(`${API_BASE}/api/cases/upload`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function sendChat({ message, history, tempDocs, file }) {
  const form = new FormData();
  form.append("message", message || "");
  form.append("history", JSON.stringify(history || []));
  form.append("temp_docs", JSON.stringify(tempDocs || []));
  if (file) form.append("file", file);
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function fetchFccpStatus() {
  const res = await fetch(`${API_BASE}/api/scraper/fccp/status`);
  return handleResponse(res);
}

export async function startFccpSync({ startPage = 1, endPage } = {}) {
  const params = new URLSearchParams({ start_page: String(startPage) });
  if (endPage) params.set("end_page", String(endPage));
  const res = await fetch(`${API_BASE}/api/scraper/fccp/sync?${params}`, {
    method: "POST",
  });
  return handleResponse(res);
}
