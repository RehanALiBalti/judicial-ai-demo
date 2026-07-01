"""Chat intent resolution, query normalization, and grounded answer generation.

Designed for many users with different writing styles (English, Roman Urdu, informal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

CHAT_SUGGESTIONS: List[str] = [
    "How many cases are indexed?",
    "Find cases about bail / zamanat",
    "F.C.P.L.A. No.73-K of 2026 explain",
    "CASE-055 summarize",
    "court ne raihat di — find cases",
    "mazeed sources",
]

# Roman Urdu / informal → search-friendly English (applied before normalize_text)
_PHRASE_REWRITES: Tuple[Tuple[str, str], ...] = (
    (r"\bsamjha[o]?\b", " explain "),
    (r"\bsamjh[a]?[ae]?\b", " explain "),
    (r"\bbata[o]?\b", " tell "),
    (r"\bbtao\b", " tell "),
    (r"\bdikha[o]?\b", " show "),
    (r"\bdhund[o]?\b", " find "),
    (r"\bdhundo\b", " find "),
    (r"\bnikal\b", " find "),
    (r"\bzamanat\b", " bail "),
    (r"\braihat\b", " relief "),
    (r"\badalat\b", " court "),
    (r"\bfaisla\b", " decision "),
    (r"\bhukm\b", " order "),
    (r"\bfaisla\b", " decision "),
    (r"\bmazeed\b", " more "),
    (r"\bmazid\b", " more "),
    (r"\bzyada\b", " more "),
    (r"\bkesy\b", " how "),
    (r"\bkese\b", " how "),
    (r"\bkya\s+hal\b", " how are you "),
    (r"\bkaise\s+ho\b", " how are you "),
    (r"\bkitn[eyi]\b", " how many "),
    (r"\bkitne\b", " how many "),
    (r"\bwo\b", " that "),
    (r"\bye\b", " this "),
    (r"\bwoh\b", " that "),
    (r"\byeh\b", " this "),
    (r"\bisi\b", " this "),
    (r"\busi\b", " that "),
    (r"\bwahi\b", " same "),
    (r"\bplz\b", " please "),
    (r"\bpls\b", " please "),
    (r"\bmeherbani\b", " please "),
    (r"\bkrna\b", " do "),
    (r"\bkarna\b", " do "),
    (r"\bkrdo\b", " do "),
    (r"\bkar\s*do\b", " do "),
    (r"\bgimme\b", " give me "),
    (r"\bwanna\b", " want to "),
    (r"\bgonna\b", " going to "),
    (r"\bu\b", " you "),
    (r"\br\b", " are "),
    (r"\bur\b", " your "),
    (r"\bwhats\b", " what is "),
    (r"\bwat\b", " what "),
    (r"\bsummery\b", " summary "),
    (r"\bsummarise\b", " summarize "),
)

_TOPIC_SYNONYMS: Dict[str, List[str]] = {
    "bail": ["bail", "zamanat", "pre-arrest", "post-arrest", "498", "497"],
    "relief": ["relief", "remedy", "raihet", "granted"],
    "rent": ["rent", "tenancy", "ejectment", "landlord"],
    "rights": ["rights", "human rights", "fundamental", "article"],
    "conviction": ["conviction", "sentence", "appeal", "ppc"],
}


@dataclass
class ParsedUserQuery:
    original: str
    normalized: str
    resolved: str
    follow_context: str
    search_queries: List[str] = field(default_factory=list)
    intent: str = "general_search"


def normalize_user_message(text: str) -> str:
    """Unify informal English + Roman Urdu before routing."""
    t = (text or "").strip()
    if not t:
        return ""
    for pattern, replacement in _PHRASE_REWRITES:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def prepare_chat_query(query: str, history: List[Dict[str, str]]) -> ParsedUserQuery:
    """Single entry: normalize, resolve follow-ups, classify intent, build search variants."""
    original = (query or "").strip()
    normalized = normalize_user_message(original)
    resolved, follow_context = resolve_follow_up_question(normalized, history)
    intent = classify_intent(resolved, history)
    search_queries = build_retrieval_queries(resolved)
    return ParsedUserQuery(
        original=original,
        normalized=normalized,
        resolved=resolved,
        follow_context=follow_context,
        search_queries=search_queries,
        intent=intent,
    )


def classify_intent(query: str, history: List[Dict[str, str]]) -> str:
    """Score-based intent for varied phrasing."""
    from backend.core import (
        extract_case_id_from_query,
        is_broad_legal_topic_query,
        is_case_inventory_query,
        is_conversational_query,
        is_judge_query,
        is_topic_case_request,
        is_topic_more_request,
        normalize_text,
        wants_case_content_answer,
    )

    qn = normalize_text(query)
    if not qn:
        return "empty"

    if is_too_vague_for_search(query):
        return "too_vague"
    if is_capabilities_query(query):
        return "capabilities"
    if is_topic_more_request(query):
        return "more_results"
    if is_conversational_query(query):
        return "conversational"
    if is_case_inventory_query(query):
        return "inventory"
    if is_judge_query(query):
        return "judge"
    if is_topic_case_request(query) or is_broad_legal_topic_query(query):
        return "topic_search"

    find_signals = (
        "find", "search", "show", "list", "dhund", "dikha", "nikal", "cases",
        "mila", "matching", "dhundo", "dikhao",
    )
    if "cases" in qn and any(s in qn for s in find_signals):
        return "topic_search"

    if extract_case_id_from_query(query):
        return "case_explain" if wants_case_content_answer(query) else "case_lookup"
    if wants_case_content_answer(query):
        return "case_explain"

    explain_signals = (
        "explain", "summar", "describe", "bata", "samjh", "detail", "overview",
        "kya hua", "kya faisla", "decision kya",
    )
    if any(s in qn for s in explain_signals):
        return "case_explain"

    if any(s in qn for s in find_signals):
        return "topic_search"

    # Short follow-up after history with sources
    if history and len(qn.split()) <= 8:
        from backend.core import assistant_has_case_sources

        for msg in reversed(history):
            if msg.get("role") == "assistant" and assistant_has_case_sources(msg.get("content", "")):
                return "case_explain"

    return "general_search"


def is_too_vague_for_search(query: str) -> bool:
    from backend.core import normalize_text

    qn = normalize_text(query)
    if not qn:
        return True
    words = qn.split()
    if len(words) > 4:
        return False
    vague_sets = (
        {"something", "anything", "help", "info"},
        {"tell", "me", "something"},
        {"kuch", "batao"},
        {"koi", "bat"},
        {"help", "me"},
        {"batao", "kuch"},
        {"tell", "me"},
    )
    token_set = set(words)
    if any(token_set <= vs or token_set == vs for vs in vague_sets):
        return True
    if qn in ("help", "info", "kuch", "something", "anything", "batao", "tell me"):
        return True
    return False


def reply_vague_query_help() -> str:
    return """I need a **bit more detail** to search your judgments accurately.

**Try any of these styles (English or Roman Urdu):**
- Case number: `F.C.P.L.A. No.73-K of 2026 explain`
- Case ID: `CASE-055 summarize`
- Topic: `bail / zamanat cases`, `court granted relief`
- Follow-up: `explain it`, `mazeed`, `aur batao`
- Records: `kitne cases indexed hain?`

Everyone can write differently — case number ya legal topic likh dein, main samajh jaunga."""


def is_capabilities_query(query: str) -> bool:
    from backend.core import normalize_text

    q = normalize_text(normalize_user_message(query))
    phrases = (
        "what can you do", "what can you ask", "what can i ask", "what should i ask",
        "how to use jams", "how to use this", "help me use",
        "what questions", "example questions", "sample questions",
        "kya puch", "kya poch", "kaise use", "madad chahiye",
        "supported questions", "what do you support", "kis tarah poch",
        "kaise sawal", "examples dikhao",
    )
    return any(p in q for p in phrases)


def reply_capabilities() -> str:
    return """### JAMS — koi bhi style chalega

**Single case (best accuracy)**
- `CASE-055 summarize` / `CASE-055 samjhao`
- `F.C.P.L.A. No.73-K of 2026 explain`
- `Writ 7652-26 KHIZAR HAYYAT batao`

**Topic search**
- `Find bail cases` / `zamanat ke cases`
- `court ne raihat di` / `granted relief`
- `human rights cases`

**Follow-up (pehle jawab ke baad)**
- `more` / `mazeed` / `aur sources`
- `explain it` / `ye samjhao` / `detail do`

**PDF attach**
- PDF attach → `explain` / `samjhao` / `summary`

**Records**
- `How many cases?` / `kitne cases hain?`

**Tips**
- Case number, CASE-ID, ya topic (bail, relief, rent) likhein.
- Roman Urdu + informal English dono chalte hain.
- Agar jawab galat lage → case number ke sath dubara poochein.

_Answers sirf indexed judgments + attached PDF se — general internet law nahi._"""


def resolve_follow_up_question(
    query: str,
    history: List[Dict[str, str]],
) -> Tuple[str, str]:
    """Expand vague follow-ups using prior chat (many phrasings)."""
    from backend.core import extract_case_ids_from_text, normalize_text

    q = normalize_user_message(query)
    qn = normalize_text(q)
    if not qn or not history:
        return q, ""

    vague_markers = (
        "explain it", "summarize it", "summary of it", "tell me more",
        "more detail", "more about", "what about", "explain this",
        "summarize this", "about it", "about this", "same case",
        "isi case", "wahi case", "ye case", "is case", "decision kya thi",
        "aur batao", "aur bata", "detail do", "poora batao", "pura batao",
        "full detail", "iske bare", "is ke bare", "is case me", "isi me",
        "samjhao is", "explain is", "ye wala", "wo wala", "same topic",
        "usi topic", "isi topic", "pehlay wala", "pichla sawal",
        "continue", "go on", "aage batao",
    )
    short_vague = len(qn.split()) <= 8 and any(
        w in qn.split()
        for w in ("it", "this", "that", "wahi", "ye", "ya", "wo", "isi", "usi", "same")
    )
    if not any(v in qn for v in vague_markers) and not short_vague:
        return q, ""

    for msg in reversed(history):
        content = msg.get("content", "") or ""
        case_ids = extract_case_ids_from_text(content)
        title_match = re.search(r"\*\*Title:\*\*\s*(.+)", content)
        if not title_match:
            title_match = re.search(r"###\s*CASE-\d+\s*[—\-]\s*(.+)", content)
        if title_match:
            title = title_match.group(1).strip().split("\n")[0]
            cid = case_ids[0] if case_ids else ""
            expanded = f"{title} — {q}"
            note = f"Follow-up about {cid}" if cid else f"Follow-up about {title[:60]}"
            return expanded, note
        if case_ids and msg.get("role") == "assistant":
            return f"{q} (case {case_ids[0]})", f"Follow-up about {case_ids[0]}"

    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        text = re.sub(r"^📎.*?\n\n", "", msg.get("content", ""), flags=re.DOTALL).strip()
        text = normalize_user_message(text)
        if len(text.split()) >= 3 and not is_capabilities_query(text) and not is_too_vague_for_search(text):
            return f"{text} — {q}", "Follow-up to your earlier question"

    return q, ""


def build_retrieval_query(query: str) -> str:
    from backend.core import get_query_tokens, normalize_text

    framing = {
        "please", "kindly", "tell", "explain", "describe", "show", "give",
        "find", "search", "indexed", "relevant", "cases", "case", "about",
        "regarding", "related", "involving", "jams", "can", "you", "me", "my",
        "the", "a", "an", "what", "how", "is", "are", "was", "were", "do",
        "krna", "karna", "krdo", "kar", "mujhe", "mujhy", "batao", "bata",
        "samjhao", "samjha", "dikhao", "dhundo", "koi", "kuch", "ye", "wo",
    }
    tokens = [t for t in get_query_tokens(normalize_user_message(query)) if t not in framing]
    if len(tokens) >= 1:
        return " ".join(tokens)
    qn = normalize_text(query)
    return qn if qn else query.strip()


def build_retrieval_queries(query: str) -> List[str]:
    """Multiple search variants for different user phrasings."""
    from backend.core import normalize_text

    q = normalize_user_message(query)
    primary = build_retrieval_query(q)
    variants: List[str] = [q.strip(), primary]

    qn = normalize_text(q)
    for key, syns in _TOPIC_SYNONYMS.items():
        if key in qn or any(s in qn for s in syns):
            variants.extend(syns[:3])

    # Word-order independent: join significant tokens
    tokens = [t for t in qn.split() if len(t) > 2]
    if len(tokens) >= 2:
        variants.append(" ".join(tokens[:6]))

    seen: set = set()
    out: List[str] = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out[:5]


def merge_search_results(
    result_lists: List[List[Dict[str, Any]]],
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    from backend.core import deduplicate_results

    merged: List[Dict[str, Any]] = []
    for batch in result_lists:
        merged.extend(batch)
    return deduplicate_results(merged, max_results=max_results)


def search_with_query_variants(
    search_queries: List[str],
    *,
    top_k: int = 8,
    case_ids: List[str] | None = None,
    balanced: bool = False,
) -> List[Dict[str, Any]]:
    """Run retrieval with multiple phrasings; merge best hits."""
    from backend.core import (
        search_documents_by_keyword,
        search_indexed_docs,
        search_indexed_docs_balanced_courts,
        search_indexed_docs_global,
    )

    batches: List[List[Dict[str, Any]]] = []
    for q in search_queries[:4]:
        if not q:
            continue
        if case_ids:
            hits = search_indexed_docs(q, top_k=top_k, case_ids=case_ids)
        elif balanced:
            hits = search_indexed_docs_balanced_courts(q, top_k=top_k)
            if not hits:
                hits = search_indexed_docs_global(q, top_k=top_k, diverse_cases=True)
        else:
            hits = search_indexed_docs_global(q, top_k=top_k, diverse_cases=False)
        if hits:
            batches.append(hits)

    merged = merge_search_results(batches, max_results=top_k)
    if not merged and search_queries:
        kw = search_documents_by_keyword(search_queries[0], top_k=top_k)
        if kw:
            merged = kw
    return merged


def score_result_relevance(query: str, result: Dict[str, Any]) -> int:
    from backend.core import get_query_tokens, normalize_text

    blob = normalize_text(
        f"{result.get('title', '')} {result.get('text', '')} {result.get('author_judge', '')}"
    )
    tokens = get_query_tokens(normalize_user_message(query))
    if not tokens:
        return 0
    score = 0
    for token in tokens:
        if len(token) > 5 and token in blob:
            score += 3
        elif token in blob:
            score += 1
    return score


def filter_relevant_results(
    query: str,
    results: List[Dict[str, Any]],
    max_results: int = 8,
) -> List[Dict[str, Any]]:
    if not results:
        return []
    scored = [(score_result_relevance(query, item), item) for item in results]
    scored.sort(key=lambda x: (-x[0], str(x[1].get("case_id", ""))))
    top_score = scored[0][0]
    if top_score == 0:
        return results[:max_results]
    threshold = max(1, int(top_score * 0.35))
    filtered = [item for score, item in scored if score >= threshold]
    return (filtered or [scored[0][1]])[:max_results]


def generate_grounded_answer(
    user_question: str,
    results: List[Dict[str, Any]],
    *,
    context_note: str = "",
    max_tokens: int = 700,
) -> str:
    from backend.core import build_sources_text, format_topic_case_cards, generate_from_model

    if not results:
        return (
            "I could not find enough indexed text to answer that question.\n\n"
            "Try a **case number**, **CASE-ID**, or a clearer legal topic "
            "(e.g. *bail / zamanat*, *relief / raihat*, *rent*)."
        )

    sources_text = build_sources_text(results, max_chars_per_source=900)
    context_line = f"\nNote: {context_note}\n" if context_note else ""
    display_q = normalize_user_message(user_question) or user_question

    prompt = f"""You are JAMS, a precise judicial research assistant for Pakistan court judgments.
Users write in different styles (English, Roman Urdu, informal). Interpret the question naturally.

STRICT RULES:
1. Use ONLY the sources below. Do not invent facts, cases, citations, or outcomes.
2. Cite each key fact inline as (Case ID, Page N).
3. If sources only partially answer, say what is supported and what is missing.
4. Ignore earlier greetings — answer the CURRENT question only.
5. Plain clear prose. No fake headings "Relevant Source" or "Reasoning".
6. If sources cannot answer, say: "The indexed excerpts do not contain enough information."
{context_line}
USER QUESTION (any style):
{display_q}

SOURCES:
{sources_text}

Structure:
1. Direct answer (2–5 sentences, with citations)
2. Key points (bullets, if helpful)
3. Confidence: High / Medium / Low

Answer:"""
    try:
        answer = generate_from_model(prompt, max_new_tokens=max_tokens).strip()
    except Exception as exc:
        return f"AI generation failed: {exc}"
    if answer.startswith("Error"):
        return format_topic_case_cards(results[:4])
    return answer
