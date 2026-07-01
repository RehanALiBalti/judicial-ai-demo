import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { fetchChatSuggestions, sendChat } from "../api/client";
import Spinner from "../components/Spinner";
import Toast from "../components/Toast";
import TypingIndicator from "../components/TypingIndicator";
import { IconAttach, IconClose, IconPdf, IconSend } from "../components/ChatIcons";

const DEFAULT_SUGGESTIONS = [
  "How many cases are indexed?",
  "zamanat / bail cases dikhao",
  "F.C.P.L.A. No.73-K of 2026 samjhao",
  "CASE-055 summarize",
  "What can you ask JAMS?",
];

export default function ChatPage({ onStatsChange }) {
  const [history, setHistory] = useState([]);
  const [tempDocs, setTempDocs] = useState([]);
  const [chatContext, setChatContext] = useState({});
  const [suggestions, setSuggestions] = useState(DEFAULT_SUGGESTIONS);
  const [message, setMessage] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const [focused, setFocused] = useState(false);
  const fileRef = useRef(null);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  const canSend = Boolean(message.trim() || file) && !loading;

  useEffect(() => {
    fetchChatSuggestions()
      .then((data) => {
        if (Array.isArray(data?.suggestions) && data.suggestions.length > 0) {
          setSuggestions(data.suggestions);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [message]);

  const showToast = (type, text) => setToast({ type, text });

  const scrollDown = () => {
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 80);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim() && !file) {
      showToast("warning", "Type a message or attach a PDF.");
      return;
    }
    setLoading(true);
    try {
      const result = await sendChat({ message, history, tempDocs, chatContext, file });
      setHistory(result.history || []);
      setTempDocs(result.temp_docs || []);
      setChatContext(result.chat_context || {});
      setMessage("");
      setFile(null);
      if (fileRef.current) fileRef.current.value = "";
      showToast(result.status || "success", result.message || "Done");
      onStatsChange?.();
      scrollDown();
    } catch (err) {
      showToast("error", err.message);
    } finally {
      setLoading(false);
    }
  };

  const pickSuggestion = (text) => {
    setMessage(text);
  };

  const clearChat = () => {
    setHistory([]);
    setTempDocs([]);
    setChatContext({});
    setMessage("");
    setFile(null);
    if (fileRef.current) fileRef.current.value = "";
    showToast("info", "Chat cleared.");
  };

  const removeFile = () => {
    setFile(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <div className="page chat-page">
      <div className="page-header animate-rise">
        <div className="page-kicker">AI Assistant</div>
        <h2>How can JAMS help you today?</h2>
        <p>Ask about indexed cases or attach a PDF for session-only analysis.</p>
      </div>

      {toast && (
        <Toast type={toast.type} text={toast.text} onClose={() => setToast(null)} />
      )}

      {tempDocs.length > 0 && (
        <div className="attach-banner animate-slide-down">
          <span className="attach-dot pulse-dot" />
          PDF attached for this chat session ({tempDocs.length} chunks indexed)
        </div>
      )}

      <div className="chat-panel animate-rise" style={{ animationDelay: "0.08s" }}>
        <div className="chat-messages">
          {history.length === 0 && !loading && (
            <div className="chat-empty animate-fade">
              <div className="chat-empty-icon float-icon">⚖️</div>
              <h3>Start a conversation</h3>
              <p>Upload a case PDF in Upload Case tab, or attach a PDF here for quick Q&amp;A.</p>
              <div className="suggestion-chips">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    type="button"
                    className="suggestion-chip"
                    onClick={() => pickSuggestion(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {history.map((msg, i) => (
            <div
              key={i}
              className={`chat-bubble chat-bubble-${msg.role} message-enter`}
              style={{ animationDelay: `${Math.min(i * 0.04, 0.2)}s` }}
            >
              <div className="chat-role">{msg.role === "user" ? "You" : "JAMS"}</div>
              <div className="chat-content">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
            </div>
          ))}

          {loading && (
            <div className="chat-bubble chat-bubble-assistant message-enter">
              <div className="chat-role">JAMS</div>
              <div className="chat-content">
                <TypingIndicator />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="composer-zone">
          <form
            className={`composer-form ${loading ? "composer-busy" : ""}`}
            onSubmit={handleSubmit}
          >
            <div className={`composer-box ${focused ? "is-focused" : ""} ${file ? "has-file" : ""}`}>
              {file && (
                <div className="composer-file-preview animate-slide-down">
                  <div className="composer-file-badge">
                    <IconPdf />
                    <span className="composer-file-name">{file.name}</span>
                  </div>
                  <button
                    type="button"
                    className="composer-file-remove"
                    onClick={removeFile}
                    aria-label="Remove attachment"
                  >
                    <IconClose />
                  </button>
                </div>
              )}

              <textarea
                ref={textareaRef}
                className="composer-input"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Message JAMS…"
                rows={1}
                disabled={loading}
                onFocus={() => setFocused(true)}
                onBlur={() => setFocused(false)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />

              <div className="composer-toolbar">
                <label
                  className={`composer-tool-btn ${file ? "is-active" : ""}`}
                  title="Attach PDF"
                >
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".pdf"
                    disabled={loading}
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                  <IconAttach />
                  <span className="composer-tool-label">Attach</span>
                </label>

                <div className="composer-toolbar-spacer" />

                <button
                  type="submit"
                  className={`composer-send-btn ${canSend ? "is-ready" : ""}`}
                  disabled={!canSend}
                  aria-label="Send message"
                >
                  {loading ? <Spinner size="sm" /> : <IconSend />}
                </button>
              </div>
            </div>
          </form>

          <div className="composer-meta">
            <button
              type="button"
              className="composer-clear-btn"
              onClick={clearChat}
              disabled={loading}
            >
              Clear chat
            </button>
            <span className="composer-disclaimer">
              JAMS answers from your indexed cases &amp; attached PDFs only
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
