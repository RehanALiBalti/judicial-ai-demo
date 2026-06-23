import { useEffect } from "react";

const ICONS = {
  success: "✓",
  error: "!",
  warning: "⚠",
  info: "i",
};

export default function Toast({ type = "info", text, onClose }) {
  useEffect(() => {
    const id = setTimeout(() => onClose?.(), 4200);
    return () => clearTimeout(id);
  }, [text, onClose]);

  if (!text) return null;

  return (
    <div className={`toast toast-${type} toast-enter`} role="status">
      <span className="toast-icon">{ICONS[type] || "i"}</span>
      <span className="toast-text">{text}</span>
      <button type="button" className="toast-close" onClick={onClose} aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}
