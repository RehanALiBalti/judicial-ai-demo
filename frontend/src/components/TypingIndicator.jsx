export default function TypingIndicator({ label = "JAMS is thinking" }) {
  return (
    <div className="typing-indicator" role="status" aria-live="polite">
      <span className="typing-label">{label}</span>
      <span className="typing-dots">
        <span />
        <span />
        <span />
      </span>
    </div>
  );
}
