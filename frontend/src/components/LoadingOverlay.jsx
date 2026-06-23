import Spinner from "./Spinner";

export default function LoadingOverlay({ message = "Processing…", submessage = "" }) {
  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-overlay-card">
        <Spinner size="lg" />
        <p className="loading-overlay-title">{message}</p>
        {submessage && <p className="loading-overlay-sub">{submessage}</p>}
      </div>
    </div>
  );
}
