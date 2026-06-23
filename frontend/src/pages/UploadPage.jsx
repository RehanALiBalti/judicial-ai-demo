import { useRef, useState } from "react";
import { extractMetadata, uploadCase } from "../api/client";
import LoadingOverlay from "../components/LoadingOverlay";
import Spinner from "../components/Spinner";
import { SkeletonLine } from "../components/Skeleton";

export default function UploadPage({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [caseTitle, setCaseTitle] = useState("");
  const [courtName, setCourtName] = useState("");
  const [decisionDate, setDecisionDate] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [metaLoading, setMetaLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [alert, setAlert] = useState({
    type: "info",
    title: "Ready",
    message: "Select a PDF file. Metadata will be auto-filled when possible.",
  });
  const fileRef = useRef(null);

  const loading = metaLoading || uploadLoading;

  const handleFile = async (selected) => {
    setFile(selected);
    if (!selected) return;
    setMetaLoading(true);
    try {
      const meta = await extractMetadata(selected);
      setCaseTitle(meta.case_title || "");
      setCourtName(meta.court_name || "");
      setDecisionDate(meta.decision_date || "");
      setAlert({
        type: meta.status || "info",
        title: meta.status === "success" ? "Auto-Fill Complete" : "Auto-Fill",
        message: meta.message || "Review metadata before uploading.",
      });
    } catch (err) {
      setAlert({ type: "error", title: "Auto-Fill Failed", message: err.message });
    } finally {
      setMetaLoading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped?.name?.toLowerCase().endsWith(".pdf")) {
      handleFile(dropped);
      if (fileRef.current) fileRef.current.files = e.dataTransfer.files;
    } else {
      setAlert({ type: "warning", title: "PDF Only", message: "Please drop a PDF file." });
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setAlert({ type: "warning", title: "PDF Required", message: "Select a PDF file first." });
      return;
    }
    setUploadLoading(true);
    try {
      const result = await uploadCase({ file, caseTitle, courtName, decisionDate });
      if (result.success) {
        setAlert({ type: "success", title: "Uploaded", message: result.message });
        setFile(null);
        setCaseTitle("");
        setCourtName("");
        setDecisionDate("");
        if (fileRef.current) fileRef.current.value = "";
        onUploaded?.();
      } else {
        setAlert({ type: "error", title: "Upload Failed", message: result.message });
      }
    } catch (err) {
      setAlert({ type: "error", title: "Upload Failed", message: err.message });
    } finally {
      setUploadLoading(false);
    }
  };

  return (
    <div className="page upload-page">
      {uploadLoading && (
        <LoadingOverlay
          message="Indexing case PDF…"
          submessage="Extracting text, chunking, and building search index"
        />
      )}

      <div className="page-header animate-rise">
        <div className="page-kicker">Case Database</div>
        <h2>Upload &amp; Index Case PDF</h2>
        <p>Add judicial PDFs to the searchable case database for AI chat.</p>
      </div>

      <div className={`alert alert-${alert.type} animate-slide-down`}>
        <strong>{alert.title}</strong>
        <span>{alert.message}</span>
      </div>

      <form className="upload-grid" onSubmit={handleUpload}>
        <div className="card drop-zone hover-lift">
          <label
            className={`drop-label ${dragOver ? "drop-label-active" : ""} ${file ? "drop-label-filled" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".pdf"
              disabled={loading}
              onChange={(e) => handleFile(e.target.files?.[0] || null)}
            />
            <div className={`drop-icon ${metaLoading ? "drop-icon-loading" : ""}`}>
              {metaLoading ? <Spinner size="lg" /> : "📄"}
            </div>
            <div className="drop-title">
              {metaLoading ? "Reading PDF metadata…" : file ? file.name : "Drop PDF here or click to browse"}
            </div>
            <div className="drop-hint">Text-based judicial PDFs work best</div>
          </label>
        </div>

        <div className="card form-card">
          {metaLoading ? (
            <div className="form-skeleton">
              <SkeletonLine width="40%" />
              <SkeletonLine />
              <SkeletonLine />
              <SkeletonLine />
              <div className="skeleton skeleton-btn" />
            </div>
          ) : (
            <>
              <div className="field">
                <label htmlFor="case-title">Case Title</label>
                <input
                  id="case-title"
                  value={caseTitle}
                  onChange={(e) => setCaseTitle(e.target.value)}
                  placeholder="Auto-filled after PDF selection"
                  disabled={loading}
                />
              </div>
              <div className="field">
                <label htmlFor="court-name">Court Name</label>
                <input
                  id="court-name"
                  value={courtName}
                  onChange={(e) => setCourtName(e.target.value)}
                  placeholder="e.g. Lahore High Court"
                  disabled={loading}
                />
              </div>
              <div className="field">
                <label htmlFor="decision-date">Decision Date</label>
                <input
                  id="decision-date"
                  value={decisionDate}
                  onChange={(e) => setDecisionDate(e.target.value)}
                  placeholder="YYYY-MM-DD"
                  disabled={loading}
                />
              </div>
              <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
                {uploadLoading ? (
                  <>
                    <Spinner size="sm" />
                    Indexing…
                  </>
                ) : (
                  "Upload & Index Case"
                )}
              </button>
            </>
          )}
        </div>
      </form>

      <div className="steps-row">
        {[
          { n: "1", title: "Select PDF", desc: "Text-based judicial PDFs work best." },
          { n: "2", title: "Auto-fill metadata", desc: "Title, court and date are detected." },
          { n: "3", title: "Index case", desc: "Case becomes available in AI chat." },
        ].map((step, i) => (
          <div
            key={step.n}
            className="step-card hover-lift animate-rise"
            style={{ animationDelay: `${0.1 + i * 0.06}s` }}
          >
            <span className="step-num">{step.n}</span>
            <b>{step.title}</b>
            <span>{step.desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
