import { useCallback, useEffect, useState } from "react";
import { fetchLhcStatus, startLhcSync } from "../api/client";
import Spinner from "../components/Spinner";
import LoadingOverlay from "../components/LoadingOverlay";

export default function LhcPage({ onSynced }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [year, setYear] = useState("");
  const [downloadLimit, setDownloadLimit] = useState(50);
  const [alert, setAlert] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchLhcStatus();
      setStatus(data);
      setSyncing(Boolean(data.sync_running));
    } catch (err) {
      setAlert({ type: "error", title: "Error", message: err.message });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, syncing ? 4000 : 15000);
    return () => clearInterval(id);
  }, [load, syncing]);

  const runSync = async (metadataOnly) => {
    setAlert(null);
    try {
      const result = await startLhcSync({
        year: year || undefined,
        metadataOnly,
        downloadLimit: metadataOnly ? undefined : downloadLimit,
      });
      setAlert({ type: "info", title: "Sync Started", message: result.message });
      setSyncing(true);
    } catch (err) {
      setAlert({ type: "error", title: "Sync Failed", message: err.message });
    }
  };

  useEffect(() => {
    if (status?.last_sync_result && !status?.sync_running && syncing) {
      setSyncing(false);
      onSynced?.();
      const r = status.last_sync_result;
      if (r.error) {
        setAlert({ type: "error", title: "Sync Error", message: r.error });
      } else if (r.metadata_only) {
        setAlert({
          type: "success",
          title: "Metadata Saved",
          message: `Listed ${r.scraped} judgments (site reports ${r.total_reported ?? "?"}).`,
        });
      } else {
        setAlert({
          type: "success",
          title: "Sync Complete",
          message: `Downloaded: ${r.downloaded}, Skipped (local): ${r.skipped_existing ?? 0}, Already indexed: ${r.skipped_indexed ?? 0}, New indexed: ${r.indexed}`,
        });
      }
    }
  }, [status, syncing, onSynced]);

  const items = status?.items || [];
  const preview = items.slice(0, 100);

  return (
    <div className="page fccp-page">
      {syncing && (
        <LoadingOverlay
          message="Syncing LHC judgments…"
          submessage="Fetching from data.lhc.gov.pk (may take 1–2 minutes for full list)"
        />
      )}

      <div className="page-header animate-rise">
        <div className="page-kicker">Dataset Builder</div>
        <h2>LHC Judgments Import</h2>
        <p>
          Scrape <strong>Judgments Approved for Reporting</strong> from{" "}
          <a
            href="https://data.lhc.gov.pk/reported_judgments/judgments_approved_for_reporting"
            target="_blank"
            rel="noreferrer"
          >
            Lahore High Court
          </a>
          . Filter <em>All Courts</em> returns ~4683 records in one request.
        </p>
      </div>

      {alert && (
        <div className={`alert alert-${alert.type} animate-slide-down`}>
          <strong>{alert.title}</strong>
          <span>{alert.message}</span>
        </div>
      )}

      <div className="fccp-stats-grid">
        <div className="fccp-stat-card hover-lift">
          <span className="fccp-stat-value">{status?.total_reported ?? status?.total_items ?? "—"}</span>
          <span className="fccp-stat-label">On LHC site</span>
        </div>
        <div className="fccp-stat-card hover-lift">
          <span className="fccp-stat-value">{status?.total_items ?? "—"}</span>
          <span className="fccp-stat-label">In manifest</span>
        </div>
        <div className="fccp-stat-card hover-lift">
          <span className="fccp-stat-value">{status?.downloaded ?? "—"}</span>
          <span className="fccp-stat-label">PDFs saved</span>
        </div>
        <div className="fccp-stat-card hover-lift">
          <span className="fccp-stat-value">{status?.indexed ?? "—"}</span>
          <span className="fccp-stat-label">Indexed</span>
        </div>
      </div>

      <div className="card fccp-sync-card">
        <h3>Step 1 — Fetch metadata (all ~4683)</h3>
        <p>Downloads the full judgment list (titles, judges, PDF links). No PDF files yet.</p>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => runSync(true)}
          disabled={syncing || loading}
        >
          {syncing ? (
            <>
              <Spinner size="sm" />
              Working…
            </>
          ) : (
            "Fetch All Metadata"
          )}
        </button>
      </div>

      <div className="card fccp-sync-card">
        <h3>Step 2 — Download &amp; index PDFs (batch)</h3>
        <p>Downloads PDFs in batches. Re-runs skip files already on disk.</p>
        <div className="fccp-sync-row">
          <div className="field">
            <label htmlFor="lhc-year">Year filter (optional)</label>
            <input
              id="lhc-year"
              type="text"
              placeholder="e.g. 2026 or empty = all"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              disabled={syncing}
            />
          </div>
          <div className="field">
            <label htmlFor="lhc-limit">Batch size</label>
            <input
              id="lhc-limit"
              type="number"
              min={1}
              max={500}
              value={downloadLimit}
              onChange={(e) => setDownloadLimit(Number(e.target.value))}
              disabled={syncing}
            />
          </div>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => runSync(false)}
            disabled={syncing || loading || !items.length}
          >
            Download &amp; Index Batch
          </button>
        </div>
        {status?.last_sync && (
          <p className="fccp-last-sync">Last sync: {new Date(status.last_sync).toLocaleString()}</p>
        )}
      </div>

      <div className="fccp-table-wrap card">
        <h3>Manifest preview ({preview.length}{items.length > 100 ? ` of ${items.length}` : ""})</h3>
        {loading && !items.length ? (
          <p className="loading-bar">Loading manifest…</p>
        ) : !items.length ? (
          <p className="loading-bar">No records yet — run Step 1 first.</p>
        ) : (
          <div className="fccp-table-scroll">
            <table className="fccp-table">
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Judge</th>
                  <th>Decision</th>
                  <th>Citation</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {preview.map((item) => (
                  <tr key={item.source_id || item.pdf_url}>
                    <td>{item.case_title}</td>
                    <td>{item.author_judge || "—"}</td>
                    <td>{item.decision_date}</td>
                    <td>{item.lhc_citation || "—"}</td>
                    <td>
                      {item.indexed ? (
                        <span className="pill pill-success">Indexed</span>
                      ) : item.pdf_path ? (
                        <span className="pill pill-warning">Downloaded</span>
                      ) : (
                        <span className="pill">Listed</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
