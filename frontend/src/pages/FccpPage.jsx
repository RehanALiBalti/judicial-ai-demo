import { useCallback, useEffect, useState } from "react";
import { fetchFccpStatus, startFccpSync } from "../api/client";
import Spinner from "../components/Spinner";
import LoadingOverlay from "../components/LoadingOverlay";

export default function FccpPage({ onSynced }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [startPage, setStartPage] = useState(1);
  const [endPage, setEndPage] = useState(5);
  const [alert, setAlert] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchFccpStatus();
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
    const id = setInterval(load, syncing ? 3000 : 15000);
    return () => clearInterval(id);
  }, [load, syncing]);

  const handleSync = async () => {
    setAlert(null);
    try {
      const result = await startFccpSync({ startPage, endPage: endPage || undefined });
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

  return (
    <div className="page fccp-page">
      {syncing && (
        <LoadingOverlay
          message="Syncing FCCP judgments…"
          submessage="Downloading PDFs from fccp.gov.pk and building chat dataset"
        />
      )}

      <div className="page-header animate-rise">
        <div className="page-kicker">Dataset Builder</div>
        <h2>FCCP Judgments Import</h2>
        <p>
          Scrape judgments from{" "}
          <a href="https://fccp.gov.pk/judgments?page=1" target="_blank" rel="noreferrer">
            Federal Constitutional Court of Pakistan
          </a>
          , store PDFs locally, and index for AI chat.
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
        <div className="fccp-stat-card hover-lift">
          <span className="fccp-stat-value">{status?.stats?.chunks ?? "—"}</span>
          <span className="fccp-stat-label">Chat chunks</span>
        </div>
      </div>

      <div className="card fccp-sync-card">
        <h3>Run Scraper</h3>
        <p>Downloads once per case. Re-runs use local PDFs — no duplicate downloads.</p>
        <div className="fccp-sync-row">
          <div className="field">
            <label htmlFor="start-page">Start page</label>
            <input
              id="start-page"
              type="number"
              min={1}
              value={startPage}
              onChange={(e) => setStartPage(Number(e.target.value))}
              disabled={syncing}
            />
          </div>
          <div className="field">
            <label htmlFor="end-page">End page</label>
            <input
              id="end-page"
              type="number"
              min={1}
              value={endPage}
              onChange={(e) => setEndPage(Number(e.target.value))}
              disabled={syncing}
            />
          </div>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleSync}
            disabled={syncing || loading}
          >
            {syncing ? (
              <>
                <Spinner size="sm" />
                Syncing…
              </>
            ) : (
              "Sync & Index All"
            )}
          </button>
        </div>
        {status?.last_sync && (
          <p className="fccp-last-sync">Last sync: {new Date(status.last_sync).toLocaleString()}</p>
        )}
      </div>

      <div className="fccp-table-wrap card">
        <h3>Scraped Records ({items.length})</h3>
        {loading && !items.length ? (
          <p className="loading-bar">Loading manifest…</p>
        ) : (
          <div className="fccp-table-scroll">
            <table className="fccp-table">
              <thead>
                <tr>
                  <th>Case Title</th>
                  <th>Judge</th>
                  <th>Upload Date</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.download_url}>
                    <td>{item.case_title}</td>
                    <td>{item.author_judge || "—"}</td>
                    <td>{item.upload_date}</td>
                    <td>
                      {item.indexed ? (
                        <span className="pill pill-success">Indexed</span>
                      ) : item.pdf_path ? (
                        <span className="pill pill-warning">Downloaded</span>
                      ) : (
                        <span className="pill">Pending</span>
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
