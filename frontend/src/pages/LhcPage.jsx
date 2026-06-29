import { useCallback, useEffect, useState } from "react";
import { fetchLhcStatus, startLhcSync } from "../api/client";
import Spinner from "../components/Spinner";
import LoadingOverlay from "../components/LoadingOverlay";

export default function LhcPage({ onSynced }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [year, setYear] = useState("");
  const [batchSize, setBatchSize] = useState(50);
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
    const id = setInterval(load, syncing ? 3000 : 15000);
    return () => clearInterval(id);
  }, [load, syncing]);

  const handleFetchMetadata = async () => {
    setAlert(null);
    try {
      const result = await startLhcSync({ year: year || undefined, metadataOnly: true });
      setAlert({ type: "info", title: "Sync Started", message: result.message });
      setSyncing(true);
    } catch (err) {
      setAlert({ type: "error", title: "Sync Failed", message: err.message });
    }
  };

  const handleSyncBatch = async () => {
    setAlert(null);
    try {
      const result = await startLhcSync({
        year: year || undefined,
        metadataOnly: false,
        downloadLimit: batchSize,
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
          message: `Listed ${r.scraped} judgments (site total: ${r.total_reported ?? "?"}).`,
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
  const hasMetadata = items.length > 0;

  return (
    <div className="page fccp-page">
      {syncing && (
        <LoadingOverlay
          message="Syncing LHC judgments…"
          submessage="Fetching from data.lhc.gov.pk and building chat dataset"
        />
      )}

      <div className="page-header animate-rise">
        <div className="page-kicker">Dataset Builder</div>
        <h2>LHC Judgments Import</h2>
        <p>
          Scrape judgments from{" "}
          <a
            href="https://data.lhc.gov.pk/reported_judgments/judgments_approved_for_reporting"
            target="_blank"
            rel="noreferrer"
          >
            Lahore High Court
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
        <p>
          First fetch the full judgment list (~4683 with All Courts), then download PDFs in
          batches. Re-runs use local PDFs — no duplicate downloads.
        </p>
        <div className="fccp-sync-row">
          <div className="field">
            <label htmlFor="lhc-year">Year (optional)</label>
            <input
              id="lhc-year"
              type="text"
              placeholder="All years"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              disabled={syncing}
            />
          </div>
          <div className="field">
            <label htmlFor="lhc-batch">Batch size</label>
            <input
              id="lhc-batch"
              type="number"
              min={1}
              max={500}
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              disabled={syncing}
            />
          </div>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={handleFetchMetadata}
            disabled={syncing || loading}
          >
            {syncing ? (
              <>
                <Spinner size="sm" />
                Working…
              </>
            ) : (
              "Fetch Metadata"
            )}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleSyncBatch}
            disabled={syncing || loading || !hasMetadata}
            title={hasMetadata ? "" : "Fetch metadata first"}
          >
            {syncing ? (
              <>
                <Spinner size="sm" />
                Syncing…
              </>
            ) : (
              "Sync & Index Batch"
            )}
          </button>
        </div>
        {status?.total_reported != null && (
          <p className="fccp-last-sync">
            LHC site total: {status.total_reported} judgments
            {status?.last_sync ? ` · Last sync: ${new Date(status.last_sync).toLocaleString()}` : ""}
          </p>
        )}
      </div>

      <div className="fccp-table-wrap card">
        <h3>Scraped Records ({items.length})</h3>
        {loading && !items.length ? (
          <p className="loading-bar">Loading manifest…</p>
        ) : !items.length ? (
          <p className="loading-bar">No records yet — click Fetch Metadata to load the judgment list.</p>
        ) : (
          <div className="fccp-table-scroll">
            <table className="fccp-table">
              <thead>
                <tr>
                  <th>Case Title</th>
                  <th>Judge</th>
                  <th>Decision Date</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.source_id || item.pdf_url}>
                    <td>{item.case_title}</td>
                    <td>{item.author_judge || "—"}</td>
                    <td>{item.decision_date || "—"}</td>
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
