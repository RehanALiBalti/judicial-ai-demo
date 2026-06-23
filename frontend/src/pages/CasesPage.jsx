import { useCallback, useEffect, useState } from "react";
import { fetchCases } from "../api/client";
import Spinner from "../components/Spinner";
import { SkeletonCard } from "../components/Skeleton";

export default function CasesPage() {
  const [cases, setCases] = useState([]);
  const [stats, setStats] = useState({ cases: 0 });
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const data = await fetchCases();
      setCases(data.cases || []);
      setStats(data.stats || { cases: 0 });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="page cases-page">
      <div className="page-header row-header animate-rise">
        <div>
          <div className="page-kicker">Records</div>
          <h2>Indexed Case Records</h2>
          <p>Cases uploaded through Upload Case. Chat PDFs do not appear here.</p>
        </div>
        <button
          type="button"
          className="btn btn-primary btn-refresh"
          onClick={() => load(true)}
          disabled={loading || refreshing}
        >
          {refreshing ? (
            <>
              <Spinner size="sm" />
              Refreshing
            </>
          ) : (
            "Refresh"
          )}
        </button>
      </div>

      {error && (
        <div className="alert alert-error animate-slide-down">
          <strong>Error</strong>
          <span>{error}</span>
        </div>
      )}

      {loading && (
        <div className="cases-grid">
          {[1, 2, 3].map((i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      )}

      {!loading && cases.length === 0 && (
        <div className="empty-state animate-fade">
          <div className="empty-icon float-icon">📂</div>
          <h3>No Indexed Cases</h3>
          <p>Upload case PDFs to build your searchable case database.</p>
        </div>
      )}

      {!loading && cases.length > 0 && (
        <>
          <div className="cases-summary animate-fade">
            Total indexed cases: <strong>{stats.cases ?? cases.length}</strong>
          </div>
          <div className="cases-grid">
            {cases.map((c, i) => (
              <article
                key={c.case_id}
                className="case-card hover-lift card-enter"
                style={{ animationDelay: `${i * 0.07}s` }}
              >
                <div className="case-card-top">
                  <div>
                    <h3 className="case-title">{c.title}</h3>
                    <p className="case-meta-line">
                      {c.case_id} · {c.file_name}
                    </p>
                  </div>
                  <span className={`pill ${c.source === "fccp" ? "pill-fccp" : ""}`}>
                    {c.source === "fccp" ? "FCCP" : "Indexed"}
                  </span>
                </div>
                <div className="case-details case-details-4">
                  <div>
                    <span>Court</span>
                    <strong>{c.court || "N/A"}</strong>
                  </div>
                  <div>
                    <span>Judge</span>
                    <strong>{c.author_judge || "—"}</strong>
                  </div>
                  <div>
                    <span>Date</span>
                    <strong>{c.decision_date || c.upload_date || "N/A"}</strong>
                  </div>
                  <div>
                    <span>Pages</span>
                    <strong>{c.pages ?? "N/A"}</strong>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
