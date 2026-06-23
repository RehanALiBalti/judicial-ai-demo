import { SkeletonStats } from "./Skeleton";

export default function Layout({
  tabs,
  activeTab,
  onTabChange,
  stats,
  statsLoading,
  backendOnline,
  children,
}) {
  return (
    <div className="app-shell">
      <header className="app-header animate-rise">
        <div className="brand">
          <div className="brand-mark pulse-glow">J</div>
          <div>
            <h1 className="brand-title">JAMS</h1>
            <p className="brand-subtitle">Judicial AI Management System</p>
          </div>
        </div>

        <div className="header-right">
          <div className={`status-pill ${backendOnline ? "online" : "offline"}`}>
            <span className="status-dot" />
            {backendOnline ? "Backend connected" : "Backend offline"}
          </div>

          {statsLoading ? (
            <SkeletonStats />
          ) : (
            <div className="header-stats">
              {[
                { value: stats.cases, label: "Cases" },
                { value: stats.chunks, label: "Chunks" },
                { value: stats.pages, label: "Pages" },
              ].map((s) => (
                <div key={s.label} className="stat-chip hover-lift">
                  <span className="stat-value">{s.value}</span>
                  <span className="stat-label">{s.label}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </header>

      <nav className="tab-bar animate-rise" style={{ animationDelay: "0.05s" }}>
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`tab-btn ${activeTab === t.id ? "active" : ""}`}
            onClick={() => onTabChange(t.id)}
          >
            <span className="tab-icon">{t.icon}</span>
            {t.label}
          </button>
        ))}
      </nav>

      <main className="app-main">{children}</main>

      <footer className="app-footer animate-fade">
        <span className="footer-badge">JAMS</span>
        <span>Demo · Session data stored in memory only</span>
      </footer>
    </div>
  );
}
