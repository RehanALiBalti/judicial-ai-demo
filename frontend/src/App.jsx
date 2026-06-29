import { useCallback, useEffect, useState } from "react";
import Layout from "./components/Layout";
import ChatPage from "./pages/ChatPage";
import UploadPage from "./pages/UploadPage";
import CasesPage from "./pages/CasesPage";
import FccpPage from "./pages/FccpPage";
import LhcPage from "./pages/LhcPage";
import { fetchStats } from "./api/client";

const TABS = [
  { id: "chat", label: "AI Chat", icon: "💬" },
  { id: "fccp", label: "FCCP Import", icon: "⚖️" },
  { id: "lhc", label: "LHC Import", icon: "🏛️" },
  { id: "upload", label: "Upload Case", icon: "📤" },
  { id: "cases", label: "Indexed Cases", icon: "📁" },
];

export default function App() {
  const [tab, setTab] = useState("chat");
  const [stats, setStats] = useState({ cases: 0, chunks: 0, pages: 0 });
  const [statsLoading, setStatsLoading] = useState(true);
  const [backendOnline, setBackendOnline] = useState(true);

  const refreshStats = useCallback(async () => {
    try {
      const data = await fetchStats();
      setStats(data);
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    } finally {
      setStatsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshStats();
    const id = setInterval(refreshStats, 15000);
    return () => clearInterval(id);
  }, [refreshStats]);

  return (
    <Layout
      tabs={TABS}
      activeTab={tab}
      onTabChange={setTab}
      stats={stats}
      statsLoading={statsLoading}
      backendOnline={backendOnline}
    >
      <div className="tab-panels">
        <div className={`tab-panel ${tab === "chat" ? "is-active" : ""}`} hidden={tab !== "chat"}>
          <ChatPage onStatsChange={refreshStats} />
        </div>
        <div className={`tab-panel ${tab === "fccp" ? "is-active" : ""}`} hidden={tab !== "fccp"}>
          <FccpPage onSynced={refreshStats} />
        </div>
        <div className={`tab-panel ${tab === "lhc" ? "is-active" : ""}`} hidden={tab !== "lhc"}>
          <LhcPage onSynced={refreshStats} />
        </div>
        <div className={`tab-panel ${tab === "upload" ? "is-active" : ""}`} hidden={tab !== "upload"}>
          <UploadPage onUploaded={refreshStats} />
        </div>
        <div className={`tab-panel ${tab === "cases" ? "is-active" : ""}`} hidden={tab !== "cases"}>
          <CasesPage />
        </div>
      </div>
    </Layout>
  );
}
