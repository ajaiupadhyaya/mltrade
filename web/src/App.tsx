import { useState } from "react";
import type { ReactNode } from "react";
import { useDashboard } from "./useDashboard";
import { Header } from "./components/Header";
import { KpiStrip } from "./components/KpiStrip";
import { Footer } from "./components/Footer";
import { Overview } from "./sections/Overview";
import { Performance } from "./sections/Performance";
import { RiskSection } from "./sections/Risk";
import { Attribution } from "./sections/Attribution";
import { Integrity } from "./sections/Integrity";
import { PortfolioSection } from "./sections/Portfolio";
import { Experiments } from "./sections/Experiments";
import type { DashboardData } from "./types";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "performance", label: "Performance" },
  { id: "risk", label: "Risk" },
  { id: "attribution", label: "Attribution" },
  { id: "integrity", label: "Integrity" },
  { id: "portfolio", label: "Portfolio" },
  { id: "experiments", label: "Experiments" },
] as const;

type TabId = (typeof TABS)[number]["id"];

function render(tab: TabId, data: DashboardData): ReactNode {
  switch (tab) {
    case "overview":
      return <Overview data={data} />;
    case "performance":
      return <Performance data={data} />;
    case "risk":
      return <RiskSection data={data} />;
    case "attribution":
      return <Attribution data={data} />;
    case "integrity":
      return <Integrity data={data} />;
    case "portfolio":
      return <PortfolioSection data={data} />;
    case "experiments":
      return <Experiments data={data} />;
  }
}

function Centered({ children, tone }: { children: ReactNode; tone?: "error" }) {
  return (
    <div className="flex min-h-screen items-center justify-center px-6">
      <div
        className={`max-w-md text-center text-[14px] leading-relaxed ${
          tone === "error" ? "text-redwood" : "text-ink-soft"
        }`}
      >
        {children}
      </div>
    </div>
  );
}

export function App() {
  const state = useDashboard();
  const [tab, setTab] = useState<TabId>("overview");

  if (state.status === "loading") return <Centered>Loading research terminal…</Centered>;
  if (state.status === "error") {
    return (
      <Centered tone="error">
        Couldn't load dashboard data — {state.message}. Run{" "}
        <code className="mx-1 rounded bg-sunk px-1.5 py-0.5 font-mono text-[12px] text-pine">
          uv run mltrade export
        </code>{" "}
        first.
      </Centered>
    );
  }

  const data = state.data;

  return (
    <div className="mx-auto max-w-[1480px] px-5 py-7 md:px-9">
      <Header meta={data.meta} />

      <div className="sticky top-0 z-20 -mx-5 mt-4 border-b border-line bg-canvas/85 px-5 py-2.5 backdrop-blur md:-mx-9 md:px-9">
        <nav className="flex items-center gap-1 overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`whitespace-nowrap rounded-md px-3.5 py-1.5 font-mono text-[11.5px] font-semibold tracking-wide transition-colors ${
                tab === t.id
                  ? "bg-pine text-surface"
                  : "text-ink-soft hover:bg-surface-2 hover:text-ink"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="mt-5">
        <KpiStrip data={data} />
      </div>

      <main className="mt-6">{render(tab, data)}</main>

      <Footer data={data} />
    </div>
  );
}
