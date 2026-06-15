import type { ReactNode } from "react";
import { useDashboard } from "./useDashboard";
import { Header } from "./components/Header";
import { EquityChart } from "./components/EquityChart";
import { AllocationPanel } from "./components/AllocationPanel";
import { RiskGates } from "./components/RiskGates";
import { ExecutionPanel } from "./components/ExecutionPanel";
import { ExperimentLeaderboard } from "./components/ExperimentLeaderboard";
import { Footer } from "./components/Footer";
import { Panel, StatCard } from "./components/primitives";
import { num, pct, thousands } from "./format";
import type { Backtest, CostSensitivityPoint } from "./types";

function CostSensitivity({ points }: { points: CostSensitivityPoint[] }) {
  const sharpes = points.map((p) => p.sharpe);
  const lo = Math.min(...sharpes) - 0.06;
  const hi = Math.max(...sharpes);
  const width = (s: number) => 22 + ((s - lo) / (hi - lo || 1)) * 78;
  return (
    <div className="rounded-2xl border border-line bg-surface p-5">
      <div className="flex items-center justify-between">
        <h3 className="font-display text-[0.98rem] font-semibold text-ink">
          Cost sensitivity
        </h3>
        <span className="font-mono text-[10px] tracking-wide text-ink-faint">
          SHARPE BY COST
        </span>
      </div>
      <div className="mt-3.5 space-y-3">
        {points.map((p) => (
          <div key={p.bps} className="flex items-center gap-3">
            <span className="w-12 font-mono text-[11px] text-ink-soft">
              {p.bps} bps
            </span>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-sunk">
              <div
                className="h-full rounded-full bg-forest"
                style={{ width: `${width(p.sharpe)}%` }}
              />
            </div>
            <span className="tnum w-10 text-right font-mono text-[11.5px] font-medium text-ink">
              {num(p.sharpe)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Baselines({ bt }: { bt: Backtest }) {
  const item = (label: string, value: string, accent?: boolean) => (
    <span className="flex items-baseline gap-1.5">
      <span className="font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
        {label}
      </span>
      <span
        className={`tnum font-mono text-[12.5px] font-medium ${
          accent ? "text-pine" : "text-ink"
        }`}
      >
        {value}
      </span>
    </span>
  );
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-line-soft pt-4">
      {item("Strategy", pct(bt.annualized_return), true)}
      {item("Equal-weight", pct(bt.equal_weight_return))}
      {item("Cash", pct(bt.cash_return))}
      <span className="ml-auto font-mono text-[10.5px] text-ink-faint">
        annualized return · net of {bt.headline_cost_bps} bps
      </span>
    </div>
  );
}

function Legend() {
  return (
    <div className="hidden items-center gap-4 sm:flex">
      <span className="flex items-center gap-2">
        <span className="h-1 w-4 rounded bg-forest" />
        <span className="text-[11.5px] text-ink-soft">Strategy</span>
      </span>
      <span className="flex items-center gap-2">
        <span className="h-0 w-4 border-t border-dashed border-sage" />
        <span className="text-[11.5px] text-ink-soft">Start $1.00M</span>
      </span>
    </div>
  );
}

function Centered({
  children,
  tone,
}: {
  children: ReactNode;
  tone?: "error";
}) {
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

  if (state.status === "loading") {
    return <Centered>Loading dashboard…</Centered>;
  }
  if (state.status === "error") {
    return (
      <Centered tone="error">
        Couldn't load dashboard data — {state.message}. Run{" "}
        <code className="mx-1 rounded bg-sunk px-1.5 py-0.5 font-mono text-[12px] text-pine">
          mltrade export
        </code>{" "}
        first.
      </Centered>
    );
  }

  const d = state.data;
  const bt = d.backtest;
  const initial = bt.equity_curve[0]?.equity ?? 1_000_000;

  return (
    <div className="mx-auto max-w-[1480px] px-6 py-8 md:px-10">
      <Header meta={d.meta} />

      <main className="mt-7 space-y-6">
        {/* Hero: backtest performance + KPIs */}
        <div className="grid gap-6 lg:grid-cols-3">
          <Panel
            className="lg:col-span-2"
            title="Backtest Performance"
            subtitle={`Walk-forward · ${thousands(bt.sessions)} sessions · ${d.meta.model_version} · headline ${bt.headline_cost_bps} bps`}
            right={<Legend />}
          >
            <EquityChart points={bt.equity_curve} initialEquity={initial} />
            <Baselines bt={bt} />
          </Panel>

          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4">
              <StatCard
                label="ANN. RETURN"
                value={pct(bt.annualized_return)}
                sub={`net of ${bt.headline_cost_bps} bps cost`}
              />
              <StatCard
                label="SHARPE"
                value={num(bt.sharpe)}
                sub="risk-free = 0%"
                tone="pine"
              />
              <StatCard
                label="VOLATILITY"
                value={pct(bt.annualized_volatility)}
                sub="target 15%"
              />
              <StatCard
                label="MAX DRAWDOWN"
                value={pct(bt.max_drawdown)}
                sub="peak-to-trough"
                tone="clay"
              />
            </div>
            <CostSensitivity points={bt.cost_sensitivity} />
          </div>
        </div>

        {/* Allocation + risk gates */}
        <div className="grid gap-6 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <AllocationPanel portfolio={d.portfolio} />
          </div>
          <div className="lg:col-span-7">
            <RiskGates risk={d.risk} />
          </div>
        </div>

        {/* Execution preview + experiment leaderboard */}
        <div className="grid gap-6 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <ExecutionPanel execution={d.execution} />
          </div>
          <div className="lg:col-span-7">
            <ExperimentLeaderboard experiments={d.experiments} />
          </div>
        </div>

        <Footer data={d} />
      </main>
    </div>
  );
}
