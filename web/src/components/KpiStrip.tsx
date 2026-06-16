import type { DashboardData } from "../types";
import { num, pct, moneyCompact } from "../format";

export function KpiStrip({ data }: { data: DashboardData }) {
  const h = data.headline;
  const p = data.performance;
  const items: { label: string; value: string; tone?: "good" | "bad" }[] = [
    { label: "SHARPE", value: num(h.sharpe), tone: "good" },
    { label: "ANN. RETURN", value: pct(h.annualized_return), tone: "good" },
    { label: "VOL", value: pct(h.annualized_volatility) },
    { label: "MAX DD", value: pct(h.max_drawdown), tone: "bad" },
    { label: "SORTINO", value: num(h.sortino) },
    { label: "CALMAR", value: num(h.calmar) },
    { label: "BETA", value: num(h.beta) },
    { label: "ALPHA t", value: num(h.alpha_tstat) },
    { label: "INFO RATIO", value: num(h.information_ratio), tone: h.information_ratio < 0 ? "bad" : undefined },
    {
      label: "DSR",
      value: h.deflated_sharpe_ratio != null ? pct(h.deflated_sharpe_ratio, 0) : "—",
    },
    { label: "PBO", value: h.pbo != null ? pct(h.pbo, 0) : "—" },
    { label: "TURNOVER", value: pct(p.turnover, 1) },
    { label: "NAV", value: moneyCompact(h.final_equity), tone: "good" },
  ];

  return (
    <div className="flex gap-px overflow-x-auto rounded-xl border border-line bg-line">
      {items.map((it) => (
        <div
          key={it.label}
          className="flex min-w-[92px] flex-1 flex-col gap-1 bg-surface px-3.5 py-2.5"
        >
          <span className="font-mono text-[9px] tracking-wide text-ink-faint">
            {it.label}
          </span>
          <span
            className={`tnum font-mono text-[15px] font-semibold leading-none ${
              it.tone === "good"
                ? "text-forest"
                : it.tone === "bad"
                  ? "text-redwood"
                  : "text-ink"
            }`}
          >
            {it.value}
          </span>
        </div>
      ))}
    </div>
  );
}
