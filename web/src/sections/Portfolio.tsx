import type { DashboardData } from "../types";
import { Panel, KeyVal, Pill, Note } from "../components/primitives";
import { Donut } from "../charts/Donut";
import { HBars } from "../charts/HBars";
import { classColor } from "../palette";
import { tickerColor } from "../palette";
import { num, pct, signedPct, money, moneyCompact } from "../format";

export function PortfolioSection({ data }: { data: DashboardData }) {
  const pf = data.portfolio;
  const ex = data.execution;
  const fc = data.forecast;

  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-5">
        <Panel className="lg:col-span-2" title="Target Allocation" subtitle="current decision · by asset class">
          <div className="flex items-center gap-5">
            <Donut
              size={168}
              centerLabel={pct(1 - pf.cash_weight, 0)}
              centerSub="invested"
              slices={pf.asset_classes.map((c) => ({
                label: c.asset_class,
                value: c.weight,
                color: classColor(c.asset_class),
              }))}
            />
            <div className="flex-1 space-y-1.5">
              {pf.asset_classes.map((c) => (
                <div key={c.asset_class} className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-sm" style={{ background: classColor(c.asset_class) }} />
                  <span className="flex-1 text-[11.5px] text-ink-soft">{c.asset_class}</span>
                  <span className="tnum font-mono text-[11.5px] font-medium text-ink">{pct(c.weight, 1)}</span>
                </div>
              ))}
              <div className="flex items-center gap-2 border-t border-line-soft pt-1.5">
                <span className="h-2.5 w-2.5 rounded-sm bg-sunk" />
                <span className="flex-1 text-[11.5px] text-ink-soft">Cash</span>
                <span className="tnum font-mono text-[11.5px] font-medium text-ink">{pct(pf.cash_weight, 1)}</span>
              </div>
            </div>
          </div>
        </Panel>

        <Panel className="lg:col-span-3" title="Position Weights" subtitle="volatility-scaled, constrained optimizer output">
          <HBars
            labelWidth="w-14"
            format={(v) => pct(v, 1)}
            rows={pf.weights.map((w) => ({
              label: w.symbol,
              value: w.weight,
              sub: w.asset_class,
              color: tickerColor(w.symbol),
            }))}
          />
        </Panel>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel
          title="Execution Preview"
          subtitle={`${ex.count} order intents · ${ex.broker}`}
          right={<Pill tone="warn">PREVIEW ONLY</Pill>}
        >
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-line-soft font-mono text-[9.5px] text-ink-faint">
                <th className="py-1.5 text-left font-normal">SIDE</th>
                <th className="py-1.5 text-left font-normal">SYMBOL</th>
                <th className="py-1.5 text-right font-normal">QTY</th>
                <th className="py-1.5 text-right font-normal">NOTIONAL</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line-soft">
              {ex.intents.map((i) => (
                <tr key={i.client_order_id} className="font-mono">
                  <td className="py-1.5">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${i.side.toLowerCase().includes("buy") ? "bg-sage-soft text-pine" : "bg-clay-soft text-redwood"}`}>
                      {i.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-1.5 font-semibold text-ink">{i.symbol}</td>
                  <td className="tnum py-1.5 text-right text-ink-soft">{num(i.quantity, 0)}</td>
                  <td className="tnum py-1.5 text-right text-ink">{money(i.notional)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 border-t border-line-soft pt-2.5">
            <KeyVal k="Reconciliation" v={ex.reconciliation_blocked ? "BLOCKED" : "clean"} tone={ex.reconciliation_blocked ? "bad" : "good"} />
          </div>
        </Panel>

        <Panel title="Forecast Cross-Section" subtitle={`${fc.model_version} · 21-session predicted forward return`}>
          <HBars
            diverging
            labelWidth="w-14"
            format={(v) => signedPct(v, 2)}
            rows={fc.forecasts.map((f) => ({ label: f.symbol, value: f.predicted_forward_return }))}
          />
          <div className="mt-3 border-t border-line-soft pt-2.5">
            <Note>
              Trained on {fc.training_session_count.toLocaleString()} sessions
              ({fc.training_row_count.toLocaleString()} rows). Reference NAV{" "}
              {moneyCompact(data.meta.reference_equity)}.
            </Note>
          </div>
        </Panel>
      </div>
    </div>
  );
}
