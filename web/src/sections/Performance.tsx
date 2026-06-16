import type { DashboardData } from "../types";
import { Panel, KeyVal } from "../components/primitives";
import { LineChart } from "../charts/LineChart";
import { Histogram } from "../charts/Histogram";
import { Heatmap } from "../charts/Heatmap";
import { HBars } from "../charts/HBars";
import { num, pct, signedPct } from "../format";

export function Performance({ data }: { data: DashboardData }) {
  const p = data.performance;
  const dd = p.drawdown;
  const roll = p.rolling_sharpe;

  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Underwater — Drawdown from Peak" subtitle="peak-to-trough decline, daily">
          <LineChart
            labels={dd.map((x) => x.date)}
            height={240}
            zeroLine
            yFormat={(v) => pct(v, 0)}
            tipFormat={(v) => pct(v, 2)}
            series={[{ name: "Drawdown", color: "var(--color-redwood)", values: dd.map((x) => x.dd), area: true }]}
          />
          <div className="mt-2 grid grid-cols-3 gap-4 border-t border-line-soft pt-2.5">
            <KeyVal k="Max DD" v={pct(p.max_drawdown)} tone="bad" />
            <KeyVal k="Underwater" v={`${p.max_drawdown_duration}d`} />
            <KeyVal k="Recovery" v={p.time_to_recovery != null ? `${p.time_to_recovery}d` : "—"} />
          </div>
        </Panel>

        <Panel title="Rolling Sharpe" subtitle="126-session trailing window · annualized">
          <LineChart
            labels={roll.map((x) => x.date)}
            height={240}
            zeroLine
            yFormat={(v) => num(v, 1)}
            tipFormat={(v) => num(v, 2)}
            series={[{ name: "Rolling Sharpe", color: "var(--color-forest)", values: roll.map((x) => x.value), area: true }]}
          />
          <div className="mt-2 grid grid-cols-3 gap-4 border-t border-line-soft pt-2.5">
            <KeyVal k="Full-sample" v={num(p.sharpe)} />
            <KeyVal k="Sortino" v={num(p.sortino)} />
            <KeyVal k="Calmar" v={num(p.calmar)} />
          </div>
        </Panel>
      </div>

      <div className="grid gap-5 lg:grid-cols-5">
        <Panel className="lg:col-span-3" title="Monthly Returns" subtitle="calendar months · % · warm = gain, rust = loss">
          <Heatmap monthly={p.monthly} />
        </Panel>

        <Panel className="lg:col-span-2" title="Daily Return Distribution" subtitle={`${p.n_sessions.toLocaleString()} sessions · skew ${num(p.skewness)} · kurt ${num(p.excess_kurtosis)}`}>
          <Histogram
            centres={p.histogram.centres}
            counts={p.histogram.counts}
            height={210}
            format={(v) => pct(v, 1)}
            markers={[
              { value: 0, label: "0", color: "var(--color-ink-faint)" },
              { value: -p.var_95, label: "VaR₉₅", color: "var(--color-redwood)" },
            ]}
          />
          <div className="mt-1 grid grid-cols-2 gap-x-6">
            <KeyVal k="Best day" v={signedPct(p.best_day, 2)} tone="good" />
            <KeyVal k="Worst day" v={signedPct(p.worst_day, 2)} tone="bad" />
            <KeyVal k="Positive days" v={pct(p.positive_fraction, 1)} />
            <KeyVal k="VaR 95%" v={pct(p.var_95, 2)} />
          </div>
        </Panel>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Cost Sensitivity" subtitle="headline Sharpe under varying transaction cost">
          <HBars
            labelWidth="w-16"
            format={(v) => num(v, 2)}
            rows={p.cost_sensitivity.map((c) => ({
              label: `${c.bps} bps`,
              value: c.sharpe,
              sub: `${signedPct(c.annualized_return)} ann`,
            }))}
          />
        </Panel>

        <Panel title="Non-Overlapping Evaluation Windows" subtitle="annual out-of-sample slices">
          <div className="max-h-[230px] overflow-y-auto">
            <table className="w-full text-[11.5px]">
              <thead className="sticky top-0 bg-surface">
                <tr className="border-b border-line-soft font-mono text-[9.5px] text-ink-faint">
                  <th className="py-1.5 text-left font-normal">WINDOW</th>
                  <th className="py-1.5 text-right font-normal">SHARPE</th>
                  <th className="py-1.5 text-right font-normal">RETURN</th>
                  <th className="py-1.5 text-right font-normal">MAX DD</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line-soft">
                {p.evaluation_windows.map((w) => (
                  <tr key={w.start} className="font-mono">
                    <td className="py-1.5 text-ink-soft">{w.start.slice(0, 7)}</td>
                    <td className={`tnum py-1.5 text-right font-medium ${w.sharpe < 0 ? "text-redwood" : "text-forest"}`}>
                      {num(w.sharpe)}
                    </td>
                    <td className="tnum py-1.5 text-right text-ink">{signedPct(w.annualized_return)}</td>
                    <td className="tnum py-1.5 text-right text-clay">{pct(w.max_drawdown)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}
