import type { DashboardData, RiskCheck } from "../types";
import { Panel, KeyVal, Pill, Note } from "../components/primitives";
import { Histogram } from "../charts/Histogram";
import { num, pct } from "../format";

export function RiskSection({ data }: { data: DashboardData }) {
  const p = data.performance;
  const risk = data.risk;

  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-3">
        <Panel title="Tail Risk" subtitle="daily loss thresholds · historical & modified">
          <KeyVal k="VaR 95%" v={pct(p.var_95, 2)} tone="bad" />
          <KeyVal k="CVaR 95% (ES)" v={pct(p.cvar_95, 2)} tone="bad" />
          <KeyVal k="VaR 99%" v={pct(p.var_99, 2)} tone="bad" />
          <KeyVal k="CVaR 99% (ES)" v={pct(p.cvar_99, 2)} tone="bad" />
          <KeyVal k="Cornish-Fisher VaR 95%" v={pct(p.cornish_fisher_var_95, 2)} />
          <div className="mt-2 border-t border-line-soft pt-2">
            <Note>
              The Cornish-Fisher VaR adjusts the Gaussian quantile for the
              distribution's negative skew and fat tails (excess kurtosis{" "}
              {num(p.excess_kurtosis)}).
            </Note>
          </div>
        </Panel>

        <Panel title="Loss Distribution" subtitle="left tail of daily returns">
          <Histogram
            centres={p.histogram.centres}
            counts={p.histogram.counts}
            height={186}
            color="var(--color-clay)"
            format={(v) => pct(v, 1)}
            markers={[
              { value: -p.var_95, label: "95%", color: "var(--color-redwood)" },
              { value: -p.var_99, label: "99%", color: "var(--color-pine)" },
            ]}
          />
        </Panel>

        <Panel title="Drawdown & Moments" subtitle="path risk and shape">
          <KeyVal k="Max drawdown" v={pct(p.max_drawdown)} tone="bad" />
          <KeyVal k="Underwater (max)" v={`${p.max_drawdown_duration} sessions`} />
          <KeyVal k="Recovery from trough" v={p.time_to_recovery != null ? `${p.time_to_recovery} sessions` : "unrecovered"} />
          <KeyVal k="Volatility (ann.)" v={pct(p.annualized_volatility)} />
          <KeyVal k="Skewness" v={num(p.skewness)} tone={p.skewness < 0 ? "bad" : "ink"} />
          <KeyVal k="Excess kurtosis" v={num(p.excess_kurtosis)} />
        </Panel>
      </div>

      <Panel
        title="Pre-Trade Risk Gates"
        subtitle={`${risk.checks.length} fail-closed checks evaluated on the current decision`}
        right={
          <div className="flex items-center gap-1.5">
            <Pill tone="good">{risk.summary.pass} PASS</Pill>
            {risk.summary.warn > 0 && <Pill tone="warn">{risk.summary.warn} WARN</Pill>}
            <Pill tone={risk.summary.block > 0 ? "bad" : "neutral"}>{risk.summary.block} BLOCK</Pill>
          </div>
        }
      >
        <div className="grid gap-x-6 gap-y-0 sm:grid-cols-2">
          {risk.checks.map((c) => (
            <GateRow key={c.code} check={c} />
          ))}
        </div>
        {risk.blocked && (
          <div className="mt-4 border-t border-line-soft pt-3">
            <Note>
              <b className="text-redwood">Blocked by design.</b> The cold-start
              allocation trips two steady-state notional caps — the fail-closed
              policy correctly refuses to submit until positions are established.
            </Note>
          </div>
        )}
      </Panel>
    </div>
  );
}

function GateRow({ check }: { check: RiskCheck }) {
  const color =
    check.status === "pass"
      ? "bg-forest"
      : check.status === "warn"
        ? "bg-ochre"
        : "bg-clay";
  const text =
    check.status === "block" ? "text-redwood" : check.status === "warn" ? "text-ochre" : "text-ink-soft";
  return (
    <div className="flex items-start gap-2.5 border-b border-line-soft py-2">
      <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${color}`} />
      <div className="min-w-0">
        <div className="font-mono text-[11px] font-medium text-ink">{check.code}</div>
        <div className={`text-[11px] leading-snug ${text}`}>{check.message}</div>
      </div>
    </div>
  );
}
