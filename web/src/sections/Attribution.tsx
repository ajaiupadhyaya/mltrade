import type { DashboardData } from "../types";
import { Panel, KeyVal, Note } from "../components/primitives";
import { HBars } from "../charts/HBars";
import { num, pct, signedPct, significance } from "../format";

export function Attribution({ data }: { data: DashboardData }) {
  const a = data.attribution;
  const systematic = a.r_squared;
  const idio = Math.max(0, 1 - a.r_squared);

  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Macro Factor Exposures" subtitle="returns-based OLS · loading (β) with t-statistic">
          <HBars
            diverging
            labelWidth="w-20"
            format={(v) => num(v, 2)}
            rows={a.exposures.map((e) => ({
              label: e.factor,
              value: e.beta,
              sub: `t ${num(e.tstat, 1)}`,
            }))}
          />
          <div className="mt-3 border-t border-line-soft pt-2.5">
            <Note>
              Each factor is a liquid ETF proxy (Equity = SPY, Duration = TLT,
              Gold = GLD, Commodity = DBC, EM = EEM). All loadings are highly
              significant — the strategy is, structurally, a basket of static
              risk-premia tilts.
            </Note>
          </div>
        </Panel>

        <Panel title="Variance Decomposition" subtitle="how much of the strategy is explained by macro beta">
          <div className="mt-1 flex h-7 w-full overflow-hidden rounded-md border border-line">
            <div
              className="flex items-center justify-center font-mono text-[10px] font-semibold text-surface"
              style={{ width: `${systematic * 100}%`, background: "var(--color-forest)" }}
            >
              {pct(systematic, 0)}
            </div>
            <div
              className="flex items-center justify-center font-mono text-[10px] font-semibold text-ink-soft"
              style={{ width: `${idio * 100}%`, background: "var(--color-sage-soft)" }}
            >
              {pct(idio, 0)}
            </div>
          </div>
          <div className="mt-2 flex justify-between font-mono text-[10px] text-ink-faint">
            <span>systematic (factor β)</span>
            <span>idiosyncratic (residual)</span>
          </div>
          <div className="mt-4 border-t border-line-soft pt-3">
            <KeyVal k="Regression R²" v={pct(a.r_squared, 0)} />
            <KeyVal k="Factor-residual alpha" v={`${signedPct(a.alpha_annualized)}/yr`} tone={a.alpha_annualized < 0 ? "bad" : "ink"} />
            <KeyVal k="Alpha t-stat" v={num(a.alpha_tstat)} />
            <KeyVal k="Significance" v={significance(Math.abs(a.alpha_tstat))} tone="soft" />
          </div>
          <div className="mt-3">
            <Note>
              After stripping out macro-factor beta, the residual alpha is{" "}
              <b className="text-redwood">{significance(Math.abs(a.alpha_tstat))}</b> —
              consistent with a diversification engine rather than a
              security-selection edge.
            </Note>
          </div>
        </Panel>
      </div>

      <Panel title="Per-Symbol Contribution" subtitle="buy-and-hold return of each universe member over the out-of-sample window">
        <HBars
          diverging
          labelWidth="w-14"
          format={(v) => signedPct(v, 0)}
          rows={data.performance.per_symbol_contribution.map((s) => ({
            label: s.symbol,
            value: s.return,
          }))}
        />
      </Panel>
    </div>
  );
}
