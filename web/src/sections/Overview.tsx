import type { DashboardData } from "../types";
import { Panel, Stat, KeyVal, Legend, Note } from "../components/primitives";
import { LineChart } from "../charts/LineChart";
import { YearBars } from "../charts/YearBars";
import { num, pct, signedPct, moneyCompact, significance } from "../format";

const STRAT = "var(--color-strategy)";
const BENCH = "var(--color-benchmark)";

export function Overview({ data }: { data: DashboardData }) {
  const h = data.headline;
  const p = data.performance;
  const b = data.benchmark;
  const ec = p.equity_curve;
  const labels = ec.map((x) => x.date);
  const benchAnn =
    (ec[ec.length - 1].benchmark / ec[0].benchmark) ** (252 / p.n_sessions) - 1;

  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-3">
        <Panel
          className="lg:col-span-2"
          title="Growth of $1.00M — Strategy vs Benchmark"
          subtitle={`Out-of-sample ${data.meta.oos_start} → ${data.meta.oos_end} · net of 5 bps · log scale`}
          right={
            <Legend
              items={[
                { label: "Strategy", color: STRAT },
                { label: "SPY", color: BENCH },
              ]}
            />
          }
        >
          <LineChart
            labels={labels}
            height={320}
            logY
            yFormat={moneyCompact}
            series={[
              { name: "Strategy", color: STRAT, values: ec.map((x) => x.strategy), area: true },
              { name: "SPY", color: BENCH, values: ec.map((x) => x.benchmark) },
            ]}
          />
          <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-1.5 border-t border-line-soft pt-3">
            <Baseline label="Strategy" value={signedPct(h.annualized_return)} accent />
            <Baseline label="SPY" value={signedPct(benchAnn)} />
            <Baseline label="Equal-weight" value={signedPct(p.equal_weight_return)} />
            <Baseline label="Cash" value={signedPct(p.cash_return)} />
            <span className="ml-auto font-mono text-[10px] text-ink-faint">
              annualized · {p.n_sessions.toLocaleString()} sessions
            </span>
          </div>
        </Panel>

        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Ann. Return" value={pct(h.annualized_return)} tone="pine" sub="net of cost" />
            <Stat label="Sharpe" value={num(h.sharpe)} tone="pine" sub="rf = 0%" />
            <Stat label="Max Drawdown" value={pct(h.max_drawdown)} tone="clay" sub="peak-to-trough" />
            <Stat label="Final NAV" value={moneyCompact(h.final_equity)} sub={`${num(h.total_return_multiple)}× start`} />
          </div>

          <Panel title="The honest read" subtitle="what the statistics actually say">
            <Note>
              Out-of-sample Sharpe is{" "}
              <b className="text-ink">{num(h.sharpe)}</b>, but alpha vs SPY is{" "}
              <b className="text-ink">{signedPct(b.alpha_annualized)}</b>/yr with a
              t-stat of <b className="text-ink">{num(b.alpha_tstat)}</b> —{" "}
              <span className="text-redwood">{significance(Math.abs(b.alpha_tstat))}</span>. Roughly{" "}
              <b className="text-ink">{pct(data.attribution.r_squared, 0)}</b> of its
              variance is explained by static macro-factor exposure.
            </Note>
            <Note>
              <span className="mt-2 block">
                This is a disciplined, low-turnover{" "}
                <b className="text-ink">diversified risk-premia</b> allocation — its
                value is drawdown control and diversification, not statistical alpha.
              </span>
            </Note>
          </Panel>
        </div>
      </div>

      <Panel
        title="Calendar-Year Returns"
        subtitle="strategy vs SPY · compounded per year"
        right={
          <Legend
            items={[
              { label: "Strategy", color: STRAT },
              { label: "SPY", color: BENCH },
            ]}
          />
        }
      >
        <YearBars data={p.yearly} format={(v) => pct(v, 0)} />
      </Panel>

      <Panel title="Key Statistics" subtitle="risk-adjusted performance · out-of-sample">
        <div className="grid grid-cols-2 gap-x-8 gap-y-0 sm:grid-cols-4">
          <KeyVal k="Sortino" v={num(p.sortino)} />
          <KeyVal k="Calmar" v={num(p.calmar)} />
          <KeyVal k="Volatility" v={pct(p.annualized_volatility)} />
          <KeyVal k="Beta (SPY)" v={num(b.beta)} />
          <KeyVal k="Information ratio" v={num(b.information_ratio)} tone={b.information_ratio < 0 ? "bad" : "ink"} />
          <KeyVal k="Correlation" v={num(b.correlation)} />
          <KeyVal k="Hit rate" v={pct(p.hit_rate, 1)} />
          <KeyVal k="Avg turnover" v={pct(p.turnover, 1)} />
          <KeyVal k="Down capture" v={pct(b.down_capture, 0)} />
          <KeyVal k="Up capture" v={pct(b.up_capture, 0)} />
          <KeyVal k="Skew" v={num(p.skewness)} />
          <KeyVal k="Excess kurtosis" v={num(p.excess_kurtosis)} />
        </div>
      </Panel>
    </div>
  );
}

function Baseline({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="font-mono text-[10px] uppercase tracking-wide text-ink-faint">{label}</span>
      <span className={`tnum font-mono text-[12.5px] font-semibold ${accent ? "text-pine" : "text-ink"}`}>
        {value}
      </span>
    </span>
  );
}
