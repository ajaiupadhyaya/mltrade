import type { DashboardData } from "../types";
import { Panel, Stat, KeyVal, Note, Pill } from "../components/primitives";
import { Histogram } from "../charts/Histogram";
import { num, pct, signedPct, significance } from "../format";

export function Integrity({ data }: { data: DashboardData }) {
  const o = data.overfitting;
  const b = data.benchmark;
  const a = data.attribution;
  const q = data.quality;

  return (
    <div className="space-y-5">
      <Panel title="Intellectual Honesty" subtitle="what this strategy is — and what it is not">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Note>
              The out-of-sample Sharpe of{" "}
              <b className="text-ink">{num(data.headline.sharpe)}</b> is real, but it
              is <b className="text-ink">not</b> evidence of alpha. Three independent
              checks point the same way:
            </Note>
            <ul className="ml-1 space-y-1.5 text-[12px] text-ink-soft">
              <li className="flex gap-2">
                <span className="text-clay">●</span>
                Alpha vs SPY is {signedPct(b.alpha_annualized)}/yr,{" "}
                <b className="text-ink">t = {num(b.alpha_tstat)}</b> ({significance(Math.abs(b.alpha_tstat))}).
              </li>
              <li className="flex gap-2">
                <span className="text-clay">●</span>
                {pct(a.r_squared, 0)} of variance is static macro-factor beta; the
                factor-residual alpha is also {significance(Math.abs(a.alpha_tstat))}.
              </li>
              <li className="flex gap-2">
                <span className="text-clay">●</span>
                Information ratio vs SPY is{" "}
                <b className="text-ink">{num(b.information_ratio)}</b> — it does not
                out-return the benchmark.
              </li>
            </ul>
          </div>
          <div className="rounded-lg border border-line-soft bg-surface-2 p-4">
            <Note>
              <b className="text-pine">The verdict.</b> MLTrade is a disciplined,
              low-turnover, drawdown-controlled <b className="text-ink">diversified
              risk-premia</b> allocation with honest, reproducible machinery — a
              research <i>framework</i>, demonstrated end-to-end. Its merit is the
              process: point-in-time data, embargoed walk-forward, fail-closed risk,
              and the overfitting diagnostics below — not a headline number.
            </Note>
          </div>
        </div>
      </Panel>

      {o && (
        <div className="grid gap-5 lg:grid-cols-5">
          <Panel className="lg:col-span-2" title="Overfitting Diagnostics" subtitle="Bailey & López de Prado">
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Deflated Sharpe" value={pct(o.deflated_sharpe_ratio, 1)} tone="pine" sub="P(true SR > threshold)" />
              <Stat label="PBO" value={pct(o.pbo, 1)} tone={o.pbo > 0.5 ? "bad" : "ink"} sub="prob. of overfitting" />
              <Stat label="PSR vs 0" value={pct(o.psr_vs_zero, 1)} sub="prob. SR > 0" />
              <Stat label="Trials" value={num(o.n_trials, 0)} sub={`${o.n_observations.toLocaleString()} obs`} />
            </div>
            <div className="mt-3 border-t border-line-soft pt-2.5">
              <KeyVal k="Observed Sharpe (ann.)" v={num(o.observed_sharpe_annualized)} />
              <KeyVal k="Deflated threshold SR" v={num(o.deflated_threshold_sharpe)} />
              <KeyVal k="CSCV splits / combos" v={`${o.pbo_n_splits} / ${o.pbo_n_combinations}`} />
            </div>
          </Panel>

          <Panel className="lg:col-span-3" title="PBO Logit Distribution" subtitle="CSCV — left of 0 ⇒ in-sample winner lagged out-of-sample">
            <Histogram
              centres={o.logit_histogram.centres}
              counts={o.logit_histogram.counts}
              height={196}
              color="var(--color-sky)"
              format={(v) => num(v, 1)}
              markers={[{ value: 0, label: "overfit ↔ robust", color: "var(--color-redwood)" }]}
            />
            <Note>
              The mass sits to the right of zero (median logit{" "}
              {num(o.logit_median)}), so the best in-sample ridge configuration
              generally held up out-of-sample — PBO ≈ {pct(o.pbo, 0)}.{" "}
              <span className="text-ink-faint">
                These diagnostics deflate for selection over the ridge-α grid only,
                not the full research process (universe, features, model).
              </span>
            </Note>
          </Panel>
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Methodology" subtitle="how the numbers are produced">
          <div className="space-y-2.5">
            <Method t="Point-in-time data" d={`Frozen ${q.adjustment} daily bars; the runtime reads only the committed snapshot, so every figure is reproducible byte-for-byte.`} />
            <Method t="Embargoed walk-forward" d="504-session minimum training history with a 21-session embargo between train and test — no look-ahead, retrained every 21 sessions." />
            <Method t="Out-of-sample only" d={`All reported performance is post-warmup, ${data.meta.oos_start}→${data.meta.oos_end}.`} />
            <Method t="Net of costs" d="5 bps headline transaction cost, with 2/5/10 bps sensitivity." />
            <Method t="Fail-closed risk" d={`${data.risk.checks.length} pre-trade gates; the pipeline blocks rather than submitting on any violation.`} />
          </div>
        </Panel>

        <Panel title="Data Provenance" subtitle="auditable lineage">
          <KeyVal k="Source" v={q.source.split(",")[0]} />
          <KeyVal k="Adjustment" v={q.adjustment} />
          <KeyVal k="Window" v={`${q.start_session} → ${q.end_session}`} />
          <KeyVal k="Sessions" v={`${q.panel_sessions.toLocaleString()} / ${q.expected_xnys_sessions.toLocaleString()} XNYS`} />
          <KeyVal k="Completeness" v={pct(q.completeness, 1)} tone="good" />
          <KeyVal k="Universe" v={`${q.n_symbols} ETFs · ${q.row_count.toLocaleString()} bars`} />
          <div className="mt-2 flex items-center gap-2 border-t border-line-soft pt-2.5">
            <Pill tone="good">SHA-256</Pill>
            <span className="tnum truncate font-mono text-[10px] text-ink-faint">{q.content_sha256}</span>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function Method({ t, d }: { t: string; d: string }) {
  return (
    <div className="flex gap-3">
      <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sage" />
      <div>
        <div className="text-[12.5px] font-semibold text-ink">{t}</div>
        <div className="text-[11.5px] leading-snug text-ink-soft">{d}</div>
      </div>
    </div>
  );
}
