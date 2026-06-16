import type { Meta } from "../types";
import { dateShort } from "../format";

export function Header({ meta }: { meta: Meta }) {
  return (
    <header className="flex flex-col gap-4 border-b border-line pb-5 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <div className="flex items-center gap-2.5">
          <Mark />
          <h1 className="font-display text-[1.7rem] font-semibold leading-none tracking-tight text-ink">
            MLTrade
          </h1>
          <span className="mt-0.5 font-mono text-[10px] font-semibold tracking-widest text-sage">
            RESEARCH TERMINAL
          </span>
        </div>
        <p className="mt-2 max-w-xl text-[12.5px] leading-snug text-ink-soft">
          Systematic cross-asset research platform — walk-forward backtest, risk
          analytics, and overfitting diagnostics on real, point-in-time market data.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Chip label="DATA" value="REAL · ADJUSTED" tone="good" />
        <Chip label="OOS" value={`${meta.oos_start.slice(0, 4)}–${meta.oos_end.slice(0, 4)}`} />
        <Chip label="UNIVERSE" value={`${meta.n_symbols} ETFs`} />
        <Chip label="MODE" value="PAPER · LIVE OFF" tone="warn" />
        <span className="font-mono text-[10px] text-ink-faint">
          snapshot {dateShort(meta.as_of)}
        </span>
      </div>
    </header>
  );
}

function Chip({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "warn";
}) {
  const cls =
    tone === "good"
      ? "border-sage/50 bg-sage-soft/40 text-pine"
      : tone === "warn"
        ? "border-ochre/40 bg-ochre-soft/30 text-redwood"
        : "border-line bg-surface text-ink-soft";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-mono text-[9.5px] font-semibold tracking-wide ${cls}`}
    >
      <span className="text-ink-faint">{label}</span>
      {value}
    </span>
  );
}

function Mark() {
  return (
    <svg viewBox="0 0 32 32" className="h-7 w-7">
      <path d="M16 29 V14" stroke="var(--color-forest)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M16 18 C16 11 9 8 3 8 C3 15 9 18 16 18 Z" fill="var(--color-sage)" />
      <path d="M16 14 C16 6 23 3 29 3 C29 11 23 14 16 14 Z" fill="var(--color-forest)" />
    </svg>
  );
}
