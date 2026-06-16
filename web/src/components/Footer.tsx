import type { DashboardData } from "../types";
import { dateShort } from "../format";

export function Footer({ data }: { data: DashboardData }) {
  const q = data.quality;
  return (
    <footer className="mt-8 border-t border-line pt-5 text-[11px] leading-relaxed text-ink-faint">
      <p className="max-w-3xl">
        <span className="font-semibold text-ink-soft">Research artifact, not investment advice.</span>{" "}
        All figures are out-of-sample backtest results on a frozen, point-in-time
        snapshot of real {q.adjustment} daily bars ({q.source}). Past performance
        does not guarantee future results. Live trading is structurally disabled in
        this build; the execution preview is simulated.
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 font-mono">
        <span>MLTrade · {data.meta.model_version}</span>
        <span>snapshot {dateShort(data.meta.as_of)}</span>
        <span>{q.panel_sessions.toLocaleString()} sessions</span>
        <span>sha256 {q.content_sha256.slice(0, 12)}…</span>
        <span>generated {data.generated_at.slice(0, 10)}</span>
      </div>
    </footer>
  );
}
