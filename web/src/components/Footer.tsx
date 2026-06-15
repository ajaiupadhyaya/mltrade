import type { DashboardData } from "../types";
import { thousands } from "../format";

function Sep() {
  return <span className="text-line">·</span>;
}

export function Footer({ data }: { data: DashboardData }) {
  const { quality, meta } = data;
  return (
    <footer className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-line bg-surface px-5 py-3.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-ink-soft">
        <span className="flex items-center gap-2">
          <span
            className={`h-2 w-2 rounded-full ${
              quality.blocked ? "bg-clay" : "bg-forest"
            }`}
          />
          data quality: {quality.blocked ? "blocked" : "pass"}
        </span>
        <Sep />
        <span>{quality.issues_count} issues</span>
        <Sep />
        <span>{thousands(quality.training_rows)} training rows</span>
        <Sep />
        <span>{thousands(quality.training_sessions)} sessions</span>
        <Sep />
        <span>snapshot {meta.snapshot_id}</span>
      </div>
      <div className="flex items-center gap-3">
        <span className="rounded-full bg-canvas-deep px-3 py-1.5 font-mono text-[9.5px] tracking-wide text-ink-soft">
          OFFLINE FIXTURE · SYNTHETIC DATA · NO LIVE TRADING
        </span>
        <span className="font-mono text-[10.5px] text-ink-faint">
          generated {data.generated_at.slice(0, 10)}
        </span>
      </div>
    </footer>
  );
}
