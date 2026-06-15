import type { Meta } from "../types";
import { Chip } from "./primitives";

function Emblem() {
  return (
    <div className="grid h-12 w-12 shrink-0 place-items-center rounded-xl bg-pine">
      <svg viewBox="0 0 44 44" className="h-12 w-12">
        <circle cx="23" cy="18" r="9" className="fill-ochre" />
        <path d="M7 38 L16 26 L22 32 L30 21 L38 38 Z" className="fill-sage" />
      </svg>
    </div>
  );
}

export function Header({ meta }: { meta: Meta }) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-3.5">
        <Emblem />
        <div>
          <div className="font-display text-[1.7rem] font-semibold leading-none text-ink">
            MLTrade
          </div>
          <div className="mt-1.5 font-mono text-[10px] tracking-[0.18em] text-ink-faint">
            RESEARCH · RISK · EXECUTION
          </div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2.5">
        <Chip label="SNAPSHOT" value={meta.snapshot_id} />
        <Chip label="ENV" value={meta.environment} tone="good" />
        <Chip
          label="LIVE"
          value={meta.live_trading_enabled ? "enabled" : "disabled"}
          tone={meta.live_trading_enabled ? "good" : "bad"}
        />
        <Chip label="LAST SESSION" value={meta.last_session} />
      </div>
    </header>
  );
}
