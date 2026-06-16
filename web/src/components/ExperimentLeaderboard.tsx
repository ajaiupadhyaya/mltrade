import type { Experiments, ExperimentRun } from "../types";
import { Panel, Pill } from "./primitives";

function Seedling() {
  return (
    <svg viewBox="0 0 48 48" className="h-11 w-11">
      <path
        d="M24 43 V25"
        className="stroke-sage"
        strokeWidth="2.5"
        fill="none"
        strokeLinecap="round"
      />
      <path
        d="M24 29 C24 21 16 17 8 17 C8 25 16 29 24 29 Z"
        className="fill-sage-soft"
      />
      <path
        d="M24 25 C24 16 32 12 40 12 C40 21 32 25 24 25 Z"
        className="fill-sage"
      />
    </svg>
  );
}

function Empty() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
      <Seedling />
      <p className="font-display text-[15px] font-semibold text-ink">
        No experiment runs yet
      </p>
      <p className="max-w-md text-[12.5px] leading-relaxed text-ink-soft">
        Tune and compare reproducible ridge experiments to populate this
        leaderboard, ranked by robust Sharpe.
      </p>
      <code className="rounded-lg bg-sunk px-3 py-1.5 font-mono text-[11px] text-pine">
        mltrade experiment run experiments/ridge-baseline.toml
      </code>
    </div>
  );
}

const COLS: { key: keyof ExperimentRun; label: string; align: "l" | "r"; w: string }[] =
  [
    { key: "rank", label: "#", align: "l", w: "w-8" },
    { key: "run_id", label: "RUN", align: "l", w: "flex-1" },
    { key: "alpha", label: "ALPHA", align: "r", w: "w-16" },
    { key: "robust_sharpe", label: "ROBUST", align: "r", w: "w-20" },
    { key: "sharpe", label: "SHARPE", align: "r", w: "w-24" },
    { key: "max_drawdown", label: "MAX DD", align: "r", w: "w-20" },
    { key: "turnover", label: "TURNOVER", align: "r", w: "w-20" },
  ];

function RunsTable({ runs }: { runs: ExperimentRun[] }) {
  return (
    <div>
      <div className="flex items-center gap-3 border-b border-line-soft pb-2 font-mono text-[9.5px] tracking-wide text-ink-faint">
        {COLS.map((c) => (
          <span key={c.key} className={`${c.w} ${c.align === "r" ? "text-right" : ""}`}>
            {c.label}
          </span>
        ))}
        <span className="w-24">STATUS</span>
      </div>
      <div className="divide-y divide-line-soft">
        {runs.map((run) => {
          const ok = run.status === "complete";
          return (
            <div key={run.run_id} className="flex items-center gap-3 py-2">
              {COLS.map((c) => (
                <span
                  key={c.key}
                  className={`${c.w} tnum font-mono text-[12px] ${
                    c.align === "r" ? "text-right" : ""
                  } ${c.key === "robust_sharpe" ? "font-semibold text-forest" : "text-ink"}`}
                >
                  {String(run[c.key])}
                </span>
              ))}
              <span className="flex w-24 items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${ok ? "bg-forest" : "bg-clay"}`}
                />
                <span
                  className={`font-mono text-[11px] ${ok ? "text-ink-soft" : "text-redwood"}`}
                >
                  {run.status}
                </span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ExperimentLeaderboard({
  experiments,
}: {
  experiments: Experiments;
}) {
  const populated = experiments.available && experiments.runs.length > 0;
  return (
    <Panel
      title="Experiment Leaderboard"
      subtitle={`ranked by ${experiments.ranked_by} · ridge-trend-v1`}
      right={
        <Pill>
          {populated ? `REGISTRY · ${experiments.runs.length} RUNS` : "REGISTRY · EMPTY"}
        </Pill>
      }
    >
      {populated ? <RunsTable runs={experiments.runs} /> : <Empty />}
    </Panel>
  );
}
