import type { CheckStatus, Risk } from "../types";
import { Panel, Pill } from "./primitives";

function statusColor(status: CheckStatus): string {
  if (status === "block") return "#BC5B3C";
  if (status === "warn") return "#CE9248";
  return "#3C5E48";
}

function textColor(status: CheckStatus): string {
  if (status === "block") return "text-redwood";
  if (status === "warn") return "text-ochre";
  return "text-ink-soft";
}

export function RiskGates({ risk }: { risk: Risk }) {
  const per = Math.ceil(risk.checks.length / 3);
  const cols = [
    risk.checks.slice(0, per),
    risk.checks.slice(per, per * 2),
    risk.checks.slice(per * 2),
  ];

  const right = risk.blocked ? (
    <Pill tone="bad">
      <span className="h-2 w-2 rounded-full bg-redwood" />
      {risk.summary.block} GATED
    </Pill>
  ) : (
    <Pill tone="good">
      <span className="h-2 w-2 rounded-full bg-pine" />
      ALL CLEAR
    </Pill>
  );

  return (
    <Panel
      title="Risk Gates"
      subtitle={`${risk.checks.length} checks · steady-state caps`}
      right={right}
    >
      <div className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-3">
        {cols.map((col, ci) => (
          <div key={ci} className="space-y-3">
            {col.map((c) => (
              <div
                key={c.code}
                className="flex items-center gap-2.5"
                title={c.message}
              >
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ background: statusColor(c.status) }}
                />
                <span className={`font-mono text-[10.5px] ${textColor(c.status)}`}>
                  {c.code}
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </Panel>
  );
}
