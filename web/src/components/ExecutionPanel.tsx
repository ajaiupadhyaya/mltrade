import type { Execution } from "../types";
import { Panel, Pill } from "./primitives";
import { thousands } from "../format";

function Side({ side }: { side: string }) {
  const buy = side.toLowerCase() === "buy";
  return (
    <span
      className={`inline-block rounded-md px-2 py-0.5 font-mono text-[9.5px] font-semibold ${
        buy ? "bg-sage-soft text-pine" : "bg-clay-soft text-redwood"
      }`}
    >
      {side.toUpperCase()}
    </span>
  );
}

export function ExecutionPanel({ execution }: { execution: Execution }) {
  return (
    <Panel
      title="Execution Preview"
      subtitle={`${execution.count} intents · ${execution.broker}`}
      right={<Pill tone="good">PREVIEW ONLY</Pill>}
    >
      <div className="flex items-center gap-3 border-b border-line-soft pb-2 font-mono text-[9.5px] tracking-wide text-ink-faint">
        <span className="w-16">SIDE</span>
        <span className="flex-1">SYMBOL</span>
        <span className="w-24 text-right">SHARES</span>
      </div>
      <div className="divide-y divide-line-soft">
        {execution.intents.map((it) => (
          <div key={it.client_order_id} className="flex items-center gap-3 py-[7px]">
            <span className="w-16">
              <Side side={it.side} />
            </span>
            <span className="flex-1 font-mono text-[12.5px] text-ink">
              {it.symbol}
            </span>
            <span className="tnum w-24 text-right font-mono text-[12.5px] text-ink">
              {thousands(it.quantity)}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}
