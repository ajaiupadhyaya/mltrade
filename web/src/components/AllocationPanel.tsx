import type { Portfolio } from "../types";
import { Panel } from "./primitives";
import { tickerColor } from "../palette";
import { pct } from "../format";

function ringArc(
  cx: number,
  cy: number,
  rO: number,
  rI: number,
  a0: number,
  a1: number,
): string {
  const pt = (r: number, a: number): [number, number] => [
    cx + r * Math.cos(a),
    cy + r * Math.sin(a),
  ];
  const large = a1 - a0 > Math.PI ? 1 : 0;
  const [x0, y0] = pt(rO, a0);
  const [x1, y1] = pt(rO, a1);
  const [x2, y2] = pt(rI, a1);
  const [x3, y3] = pt(rI, a0);
  return (
    `M${x0.toFixed(2)},${y0.toFixed(2)} ` +
    `A${rO},${rO} 0 ${large} 1 ${x1.toFixed(2)},${y1.toFixed(2)} ` +
    `L${x2.toFixed(2)},${y2.toFixed(2)} ` +
    `A${rI},${rI} 0 ${large} 0 ${x3.toFixed(2)},${y3.toFixed(2)} Z`
  );
}

export function AllocationPanel({ portfolio }: { portfolio: Portfolio }) {
  const slices = [
    ...portfolio.weights.map((w) => ({ symbol: w.symbol, weight: w.weight })),
    { symbol: "cash", weight: portfolio.cash_weight },
  ];
  const total = slices.reduce((s, x) => s + x.weight, 0) || 1;

  const cx = 110;
  const cy = 110;
  const rO = 100;
  const rI = 62;
  let angle = -Math.PI / 2;
  const arcs = slices.map((s) => {
    const a0 = angle;
    const a1 = angle + (s.weight / total) * Math.PI * 2;
    angle = a1;
    return { ...s, d: ringArc(cx, cy, rO, rI, a0, a1) };
  });

  return (
    <Panel title="Portfolio Allocation" subtitle="target weights · mvp-etf-v1">
      <div className="flex items-center gap-7">
        <div className="relative shrink-0">
          <svg viewBox="0 0 220 220" className="h-[176px] w-[176px]">
            {arcs.map((arc) => (
              <path key={arc.symbol} d={arc.d} fill={tickerColor(arc.symbol)} />
            ))}
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="tnum font-display text-[1.45rem] font-semibold text-ink">
              {pct(portfolio.invested_weight, 0)}
            </div>
            <div className="font-mono text-[9px] tracking-wide text-ink-faint">
              INVESTED
            </div>
          </div>
        </div>
        <div className="grid flex-1 grid-cols-2 gap-x-6 gap-y-2.5">
          {slices.map((s) => (
            <div key={s.symbol} className="flex items-center gap-2.5">
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: tickerColor(s.symbol) }}
              />
              <span className="flex-1 font-mono text-[12px] text-ink">
                {s.symbol}
              </span>
              <span className="tnum font-mono text-[12px] text-ink-soft">
                {pct(s.weight)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}
