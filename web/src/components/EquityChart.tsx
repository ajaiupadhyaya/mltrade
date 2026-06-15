import type { EquityPoint } from "../types";
import { moneyCompact, year } from "../format";

// Hand-rolled responsive SVG area+line chart for the strategy equity curve.
export function EquityChart({
  points,
  initialEquity,
}: {
  points: EquityPoint[];
  initialEquity: number;
}) {
  const W = 1000;
  const H = 300;
  const padL = 12;
  const padR = 16;
  const padT = 18;
  const padB = 30;
  const n = points.length;
  if (n < 2) return null;

  const equities = points.map((p) => p.equity);
  const minE = Math.min(...equities, initialEquity);
  const maxE = Math.max(...equities);
  const pad = (maxE - minE) * 0.06 || 1;
  const lo = minE - pad;
  const hi = maxE + pad;

  const x = (i: number) => padL + (i / (n - 1)) * (W - padL - padR);
  const y = (e: number) => padT + (1 - (e - lo) / (hi - lo)) * (H - padT - padB);

  const line = points
    .map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`)
    .join(" ");
  const area = `${line} L${x(n - 1).toFixed(1)},${(H - padB).toFixed(1)} L${x(
    0,
  ).toFixed(1)},${(H - padB).toFixed(1)} Z`;

  const gridLevels = 4;
  const grid = Array.from(
    { length: gridLevels + 1 },
    (_, k) => lo + (hi - lo) * (k / gridLevels),
  );

  const yearMarks: { i: number; label: string }[] = [];
  let prevYear = "";
  points.forEach((p, i) => {
    const yy = year(p.date);
    if (yy !== prevYear) {
      yearMarks.push({ i, label: yy });
      prevYear = yy;
    }
  });

  const last = points[n - 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full">
      {grid.map((g, k) => (
        <line
          key={k}
          x1={padL}
          x2={W - padR}
          y1={y(g)}
          y2={y(g)}
          className="stroke-line-soft"
          strokeWidth={1}
        />
      ))}

      {/* $1.00M starting-capital reference */}
      <line
        x1={padL}
        x2={W - padR}
        y1={y(initialEquity)}
        y2={y(initialEquity)}
        className="stroke-sage"
        strokeWidth={1}
        strokeDasharray="3 6"
      />

      <path d={area} className="fill-forest/10" />
      <path
        d={line}
        className="fill-none stroke-forest"
        strokeWidth={2.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      <circle cx={x(n - 1)} cy={y(last.equity)} r={4.5} className="fill-forest" />

      <text
        x={padL}
        y={y(initialEquity) - 6}
        className="fill-ink-faint font-mono"
        fontSize={11}
      >
        {moneyCompact(initialEquity)}
      </text>
      <text
        x={W - padR}
        y={y(last.equity) - 10}
        textAnchor="end"
        className="fill-forest font-mono"
        fontSize={12}
        fontWeight={600}
      >
        {moneyCompact(last.equity)}
      </text>

      {yearMarks.map((m) => (
        <text
          key={m.label}
          x={x(m.i)}
          y={H - 8}
          textAnchor={m.i === 0 ? "start" : "middle"}
          className="fill-ink-faint font-mono"
          fontSize={11}
        >
          {m.label}
        </text>
      ))}
    </svg>
  );
}
