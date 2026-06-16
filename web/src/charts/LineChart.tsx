import { useId, useMemo, useState } from "react";
import { extent, linear, niceTicks } from "./scale";
import { useMeasure } from "./useMeasure";

export interface LineSeries {
  name: string;
  color: string;
  values: number[];
  dashed?: boolean;
  area?: boolean;
}

interface LineChartProps {
  labels: string[];
  series: LineSeries[];
  height?: number;
  yFormat: (n: number) => string;
  tipFormat?: (n: number) => string;
  logY?: boolean;
  zeroLine?: boolean;
  yTickCount?: number;
}

const PAD = { l: 54, r: 16, t: 12, b: 24 };

export function LineChart({
  labels,
  series,
  height = 300,
  yFormat,
  tipFormat,
  logY = false,
  zeroLine = false,
  yTickCount = 5,
}: LineChartProps) {
  const [ref, width] = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const gid = useId().replace(/:/g, "");

  const n = labels.length;
  const plotW = Math.max(10, width - PAD.l - PAD.r);
  const plotH = height - PAD.t - PAD.b;

  const fmt = tipFormat ?? yFormat;

  const { yLo, yHi, ticks, ty } = useMemo(() => {
    const all = series.flatMap((s) => s.values);
    let [lo, hi] = extent(all);
    if (logY) {
      const positive = all.filter((v) => v > 0);
      [lo, hi] = extent(positive.length ? positive : [1, 10]);
      const lLo = Math.log10(lo);
      const lHi = Math.log10(hi);
      const t = niceTicks(lLo, lHi, yTickCount);
      const toY = (v: number) =>
        linear(Math.log10(v), t.lo, t.hi, PAD.t + plotH, PAD.t);
      return {
        yLo: 10 ** t.lo,
        yHi: 10 ** t.hi,
        ticks: t.values.map((v) => 10 ** v),
        ty: toY,
      };
    }
    const t = niceTicks(lo, hi, yTickCount);
    const toY = (v: number) => linear(v, t.lo, t.hi, PAD.t + plotH, PAD.t);
    return { yLo: t.lo, yHi: t.hi, ticks: t.values, ty: toY };
  }, [series, logY, plotH, yTickCount]);

  const tx = (i: number) =>
    n <= 1 ? PAD.l + plotW / 2 : linear(i, 0, n - 1, PAD.l, PAD.l + plotW);

  const linePath = (vals: number[]) =>
    vals
      .map((v, i) => `${i === 0 ? "M" : "L"}${tx(i).toFixed(1)},${ty(v).toFixed(1)}`)
      .join(" ");

  const areaPath = (vals: number[]) => {
    const top = vals
      .map((v, i) => `${i === 0 ? "M" : "L"}${tx(i).toFixed(1)},${ty(v).toFixed(1)}`)
      .join(" ");
    const baseY = ty(logY ? yLo : Math.max(yLo, 0));
    return `${top} L${tx(n - 1).toFixed(1)},${baseY.toFixed(1)} L${tx(0).toFixed(1)},${baseY.toFixed(1)} Z`;
  };

  // Year tick positions from ISO labels.
  const xTicks = useMemo(() => {
    const out: { i: number; label: string }[] = [];
    let lastYear = "";
    labels.forEach((iso, i) => {
      const y = iso.slice(0, 4);
      if (y !== lastYear) {
        out.push({ i, label: y });
        lastYear = y;
      }
    });
    const stride = Math.ceil(out.length / 9);
    return out.filter((_, k) => k % stride === 0);
  }, [labels]);

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left - PAD.l;
    const i = Math.round((mx / plotW) * (n - 1));
    setHover(Math.min(n - 1, Math.max(0, i)));
  };

  const hx = hover != null ? tx(hover) : 0;
  const tipRight = hover != null && hx > PAD.l + plotW * 0.62;

  return (
    <div ref={ref} className="relative w-full select-none">
      <svg width={width} height={height} className="block">
        <defs>
          {series.map((s, k) =>
            s.area ? (
              <linearGradient key={k} id={`${gid}-a${k}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity="0.18" />
                <stop offset="100%" stopColor={s.color} stopOpacity="0" />
              </linearGradient>
            ) : null,
          )}
        </defs>

        {/* gridlines + y labels */}
        {ticks.map((v, k) => {
          const y = ty(v);
          if (y < PAD.t - 1 || y > PAD.t + plotH + 1) return null;
          return (
            <g key={k}>
              <line
                x1={PAD.l}
                x2={PAD.l + plotW}
                y1={y}
                y2={y}
                stroke="var(--color-grid)"
                strokeWidth={1}
                shapeRendering="crispEdges"
              />
              <text
                x={PAD.l - 8}
                y={y + 3}
                textAnchor="end"
                className="fill-[var(--color-ink-faint)] font-mono"
                fontSize={9.5}
              >
                {yFormat(v)}
              </text>
            </g>
          );
        })}

        {zeroLine && yLo < 0 && yHi > 0 && (
          <line
            x1={PAD.l}
            x2={PAD.l + plotW}
            y1={ty(0)}
            y2={ty(0)}
            stroke="var(--color-line)"
            strokeWidth={1}
          />
        )}

        {/* x labels */}
        {xTicks.map((t, k) => (
          <text
            key={k}
            x={tx(t.i)}
            y={height - 7}
            textAnchor="middle"
            className="fill-[var(--color-ink-faint)] font-mono"
            fontSize={9.5}
          >
            {t.label}
          </text>
        ))}

        {/* areas then lines */}
        {series.map((s, k) =>
          s.area ? (
            <path key={`area${k}`} d={areaPath(s.values)} fill={`url(#${gid}-a${k})`} />
          ) : null,
        )}
        {series.map((s, k) => (
          <path
            key={`line${k}`}
            d={linePath(s.values)}
            fill="none"
            stroke={s.color}
            strokeWidth={1.6}
            strokeDasharray={s.dashed ? "4 3" : undefined}
            strokeLinejoin="round"
          />
        ))}

        {/* crosshair */}
        {hover != null && (
          <g pointerEvents="none">
            <line
              x1={hx}
              x2={hx}
              y1={PAD.t}
              y2={PAD.t + plotH}
              stroke="var(--color-ink-soft)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            {series.map((s, k) => (
              <circle
                key={k}
                cx={hx}
                cy={ty(s.values[hover])}
                r={3}
                fill="var(--color-surface)"
                stroke={s.color}
                strokeWidth={1.6}
              />
            ))}
          </g>
        )}

        <rect
          x={PAD.l}
          y={PAD.t}
          width={plotW}
          height={plotH}
          fill="transparent"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
      </svg>

      {hover != null && (
        <div
          className="pointer-events-none absolute top-2 z-10 rounded-lg border border-line bg-raise/95 px-3 py-2 shadow-sm backdrop-blur"
          style={tipRight ? { right: PAD.r + 4 } : { left: PAD.l + 4 }}
        >
          <div className="mb-1 font-mono text-[10px] text-ink-faint">
            {labels[hover]}
          </div>
          {series.map((s, k) => (
            <div key={k} className="flex items-center gap-2 text-[11px]">
              <span
                className="inline-block h-1.5 w-3 rounded-full"
                style={{ background: s.color }}
              />
              <span className="text-ink-soft">{s.name}</span>
              <span className="tnum ml-auto font-mono font-medium text-ink">
                {fmt(s.values[hover])}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
