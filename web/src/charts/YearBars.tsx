import { useMemo } from "react";
import { niceTicks, linear } from "./scale";
import { useMeasure } from "./useMeasure";

export interface YearDatum {
  year: number;
  strategy: number;
  benchmark: number;
}

interface YearBarsProps {
  data: YearDatum[];
  height?: number;
  format: (n: number) => string;
}

const PAD = { l: 42, r: 10, t: 10, b: 22 };

// Grouped per-year bars: strategy vs benchmark, with a zero baseline.
export function YearBars({ data, height = 260, format }: YearBarsProps) {
  const [ref, width] = useMeasure<HTMLDivElement>();
  const plotW = Math.max(10, width - PAD.l - PAD.r);
  const plotH = height - PAD.t - PAD.b;

  const ticks = useMemo(() => {
    const vals = data.flatMap((d) => [d.strategy, d.benchmark]);
    return niceTicks(Math.min(0, ...vals), Math.max(0, ...vals), 5);
  }, [data]);

  const ty = (v: number) => linear(v, ticks.lo, ticks.hi, PAD.t + plotH, PAD.t);
  const n = data.length;
  const groupW = plotW / Math.max(1, n);
  const barW = (groupW * 0.74) / 2;
  const zero = ty(0);

  const labelStride = Math.ceil(n / 12);

  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={height} className="block">
        {ticks.values.map((v, k) => {
          const y = ty(v);
          return (
            <g key={k}>
              <line
                x1={PAD.l}
                x2={PAD.l + plotW}
                y1={y}
                y2={y}
                stroke={Math.abs(v) < 1e-9 ? "var(--color-line)" : "var(--color-grid)"}
                strokeWidth={1}
                shapeRendering="crispEdges"
              />
              <text
                x={PAD.l - 7}
                y={y + 3}
                textAnchor="end"
                className="fill-[var(--color-ink-faint)] font-mono"
                fontSize={9}
              >
                {format(v)}
              </text>
            </g>
          );
        })}

        {data.map((d, i) => {
          const cx = PAD.l + groupW * i + groupW / 2;
          const sTop = ty(Math.max(0, d.strategy));
          const sH = Math.abs(ty(d.strategy) - zero);
          const bTop = ty(Math.max(0, d.benchmark));
          const bH = Math.abs(ty(d.benchmark) - zero);
          return (
            <g key={d.year}>
              <rect
                x={cx - barW - 1}
                y={sTop}
                width={barW}
                height={Math.max(0.5, sH)}
                rx={1}
                fill="var(--color-strategy)"
              />
              <rect
                x={cx + 1}
                y={bTop}
                width={barW}
                height={Math.max(0.5, bH)}
                rx={1}
                fill="var(--color-benchmark)"
                opacity={0.85}
              />
              {i % labelStride === 0 && (
                <text
                  x={cx}
                  y={height - 7}
                  textAnchor="middle"
                  className="fill-[var(--color-ink-faint)] font-mono"
                  fontSize={8.5}
                >
                  {`'${String(d.year).slice(2)}`}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
