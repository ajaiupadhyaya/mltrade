import { useMemo } from "react";
import { extent, linear } from "./scale";
import { useMeasure } from "./useMeasure";

interface HistogramProps {
  centres: number[];
  counts: number[];
  height?: number;
  color?: string;
  markers?: { value: number; label: string; color: string }[];
  format: (n: number) => string;
}

const PAD = { l: 10, r: 10, t: 8, b: 20 };

// Vertical distribution histogram with optional vertical markers (e.g. VaR, 0).
export function Histogram({
  centres,
  counts,
  height = 200,
  color = "var(--color-moss)",
  markers = [],
  format,
}: HistogramProps) {
  const [ref, width] = useMeasure<HTMLDivElement>();
  const plotW = Math.max(10, width - PAD.l - PAD.r);
  const plotH = height - PAD.t - PAD.b;

  const { x0, x1, maxCount } = useMemo(() => {
    const [lo, hi] = extent(centres);
    return { x0: lo, x1: hi, maxCount: Math.max(1, ...counts) };
  }, [centres, counts]);

  const n = centres.length;
  const bw = n > 0 ? (plotW / n) * 0.84 : 0;
  const tx = (v: number) => linear(v, x0, x1, PAD.l, PAD.l + plotW);
  const ty = (c: number) => linear(c, 0, maxCount, PAD.t + plotH, PAD.t);

  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={height} className="block">
        {centres.map((c, i) => {
          const h = PAD.t + plotH - ty(counts[i]);
          return (
            <rect
              key={i}
              x={tx(c) - bw / 2}
              y={ty(counts[i])}
              width={bw}
              height={Math.max(0, h)}
              rx={1}
              fill={color}
              opacity={0.85}
            />
          );
        })}
        {markers.map((m, k) => {
          const x = tx(m.value);
          if (x < PAD.l || x > PAD.l + plotW) return null;
          return (
            <g key={k}>
              <line
                x1={x}
                x2={x}
                y1={PAD.t}
                y2={PAD.t + plotH}
                stroke={m.color}
                strokeWidth={1.4}
                strokeDasharray="3 2"
              />
              <text
                x={x}
                y={PAD.t + 8}
                textAnchor="middle"
                className="font-mono"
                fontSize={9}
                fill={m.color}
              >
                {m.label}
              </text>
            </g>
          );
        })}
        {[x0, (x0 + x1) / 2, x1].map((v, k) => (
          <text
            key={k}
            x={tx(v)}
            y={height - 6}
            textAnchor="middle"
            className="fill-[var(--color-ink-faint)] font-mono"
            fontSize={9}
          >
            {format(v)}
          </text>
        ))}
      </svg>
    </div>
  );
}
