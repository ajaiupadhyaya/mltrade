interface DonutSlice {
  label: string;
  value: number;
  color: string;
}

interface DonutProps {
  slices: DonutSlice[];
  size?: number;
  centerLabel?: string;
  centerSub?: string;
}

function arc(cx: number, cy: number, r: number, a0: number, a1: number): string {
  const p0x = cx + r * Math.cos(a0);
  const p0y = cy + r * Math.sin(a0);
  const p1x = cx + r * Math.cos(a1);
  const p1y = cy + r * Math.sin(a1);
  const large = a1 - a0 > Math.PI ? 1 : 0;
  return `M${p0x.toFixed(2)},${p0y.toFixed(2)} A${r},${r} 0 ${large} 1 ${p1x.toFixed(2)},${p1y.toFixed(2)}`;
}

export function Donut({ slices, size = 168, centerLabel, centerSub }: DonutProps) {
  const total = slices.reduce((s, x) => s + x.value, 0) || 1;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 10;
  const stroke = 18;
  let angle = -Math.PI / 2;

  return (
    <svg width={size} height={size} className="block">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--color-sunk)" strokeWidth={stroke} />
      {slices.map((s, i) => {
        const sweep = (s.value / total) * Math.PI * 2;
        const a0 = angle;
        const a1 = angle + sweep;
        angle = a1;
        if (sweep < 0.0001) return null;
        return (
          <path
            key={i}
            d={arc(cx, cy, r, a0, Math.min(a1, a0 + Math.PI * 1.9999))}
            fill="none"
            stroke={s.color}
            strokeWidth={stroke}
            strokeLinecap="butt"
          />
        );
      })}
      {centerLabel && (
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          className="fill-[var(--color-ink)] font-mono"
          fontSize={19}
          fontWeight={600}
        >
          {centerLabel}
        </text>
      )}
      {centerSub && (
        <text
          x={cx}
          y={cy + 14}
          textAnchor="middle"
          className="fill-[var(--color-ink-faint)] font-mono"
          fontSize={9}
        >
          {centerSub}
        </text>
      )}
    </svg>
  );
}
