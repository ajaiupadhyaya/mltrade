// Horizontal labelled bars; supports diverging (negative extends left).

export interface HBarRow {
  label: string;
  value: number;
  sub?: string;
  color?: string;
}

interface HBarsProps {
  rows: HBarRow[];
  format: (n: number) => string;
  diverging?: boolean;
  labelWidth?: string;
  barColor?: string;
  negColor?: string;
}

export function HBars({
  rows,
  format,
  diverging = false,
  labelWidth = "w-24",
  barColor = "var(--color-forest)",
  negColor = "var(--color-redwood)",
}: HBarsProps) {
  const max = Math.max(1e-9, ...rows.map((r) => Math.abs(r.value)));

  return (
    <div className="space-y-2.5">
      {rows.map((r) => {
        const frac = Math.abs(r.value) / max;
        const color = r.color ?? (r.value < 0 ? negColor : barColor);
        return (
          <div key={r.label} className="flex items-center gap-3">
            <div className={`${labelWidth} shrink-0`}>
              <div className="truncate text-[12px] font-medium text-ink">{r.label}</div>
              {r.sub && (
                <div className="font-mono text-[9.5px] text-ink-faint">{r.sub}</div>
              )}
            </div>
            {diverging ? (
              <div className="relative h-3 flex-1">
                <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                <div
                  className="absolute top-0 h-3 rounded-sm"
                  style={
                    r.value >= 0
                      ? { left: "50%", width: `${(frac * 50).toFixed(1)}%`, background: color }
                      : { right: "50%", width: `${(frac * 50).toFixed(1)}%`, background: color }
                  }
                />
              </div>
            ) : (
              <div className="h-3 flex-1 overflow-hidden rounded-sm bg-sunk">
                <div
                  className="h-full rounded-sm"
                  style={{ width: `${(frac * 100).toFixed(1)}%`, background: color }}
                />
              </div>
            )}
            <span className="tnum w-16 shrink-0 text-right font-mono text-[11.5px] font-medium text-ink">
              {format(r.value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
