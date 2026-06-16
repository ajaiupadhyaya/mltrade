import { diverge, lightText } from "./color";
import { monthName } from "../format";
import type { MonthlyPoint } from "../types";

interface HeatmapProps {
  monthly: MonthlyPoint[];
}

// Calendar heatmap of monthly returns (years × months), warm diverging scale.
export function Heatmap({ monthly }: HeatmapProps) {
  const byYear = new Map<number, Map<number, number>>();
  let scale = 0.001;
  for (const m of monthly) {
    if (!byYear.has(m.year)) byYear.set(m.year, new Map());
    byYear.get(m.year)!.set(m.month, m.ret);
    scale = Math.max(scale, Math.abs(m.ret));
  }
  const years = [...byYear.keys()].sort((a, b) => a - b);
  const months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate" style={{ borderSpacing: "2px" }}>
        <thead>
          <tr>
            <th className="w-9" />
            {months.map((m) => (
              <th
                key={m}
                className="font-mono text-[8.5px] font-normal text-ink-faint"
              >
                {monthName(m).slice(0, 1)}
              </th>
            ))}
            <th className="pl-2 text-right font-mono text-[8.5px] font-normal text-ink-faint">
              YR
            </th>
          </tr>
        </thead>
        <tbody>
          {years.map((y) => {
            const row = byYear.get(y)!;
            let yearRet = 0;
            let any = false;
            for (const m of months) {
              if (row.has(m)) {
                yearRet = (1 + yearRet) * (1 + row.get(m)!) - 1;
                any = true;
              }
            }
            return (
              <tr key={y}>
                <td className="pr-1 text-right font-mono text-[9px] text-ink-faint">
                  {y}
                </td>
                {months.map((m) => {
                  const v = row.get(m);
                  if (v === undefined) {
                    return <td key={m} className="h-4 rounded-sm bg-sunk/40" />;
                  }
                  const norm = v / scale;
                  return (
                    <td
                      key={m}
                      title={`${monthName(m)} ${y}: ${(v * 100).toFixed(2)}%`}
                      className="h-4 rounded-sm text-center font-mono text-[7.5px]"
                      style={{
                        background: diverge(norm),
                        color: lightText(norm) ? "#f3ecdd" : "var(--color-ink-soft)",
                      }}
                    >
                      {(v * 100).toFixed(0)}
                    </td>
                  );
                })}
                <td
                  className="rounded-sm pl-1 text-right font-mono text-[9px] font-semibold"
                  style={{ color: any && yearRet < 0 ? "var(--color-redwood)" : "var(--color-forest)" }}
                >
                  {any ? `${yearRet >= 0 ? "+" : "−"}${Math.abs(yearRet * 100).toFixed(0)}` : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
