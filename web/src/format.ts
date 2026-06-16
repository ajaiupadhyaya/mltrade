// Number / date formatting helpers — tabular, terminal-grade.

export function num(x: number, dp = 2): string {
  return x.toLocaleString("en-US", {
    minimumFractionDigits: dp,
    maximumFractionDigits: dp,
  });
}

export function signed(x: number, dp = 2): string {
  const s = num(Math.abs(x), dp);
  return `${x < 0 ? "−" : "+"}${s}`;
}

export function pct(x: number, dp = 1): string {
  return `${num(x * 100, dp)}%`;
}

export function signedPct(x: number, dp = 1): string {
  const s = num(Math.abs(x) * 100, dp);
  return `${x < 0 ? "−" : "+"}${s}%`;
}

export function bps(x: number): string {
  return `${num(x * 10000, 0)} bps`;
}

export function money(x: number): string {
  return `$${x.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

export function moneyCompact(x: number): string {
  const abs = Math.abs(x);
  if (abs >= 1_000_000) return `$${num(x / 1_000_000, 2)}M`;
  if (abs >= 1_000) return `$${num(x / 1_000, 0)}K`;
  return `$${num(x, 0)}`;
}

export function multiple(x: number): string {
  return `${num(x, 2)}×`;
}

export function thousands(x: number): string {
  return x.toLocaleString("en-US");
}

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

export function monthName(m: number): string {
  return MONTHS[m - 1] ?? String(m);
}

export function year(iso: string): string {
  return iso.slice(0, 4);
}

export function shortDate(iso: string): string {
  return iso;
}

export function dateShort(iso: string): string {
  const [y, m, d] = iso.split("-");
  return `${monthName(Number(m))} ${Number(d)}, ${y}`;
}

export function dateMed(iso: string): string {
  const [y, m] = iso.split("-");
  return `${monthName(Number(m))} ${y}`;
}

// Short t-stat verdict used across the integrity views.
export function significance(absT: number): string {
  if (absT >= 2.58) return "p < 0.01";
  if (absT >= 1.96) return "p < 0.05";
  if (absT >= 1.64) return "p < 0.10";
  return "not significant";
}
