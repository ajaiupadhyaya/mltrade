// Deterministic, locale-independent formatters for the dashboard.

export function pct(value: number, dp = 1): string {
  return `${(value * 100).toFixed(dp)}%`;
}

export function signedPct(value: number, dp = 1): string {
  const s = (value * 100).toFixed(dp);
  return value > 0 ? `+${s}%` : `${s}%`;
}

export function num(value: number, dp = 2): string {
  return value.toFixed(dp);
}

export function thousands(value: number): string {
  return Math.round(value)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Compact money: $1.00M, $948K, $2.34M.
export function moneyCompact(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `$${(value / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `$${(value / 1_000).toFixed(0)}K`;
  return `$${value.toFixed(0)}`;
}

export function shortDate(iso: string): string {
  // "2026-06-12" -> "Jun 2026" style is overkill here; keep the ISO date.
  return iso;
}

export function year(iso: string): string {
  return iso.slice(0, 4);
}
