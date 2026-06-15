// Earthy per-instrument colors for allocation + contribution visuals.
export const TICKER_COLORS: Record<string, string> = {
  SPY: "#2E4A3A",
  TLT: "#9C4A2E",
  EFA: "#CE9248",
  VNQ: "#A8754C",
  QQQ: "#3C5E48",
  GLD: "#C8A24A",
  EEM: "#BC5B3C",
  DBC: "#7E8A5A",
  IEF: "#6E8CA0",
  IWM: "#8AA383",
  cash: "#D8C7AD",
};

export function tickerColor(symbol: string): string {
  return TICKER_COLORS[symbol] ?? "#8AA383";
}
