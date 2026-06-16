// Earthy per-instrument + per-asset-class colors for allocation visuals.
export const TICKER_COLORS: Record<string, string> = {
  SPY: "#2e4a3a",
  QQQ: "#3c5e48",
  IWM: "#8aa383",
  EFA: "#ce9248",
  EEM: "#bc5b3c",
  TLT: "#5f8198",
  IEF: "#7fa0b4",
  GLD: "#c8a24a",
  DBC: "#7e8a5a",
  VNQ: "#6d4a5b",
  cash: "#d8c7ad",
};

export function tickerColor(symbol: string): string {
  return TICKER_COLORS[symbol] ?? "#8aa383";
}

export const ASSET_CLASS_COLORS: Record<string, string> = {
  "US Equity": "#2e4a3a",
  "Intl Equity": "#5c7b58",
  Rates: "#5f8198",
  Gold: "#c8a24a",
  Commodities: "#7e8a5a",
  "Real Estate": "#6d4a5b",
  Other: "#b6a98f",
};

export function classColor(name: string): string {
  return ASSET_CLASS_COLORS[name] ?? "#b6a98f";
}
