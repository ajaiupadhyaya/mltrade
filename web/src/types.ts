// Shape of the JSON produced by `mltrade export` (src/mltrade/export.py).

export interface Meta {
  platform: string;
  snapshot_id: string;
  environment: string;
  live_trading_enabled: boolean;
  last_session: string;
  reference_equity: number;
  model_version: string;
  feature_version: string;
  universe_version: string;
  universe: string[];
  data_mode: string;
  synthetic: boolean;
}

export interface CostSensitivityPoint {
  bps: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe: number;
  max_drawdown: number;
}

export interface EquityPoint {
  date: string;
  equity: number;
}

export interface SymbolContribution {
  symbol: string;
  return: number;
}

export interface Backtest {
  sessions: number;
  headline_cost_bps: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe: number;
  max_drawdown: number;
  turnover: number;
  total_costs: number;
  hit_rate: number;
  equal_weight_return: number;
  cash_return: number;
  cost_sensitivity: CostSensitivityPoint[];
  per_symbol_contribution: SymbolContribution[];
  equity_curve: EquityPoint[];
}

export interface Weight {
  symbol: string;
  weight: number;
}

export interface Portfolio {
  blocked: boolean;
  cash_weight: number;
  invested_weight: number;
  weights: Weight[];
}

export type CheckStatus = "pass" | "warn" | "block";

export interface RiskCheck {
  code: string;
  status: CheckStatus;
  message: string;
}

export interface Risk {
  blocked: boolean;
  summary: { pass: number; warn: number; block: number };
  checks: RiskCheck[];
}

export interface Intent {
  side: string;
  symbol: string;
  quantity: number;
  client_order_id: string;
}

export interface Execution {
  preview_only: boolean;
  broker: string;
  count: number;
  reconciliation_blocked: boolean;
  intents: Intent[];
}

export interface QualityIssue {
  code: string;
  severity: string;
  message: string;
}

export interface Quality {
  blocked: boolean;
  issues_count: number;
  issues: QualityIssue[];
  training_rows: number;
  training_sessions: number;
}

export interface ForecastPoint {
  symbol: string;
  predicted_forward_return: number;
}

export interface Forecast {
  decision_session: string;
  model_version: string;
  count: number;
  training_row_count: number;
  training_session_count: number;
  forecasts: ForecastPoint[];
}

export interface ExperimentRun {
  rank: number | string;
  run_id: string;
  alpha: string;
  robust_sharpe: string;
  sharpe: string;
  max_drawdown: string;
  turnover: string;
  status: string;
}

export interface Experiments {
  available: boolean;
  ranked_by: string;
  runs: ExperimentRun[];
  note?: string;
}

export interface DashboardData {
  schema_version: number;
  generated_at: string;
  meta: Meta;
  backtest: Backtest;
  portfolio: Portfolio;
  risk: Risk;
  execution: Execution;
  quality: Quality;
  forecast: Forecast;
  experiments: Experiments;
}
