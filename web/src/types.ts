// Shape of the JSON produced by `mltrade export` (src/mltrade/export.py, schema v2).
// All numbers derive from a frozen real-market-data snapshot (no live trading).

export interface Meta {
  platform: string;
  snapshot_id: string;
  as_of: string;
  environment: string;
  live_trading_enabled: boolean;
  data_mode: string;
  synthetic: boolean;
  source: string;
  adjustment: string;
  benchmark: string;
  universe: string[];
  universe_version: string;
  model_version: string;
  feature_version: string;
  reference_equity: number;
  oos_start: string;
  oos_end: string;
  n_sessions: number;
  n_symbols: number;
  generated_at: string;
}

export interface Headline {
  sharpe: number;
  annualized_return: number;
  annualized_volatility: number;
  max_drawdown: number;
  sortino: number;
  calmar: number;
  beta: number;
  alpha_annualized: number;
  alpha_tstat: number;
  alpha_pvalue: number;
  information_ratio: number;
  final_equity: number;
  total_return_multiple: number;
  deflated_sharpe_ratio?: number;
  pbo?: number;
}

export interface EquityPoint {
  date: string;
  strategy: number;
  benchmark: number;
}
export interface DrawdownPoint {
  date: string;
  dd: number;
}
export interface RollingPoint {
  date: string;
  value: number;
}
export interface MonthlyPoint {
  year: number;
  month: number;
  ret: number;
}
export interface YearlyPoint {
  year: number;
  strategy: number;
  benchmark: number;
}
export interface Histogram {
  centres: number[];
  counts: number[];
}
export interface CostSensitivityPoint {
  bps: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe: number;
  max_drawdown: number;
}
export interface EvaluationWindow {
  start: string;
  end: string;
  sharpe: number;
  annualized_return: number;
  max_drawdown: number;
}
export interface SymbolContribution {
  symbol: string;
  return: number;
}

export interface Performance {
  n_sessions: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown: number;
  max_drawdown_duration: number;
  time_to_recovery: number | null;
  skewness: number;
  excess_kurtosis: number;
  best_day: number;
  worst_day: number;
  positive_fraction: number;
  var_95: number;
  cvar_95: number;
  var_99: number;
  cvar_99: number;
  cornish_fisher_var_95: number;
  headline_cost_bps: number;
  turnover: number;
  hit_rate: number;
  total_costs: number;
  equal_weight_return: number;
  cash_return: number;
  equity_curve: EquityPoint[];
  drawdown: DrawdownPoint[];
  rolling_sharpe: RollingPoint[];
  histogram: Histogram;
  monthly: MonthlyPoint[];
  yearly: YearlyPoint[];
  cost_sensitivity: CostSensitivityPoint[];
  evaluation_windows: EvaluationWindow[];
  per_symbol_contribution: SymbolContribution[];
}

export interface BenchmarkStats {
  benchmark: string;
  beta: number;
  alpha_annualized: number;
  alpha_tstat: number;
  alpha_pvalue: number;
  correlation: number;
  r_squared: number;
  tracking_error: number;
  information_ratio: number;
  up_capture: number;
  down_capture: number;
  n_sessions: number;
}

export interface FactorExposure {
  factor: string;
  beta: number;
  tstat: number;
}
export interface Attribution {
  exposures: FactorExposure[];
  alpha_annualized: number;
  alpha_tstat: number;
  r_squared: number;
  n_sessions: number;
}

export interface Overfitting {
  observed_sharpe_annualized: number;
  n_trials: number;
  n_observations: number;
  skewness: number;
  kurtosis: number;
  deflated_threshold_sharpe: number;
  deflated_sharpe_ratio: number;
  psr_vs_zero: number;
  pbo: number;
  pbo_n_splits: number;
  pbo_n_combinations: number;
  logit_median: number;
  logit_histogram: Histogram;
}

export interface Weight {
  symbol: string;
  weight: number;
  asset_class: string;
}
export interface AssetClassWeight {
  asset_class: string;
  weight: number;
}
export interface Portfolio {
  blocked: boolean;
  cash_weight: number;
  invested_weight: number;
  weights: Weight[];
  asset_classes: AssetClassWeight[];
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
  notional: number;
  client_order_id: string;
}
export interface Execution {
  preview_only: boolean;
  broker: string;
  count: number;
  reconciliation_blocked: boolean;
  intents: Intent[];
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

export interface Quality {
  source: string;
  adjustment: string;
  start_session: string;
  end_session: string;
  panel_sessions: number;
  expected_xnys_sessions: number;
  completeness: number;
  excluded_sessions: number;
  n_symbols: number;
  row_count: number;
  content_sha256: string;
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
  compatible?: boolean;
}

export interface DashboardData {
  schema_version: number;
  generated_at: string;
  meta: Meta;
  headline: Headline;
  performance: Performance;
  benchmark: BenchmarkStats;
  attribution: Attribution;
  overfitting: Overfitting | null;
  portfolio: Portfolio;
  risk: Risk;
  execution: Execution;
  forecast: Forecast;
  quality: Quality;
  experiments: Experiments;
}
