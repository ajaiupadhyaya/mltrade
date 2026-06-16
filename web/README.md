# MLTrade Research Terminal

A warm, institutional research terminal for the MLTrade systematic cross-asset
platform — real out-of-sample backtest performance, risk analytics, factor
attribution, and backtest-overfitting diagnostics, on a frozen point-in-time
snapshot of real market data.

Built with Vite + React + TypeScript + Tailwind CSS v4. Hand-rolled, interactive
SVG charts (no charting dependency). "Redwood" design system — a warm 70s
California-forest palette tuned for a dense, precise terminal.

## What it shows

Seven views, switchable from the top nav:

- **Overview** — growth of $1M vs SPY (log), KPI strip, the honest read, calendar-year returns, key statistics.
- **Performance** — underwater drawdown, rolling Sharpe, monthly-return heatmap, return distribution, cost sensitivity, evaluation windows.
- **Risk** — historical & Cornish-Fisher VaR/CVaR, distribution moments, the pre-trade risk gates.
- **Attribution** — returns-based macro-factor exposures (with t-stats), systematic-vs-idiosyncratic variance decomposition, per-symbol contribution.
- **Integrity** — Deflated Sharpe Ratio and Probability of Backtest Overfitting (PBO via CSCV), methodology, and auditable data provenance.
- **Portfolio** — target allocation, position weights, execution preview, forecast cross-section.
- **Experiments** — the reproducible research-experiment leaderboard.

## Data

The dashboard reads `public/data/dashboard.json`, produced by the platform CLI:

```bash
# from the repo root (Python project)
uv run mltrade export        # writes web/public/data/dashboard.json
```

Every number derives from the committed, frozen real-data snapshot under
`data/snapshots/real/` (split/dividend-adjusted daily bars, 2007→2026). The
export is deterministic and offline — no network, no secrets, live trading
structurally disabled. A committed sample JSON ships so the terminal runs
out-of-the-box.

To refresh the underlying snapshot (rarely needed), re-run the one-off fetcher:

```bash
uv run --with yfinance python scripts/fetch_real_snapshot.py
uv run python scripts/compute_trials.py      # overfitting trials matrix
uv run mltrade export
```

## Live

Deployed (production): **https://mltrade-dashboard.vercel.app**

Redeploy after regenerating data: `cd web && npx vercel deploy --prod --scope ajaiupadhyayas-projects`.

## Develop

```bash
cd web
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production build to dist/
```

## Notes

- Fonts (Fraunces / Hanken Grotesk / IBM Plex Mono) load from Google Fonts with
  system fallbacks; the terminal works offline minus the web fonts.
- The experiment leaderboard shows an empty state until `mltrade experiment run`
  populates the registry.
