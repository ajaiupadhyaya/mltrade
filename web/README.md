# MLTrade Dashboard

A warm, local-first dashboard for the MLTrade systematic-trading platform —
backtest performance, portfolio allocation, the 17 pre-trade risk gates,
execution preview, and the research-experiment leaderboard.

Built with Vite + React + TypeScript + Tailwind CSS v4. Hand-rolled SVG charts,
no charting dependency. "Redwood" design system (warm 70s California forest
palette) — the source design lives in `../design/` (Pencil + exported PNG).

## Data

The dashboard reads `public/data/dashboard.json`, produced by the platform CLI:

```bash
# from the repo root (Python project)
uv run mltrade export                       # writes web/public/data/dashboard.json
```

The JSON is deterministic and derived entirely from offline fixture data — no
network, no secrets, live trading disabled. A committed sample is included so
the dashboard runs out-of-the-box.

## Develop

```bash
cd web
npm install
npm run dev        # http://localhost:5173
```

## Build

```bash
npm run build      # type-check + production build to dist/
npm run preview    # serve the production build
```

## Notes

- Fonts (Fraunces / Hanken Grotesk / IBM Plex Mono) load from Google Fonts with
  system-font fallbacks; the dashboard works offline minus the web fonts.
- The experiment leaderboard shows an empty state until the research-experiment
  platform is merged and `mltrade export` can read its registry.
