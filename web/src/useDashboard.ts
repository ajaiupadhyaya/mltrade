import { useEffect, useState } from "react";
import type { DashboardData } from "./types";

type State =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: DashboardData };

// Loads the JSON produced by `mltrade export`. Served from /public/data so the
// dashboard refreshes whenever the export is re-run — no rebuild required.
export function useDashboard(): State {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    const url = `${import.meta.env.BASE_URL}data/dashboard.json`;

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
        return res.json() as Promise<DashboardData>;
      })
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          const message =
            err instanceof Error ? err.message : "Failed to load dashboard data";
          setState({ status: "error", message });
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
