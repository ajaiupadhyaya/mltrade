import type { ReactNode } from "react";

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-2xl border border-line bg-surface ${className}`}>
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 px-6 pb-3.5 pt-5">
          <div>
            {title && (
              <h2 className="font-display text-[1.06rem] font-semibold leading-tight text-ink">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-1 font-mono text-[11px] text-ink-faint">{subtitle}</p>
            )}
          </div>
          {right}
        </header>
      )}
      <div className="px-6 pb-6">{children}</div>
    </section>
  );
}

export function Chip({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "bad";
}) {
  const bg =
    tone === "good" ? "bg-sage-soft" : tone === "bad" ? "bg-clay-soft" : "bg-surface";
  const valueColor =
    tone === "good" ? "text-pine" : tone === "bad" ? "text-redwood" : "text-ink";
  return (
    <div
      className={`flex items-center gap-2 rounded-full border border-line px-3.5 py-1.5 ${bg}`}
    >
      <span className="font-mono text-[10px] tracking-wide text-ink-faint">
        {label}
      </span>
      <span className={`tnum font-mono text-[11px] font-medium ${valueColor}`}>
        {value}
      </span>
    </div>
  );
}

export function StatCard({
  label,
  value,
  sub,
  tone = "ink",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "ink" | "pine" | "clay";
}) {
  const color =
    tone === "pine" ? "text-pine" : tone === "clay" ? "text-clay" : "text-ink";
  return (
    <div className="rounded-2xl border border-line bg-surface p-5">
      <div className="font-mono text-[10.5px] tracking-wide text-ink-faint">
        {label}
      </div>
      <div
        className={`tnum mt-1.5 font-display text-[2rem] font-semibold leading-none ${color}`}
      >
        {value}
      </div>
      {sub && <div className="mt-2 text-[12px] text-ink-soft">{sub}</div>}
    </div>
  );
}

export function Pill({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "bad";
}) {
  const cls =
    tone === "good"
      ? "bg-sage-soft text-pine"
      : tone === "bad"
        ? "bg-clay-soft text-redwood"
        : "bg-sunk text-ink-soft";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 font-mono text-[10px] font-semibold tracking-wide ${cls}`}
    >
      {children}
    </span>
  );
}
