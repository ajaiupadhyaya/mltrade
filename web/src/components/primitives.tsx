import type { ReactNode } from "react";

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = "",
  pad = true,
}: {
  title?: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section
      className={`rounded-xl border border-line bg-surface shadow-[0_1px_2px_rgba(60,45,25,0.04)] ${className}`}
    >
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 border-b border-line-soft px-5 pb-3 pt-4">
          <div>
            {title && (
              <h2 className="font-display text-[1rem] font-semibold leading-tight text-ink">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-1 font-mono text-[10.5px] leading-snug text-ink-faint">
                {subtitle}
              </p>
            )}
          </div>
          {right && <div className="shrink-0 pt-0.5">{right}</div>}
        </header>
      )}
      <div className={pad ? "p-5" : ""}>{children}</div>
    </section>
  );
}

export function Pill({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "bad" | "warn";
}) {
  const cls =
    tone === "good"
      ? "bg-sage-soft text-pine"
      : tone === "bad"
        ? "bg-clay-soft text-redwood"
        : tone === "warn"
          ? "bg-ochre-soft text-redwood"
          : "bg-sunk text-ink-soft";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 font-mono text-[9.5px] font-semibold tracking-wide ${cls}`}
    >
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  sub,
  tone = "ink",
  big = false,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "ink" | "pine" | "clay" | "good" | "bad";
  big?: boolean;
}) {
  const color =
    tone === "pine"
      ? "text-pine"
      : tone === "clay"
        ? "text-clay"
        : tone === "good"
          ? "text-forest"
          : tone === "bad"
            ? "text-redwood"
            : "text-ink";
  return (
    <div className="rounded-lg border border-line-soft bg-raise px-4 py-3.5">
      <div className="font-mono text-[9.5px] uppercase tracking-wide text-ink-faint">
        {label}
      </div>
      <div
        className={`tnum mt-1 font-display font-semibold leading-none ${color} ${
          big ? "text-[2.1rem]" : "text-[1.5rem]"
        }`}
      >
        {value}
      </div>
      {sub && <div className="mt-1.5 text-[11px] leading-tight text-ink-soft">{sub}</div>}
    </div>
  );
}

export function KeyVal({
  k,
  v,
  tone = "ink",
  mono = true,
}: {
  k: string;
  v: string;
  tone?: "ink" | "good" | "bad" | "soft";
  mono?: boolean;
}) {
  const color =
    tone === "good"
      ? "text-forest"
      : tone === "bad"
        ? "text-redwood"
        : tone === "soft"
          ? "text-ink-soft"
          : "text-ink";
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="text-[12px] text-ink-soft">{k}</span>
      <span
        className={`tnum text-[12.5px] font-medium ${color} ${mono ? "font-mono" : ""}`}
      >
        {v}
      </span>
    </div>
  );
}

export function Verdict({
  ok,
  children,
}: {
  ok: boolean;
  children: ReactNode;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 font-mono text-[10px] font-semibold ${
        ok ? "bg-sage-soft text-pine" : "bg-ochre-soft text-redwood"
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-forest" : "bg-clay"}`} />
      {children}
    </span>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      {items.map((it) => (
        <span key={it.label} className="flex items-center gap-1.5">
          <span
            className="inline-block h-0.5 w-4 rounded"
            style={{
              background: it.dashed ? "transparent" : it.color,
              borderTop: it.dashed ? `1.5px dashed ${it.color}` : undefined,
            }}
          />
          <span className="text-[11px] text-ink-soft">{it.label}</span>
        </span>
      ))}
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="text-[12px] leading-relaxed text-ink-soft">{children}</p>
  );
}
