// Minimal, dependency-free scale + tick helpers for the SVG charts.

export function niceNum(range: number, round: boolean): number {
  if (range === 0) return 1;
  const exp = Math.floor(Math.log10(range));
  const frac = range / 10 ** exp;
  let nice: number;
  if (round) {
    if (frac < 1.5) nice = 1;
    else if (frac < 3) nice = 2;
    else if (frac < 7) nice = 5;
    else nice = 10;
  } else {
    if (frac <= 1) nice = 1;
    else if (frac <= 2) nice = 2;
    else if (frac <= 5) nice = 5;
    else nice = 10;
  }
  return nice * 10 ** exp;
}

export interface Ticks {
  lo: number;
  hi: number;
  step: number;
  values: number[];
}

export function niceTicks(min: number, max: number, count = 5): Ticks {
  if (min === max) {
    const pad = Math.abs(min) || 1;
    min -= pad;
    max += pad;
  }
  const range = niceNum(max - min, false);
  const step = niceNum(range / Math.max(1, count - 1), true);
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const values: number[] = [];
  for (let v = lo; v <= hi + step * 0.5; v += step) {
    values.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  }
  return { lo, hi, step, values };
}

// Map a value in [d0,d1] to a pixel in [p0,p1].
export function linear(value: number, d0: number, d1: number, p0: number, p1: number): number {
  if (d1 === d0) return (p0 + p1) / 2;
  return p0 + ((value - d0) / (d1 - d0)) * (p1 - p0);
}

export function extent(values: number[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return [0, 1];
  return [lo, hi];
}
