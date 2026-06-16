// Warm diverging color scale (redwood ← parchment → forest) for heatmaps/bars.

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function mix(c0: string, c1: string, t: number): string {
  const [r0, g0, b0] = hexToRgb(c0);
  const [r1, g1, b1] = hexToRgb(c1);
  return `rgb(${lerp(r0, r1, t)}, ${lerp(g0, g1, t)}, ${lerp(b0, b1, t)})`;
}

const NEUTRAL = "#efe7d6";
const POS = "#2f5240"; // forest
const NEG = "#9c4329"; // redwood

// value normalised to [-1, 1] -> background color.
export function diverge(norm: number): string {
  const t = Math.max(-1, Math.min(1, norm));
  if (t >= 0) return mix(NEUTRAL, POS, Math.sqrt(t));
  return mix(NEUTRAL, NEG, Math.sqrt(-t));
}

// Whether to use light text on a given diverging intensity.
export function lightText(norm: number): boolean {
  return Math.abs(norm) > 0.55;
}
