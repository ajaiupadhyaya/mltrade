import { useEffect, useRef, useState } from "react";

// Measure a container's pixel width so SVG charts can render crisp, 1:1 with
// the DOM (required for accurate hover/crosshair hit-testing).
export function useMeasure<T extends HTMLElement>(): [
  React.RefObject<T | null>,
  number,
] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(720);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setWidth(w);
    });
    observer.observe(el);
    setWidth(el.getBoundingClientRect().width || 720);
    return () => observer.disconnect();
  }, []);

  return [ref, width];
}
