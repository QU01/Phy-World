import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { geoContains, geoNaturalEarth1, geoPath } from "d3-geo";
import { COUNTRIES, LAND } from "../lib/geo";
import { cssVar, readRamp } from "../lib/colors";
import { useWidth } from "./charts";

export type CellLayer = {
  lon: number[];
  lat: number[];
  gid: number[];
  cls: Int8Array;            // clase de color por celda, -1 = no se pinta
};

type View = { k: number; x: number; y: number };

/** gid de PRIO-GRID 2.0 de un punto: (fila - 1) * 720 + columna, fila 1 en -90°, columna 1 en -180°. */
export function gidAt(lon: number, lat: number): number {
  const row = Math.floor((lat + 90) / 0.5) + 1;
  const col = Math.floor((lon + 180) / 0.5) + 1;
  return (row - 1) * 720 + col;
}

export function CellMap({ layer, selectedGid, onSelect, tooltip, focus, theme, height }: {
  layer: CellLayer;
  selectedGid: number | null;
  onSelect: (gid: number | null) => void;
  tooltip: (gid: number, index: number | undefined) => ReactNode;
  focus: [number, number, number, number] | null;
  theme: string;
  height?: number;
}) {
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const h = height ?? Math.round(Math.min(620, Math.max(320, width * 0.52)));
  const canvas = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<View>({ k: 1, x: 0, y: 0 });
  const [hover, setHover] = useState<{ gid: number; px: number; py: number } | null>(null);
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: boolean } | null>(null);

  const projection = useMemo(() => geoNaturalEarth1().fitExtent([[6, 6], [width - 6, h - 6]], { type: "Sphere" }),
                             [width, h]);
  const index = useMemo(() => {
    const m = new Map<number, number>();
    layer.gid.forEach((g, i) => m.set(g, i));
    return m;
  }, [layer]);

  // geometría proyectada: Path2D del mapa base y las 4 esquinas de cada celda
  const geom = useMemo(() => {
    const p = geoPath(projection);
    const land = new Path2D(p(LAND) ?? "");
    const borders = new Path2D(COUNTRIES.features.map((f) => p(f) ?? "").join(""));
    const sphere = new Path2D(p({ type: "Sphere" }) ?? "");
    const n = layer.gid.length;
    const quad = new Float32Array(n * 8);
    for (let i = 0; i < n; i++) {
      const lo = layer.lon[i], la = layer.lat[i];
      const c = [[lo - 0.25, la - 0.25], [lo + 0.25, la - 0.25], [lo + 0.25, la + 0.25], [lo - 0.25, la + 0.25]];
      for (let j = 0; j < 4; j++) {
        const q = projection(c[j] as [number, number]) ?? [NaN, NaN];
        quad[i * 8 + j * 2] = q[0];
        quad[i * 8 + j * 2 + 1] = q[1];
      }
    }
    return { land, borders, sphere, quad };
  }, [projection, layer]);

  // dibujo
  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(h * dpr);
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = cssVar("--ocean");
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.setTransform(dpr * view.k, 0, 0, dpr * view.k, dpr * view.x, dpr * view.y);
    ctx.fillStyle = cssVar("--surface-1");
    ctx.fill(geom.sphere);
    ctx.fillStyle = cssVar("--land");
    ctx.fill(geom.land);
    const ramp = readRamp();
    const { quad } = geom;
    for (let c = 0; c < ramp.length; c++) {
      ctx.beginPath();
      for (let i = 0; i < layer.cls.length; i++) {
        if (layer.cls[i] !== c) continue;
        const o = i * 8;
        if (!Number.isFinite(quad[o])) continue;
        ctx.moveTo(quad[o], quad[o + 1]);
        ctx.lineTo(quad[o + 2], quad[o + 3]);
        ctx.lineTo(quad[o + 4], quad[o + 5]);
        ctx.lineTo(quad[o + 6], quad[o + 7]);
        ctx.closePath();
      }
      ctx.fillStyle = ramp[c];
      ctx.fill();
    }
    ctx.lineWidth = 0.6 / view.k;
    ctx.strokeStyle = cssVar("--land-stroke");
    ctx.stroke(geom.borders);
    const ring = (gid: number, color: string, w: number) => {
      const row = Math.floor((gid - 1) / 720) + 1, col = ((gid - 1) % 720) + 1;
      const lon = -180 + (col - 0.5) * 0.5, lat = -90 + (row - 0.5) * 0.5;
      const pts = [[lon - 0.25, lat - 0.25], [lon + 0.25, lat - 0.25], [lon + 0.25, lat + 0.25], [lon - 0.25, lat + 0.25]]
        .map((c) => projection(c as [number, number]));
      if (pts.some((p) => !p)) return;
      ctx.beginPath();
      pts.forEach((p, j) => (j ? ctx.lineTo(p![0], p![1]) : ctx.moveTo(p![0], p![1])));
      ctx.closePath();
      ctx.lineWidth = (w + 2) / view.k;
      ctx.strokeStyle = cssVar("--surface-1");
      ctx.stroke();
      ctx.lineWidth = w / view.k;
      ctx.strokeStyle = color;
      ctx.stroke();
    };
    if (hover) ring(hover.gid, cssVar("--text-secondary"), 1.5);
    if (selectedGid !== null) ring(selectedGid, cssVar("--text-primary"), 2.5);
  }, [geom, layer, view, width, h, selectedGid, hover, theme]);

  // zoom a una caja (país)
  useEffect(() => {
    if (!focus) {
      setView({ k: 1, x: 0, y: 0 });
      return;
    }
    const [x0, y0, x1, y1] = focus;
    const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1]]
      .map((c) => projection(c as [number, number])).filter(Boolean) as [number, number][];
    const px0 = Math.min(...pts.map((p) => p[0])), px1 = Math.max(...pts.map((p) => p[0]));
    const py0 = Math.min(...pts.map((p) => p[1])), py1 = Math.max(...pts.map((p) => p[1]));
    const k = Math.min(40, Math.max(1, 0.85 * Math.min(width / (px1 - px0 || 1), h / (py1 - py0 || 1))));
    setView({ k, x: width / 2 - k * (px0 + px1) / 2, y: h / 2 - k * (py0 + py1) / 2 });
  }, [focus, projection, width, h]);

  const toBase = (px: number, py: number) => [(px - view.x) / view.k, (py - view.y) / view.k] as [number, number];
  const gidUnder = (px: number, py: number) => {
    const ll = projection.invert?.(toBase(px, py));
    if (!ll || !Number.isFinite(ll[0])) return null;
    const g = gidAt(ll[0], ll[1]);
    return index.has(g) || geoContains(LAND, ll) ? g : null;
  };
  const zoomAt = (px: number, py: number, f: number) =>
    setView((v) => {
      const k = Math.min(40, Math.max(1, v.k * f));
      const r = k / v.k;
      return { k, x: px - (px - v.x) * r, y: py - (py - v.y) * r };
    });

  const local = (e: { clientX: number; clientY: number }) => {
    const r = canvas.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as [number, number];
  };

  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [px, py] = local(e);
      zoomAt(px, py, Math.exp(-e.deltaY * 0.0015));
    };
    cv.addEventListener("wheel", onWheel, { passive: false });
    return () => cv.removeEventListener("wheel", onWheel);
  });

  const hovered = hover ? index.get(hover.gid) : undefined;
  return (
    <div className="mapwrap" ref={wrapRef}>
      <canvas
        ref={canvas}
        style={{ height: h }}
        className={drag.current?.moved ? "dragging" : ""}
        aria-label="Mapa de celdas PRIO-GRID coloreadas por riesgo; haga clic en una celda para ver su ficha"
        onPointerDown={(e) => {
          (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
          drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false };
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (d) {
            const dx = e.clientX - d.x, dy = e.clientY - d.y;
            if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
            if (d.moved) {
              setView((v) => ({ ...v, x: d.vx + dx, y: d.vy + dy }));
              setHover(null);
              return;
            }
          }
          const [px, py] = local(e);
          const g = gidUnder(px, py);
          setHover(g === null ? null : { gid: g, px, py });
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (d && !d.moved) {
            const [px, py] = local(e);
            onSelect(gidUnder(px, py));
          }
        }}
        onPointerLeave={() => setHover(null)}
      />
      <div className="maptools">
        <button aria-label="Acercar" onClick={() => zoomAt(width / 2, h / 2, 1.6)}>+</button>
        <button aria-label="Alejar" onClick={() => zoomAt(width / 2, h / 2, 1 / 1.6)}>−</button>
        <button aria-label="Ver el mundo entero" title="Mundo" onClick={() => setView({ k: 1, x: 0, y: 0 })}>⌂</button>
      </div>
      {hover && (
        <div className="tooltip" style={{ left: hover.px, top: hover.py }}>{tooltip(hover.gid, hovered)}</div>
      )}
    </div>
  );
}
