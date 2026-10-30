import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { scaleLinear, scaleBand } from "d3-scale";
import { area, line, curveMonotoneX, curveStepAfter } from "d3-shape";
import { compact } from "../lib/format";

export function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useLayoutEffect(() => {
    // medir antes de pintar: sin esto el primer dibujo usa el ancho por defecto y sale deformado
    if (ref.current) setW(Math.max(160, Math.floor(ref.current.getBoundingClientRect().width)));
  }, []);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((e) => setW(Math.max(160, Math.floor(e[0].contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export type Series = {
  key: string;
  label: string;
  color: string;
  values: (number | null)[];
  dash?: boolean;
  band?: { lo: (number | null)[]; hi: (number | null)[] };
  step?: boolean;
};

type LineProps = {
  x: number[];
  series: Series[];
  height?: number;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  xTicks?: number;
  yMin?: number;
  yMax?: number;
  marker?: { x: number; label: string };
  title?: string;
  ariaLabel?: string;
};

/** Líneas con bandas opcionales; una sola escala vertical; cruz y tooltip al pasar el cursor. */
export function LineChart({ x, series, height = 200, yFormat = compact, xFormat = (v) => String(v), xTicks = 6,
                            yMin, yMax, marker, ariaLabel }: LineProps) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const m = { top: 8, right: 12, bottom: 22, left: 48 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const all: number[] = [];
  for (const s of series) {
    for (const v of s.values) if (v !== null && Number.isFinite(v)) all.push(v);
    if (s.band) for (const v of [...s.band.lo, ...s.band.hi]) if (v !== null && Number.isFinite(v)) all.push(v);
  }
  const lo = yMin ?? Math.min(0, ...all);
  const hi = yMax ?? (all.length ? Math.max(...all) : 1);
  const xs = scaleLinear().domain([x[0], x[x.length - 1]]).range([0, iw]);
  const ys = scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).nice(4).range([ih, 0]);
  const curve = (s: Series) => (s.step ? curveStepAfter : curveMonotoneX);
  const path = (s: Series) =>
    line<number | null>().defined((v) => v !== null && Number.isFinite(v)).x((_, i) => xs(x[i])).y((v) => ys(v as number))
      .curve(curve(s))(s.values) ?? "";
  const bandPath = (s: Series) =>
    s.band
      ? area<number>().defined((i) => s.band!.lo[i] !== null && s.band!.hi[i] !== null).x((i) => xs(x[i]))
          .y0((i) => ys(s.band!.lo[i] as number)).y1((i) => ys(s.band!.hi[i] as number)).curve(curve(s))(x.map((_, i) => i)) ?? ""
      : "";
  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = (e.target as SVGRectElement).getBoundingClientRect();
    const xv = xs.invert(e.clientX - r.left);
    let best = 0;
    for (let i = 1; i < x.length; i++) if (Math.abs(x[i] - xv) < Math.abs(x[best] - xv)) best = i;
    setHover(best);
  };
  const ticks = xs.ticks(Math.min(xTicks, Math.max(2, Math.floor(iw / 70))));
  return (
    <div className="chart" ref={ref} role="img" aria-label={ariaLabel}>
      <svg width={width} height={height}>
        <g transform={`translate(${m.left},${m.top})`}>
          <g className="grid">
            {ys.ticks(4).map((t) => <line key={t} x1={0} x2={iw} y1={ys(t)} y2={ys(t)} />)}
          </g>
          <g className="axis">
            {ys.ticks(4).map((t) => (
              <text key={t} x={-6} y={ys(t)} dy="0.32em" textAnchor="end">{yFormat(t)}</text>
            ))}
            {ticks.map((t) => (
              <text key={t} x={xs(t)} y={ih + 15} textAnchor="middle">{xFormat(t)}</text>
            ))}
          </g>
          <line className="baseline" x1={0} x2={iw} y1={ih} y2={ih} />
          {marker && (
            <g>
              <line x1={xs(marker.x)} x2={xs(marker.x)} y1={0} y2={ih} stroke="var(--axis)" strokeDasharray="3 3" />
              <text x={xs(marker.x) + 4} y={10} fontSize={11} fill="var(--text-muted)">{marker.label}</text>
            </g>
          )}
          {series.map((s) => s.band && <path key={`b${s.key}`} d={bandPath(s)} fill={s.color} opacity={0.18} />)}
          {series.map((s) => (
            <path key={s.key} d={path(s)} fill="none" stroke={s.color} strokeWidth={2}
                  strokeDasharray={s.dash ? "4 3" : undefined} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {hover !== null && (
            <g pointerEvents="none">
              <line x1={xs(x[hover])} x2={xs(x[hover])} y1={0} y2={ih} stroke="var(--text-muted)" strokeWidth={1} />
              {series.map((s) => {
                const v = s.values[hover];
                return v !== null && Number.isFinite(v) ? (
                  <circle key={s.key} cx={xs(x[hover])} cy={ys(v)} r={4} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} />
                ) : null;
              })}
            </g>
          )}
          <rect width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        </g>
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: m.left + xs(x[hover]), top: m.top + 4 }}>
          <div className="t">{xFormat(x[hover])}</div>
          {series.map((s) => (
            <div className="r" key={s.key}>
              <i className="sw line" style={{ background: s.color }} />
              <span>{s.label}</span>
              <b style={{ marginLeft: "auto", paddingLeft: 8 }}>{fmtV(s.values[hover], yFormat)}</b>
              {s.band && (
                <span className="muted">
                  ({fmtV(s.band.lo[hover], yFormat)}–{fmtV(s.band.hi[hover], yFormat)})
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function fmtV(v: number | null | undefined, f: (v: number) => string) {
  return v === null || v === undefined || !Number.isFinite(v) ? "–" : f(v);
}

/** Columnas apiladas (p. ej. eventos por año y tipo), 2 px de separación entre segmentos y entre columnas. */
export function StackedColumns({ labels, stacks, height = 160, yFormat = compact, ariaLabel }: {
  labels: string[]; stacks: { key: string; label: string; color: string; values: number[] }[];
  height?: number; yFormat?: (v: number) => string; ariaLabel?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const m = { top: 8, right: 8, bottom: 22, left: 40 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const totals = labels.map((_, i) => stacks.reduce((a, s) => a + (s.values[i] ?? 0), 0));
  const xs = scaleBand<number>().domain(labels.map((_, i) => i)).range([0, iw]).paddingInner(0.15);
  const ys = scaleLinear().domain([0, Math.max(1, ...totals)]).nice(4).range([ih, 0]);
  const every = Math.ceil(labels.length / Math.max(2, Math.floor(iw / 44)));
  return (
    <div className="chart" ref={ref} role="img" aria-label={ariaLabel}>
      <svg width={width} height={height}>
        <g transform={`translate(${m.left},${m.top})`}>
          <g className="grid">{ys.ticks(4).map((t) => <line key={t} x1={0} x2={iw} y1={ys(t)} y2={ys(t)} />)}</g>
          <g className="axis">
            {ys.ticks(4).map((t) => <text key={t} x={-6} y={ys(t)} dy="0.32em" textAnchor="end">{yFormat(t)}</text>)}
            {labels.map((l, i) => (i % every === 0 ? (
              <text key={l} x={(xs(i) ?? 0) + xs.bandwidth() / 2} y={ih + 15} textAnchor="middle">{l}</text>
            ) : null))}
          </g>
          {labels.map((_, i) => {
            let acc = 0;
            return (
              <g key={i} opacity={hover === null || hover === i ? 1 : 0.55}>
                {stacks.map((s) => {
                  const v = s.values[i] ?? 0;
                  if (v <= 0) return null;
                  const y0 = ys(acc), y1 = ys(acc + v);
                  acc += v;
                  const h = Math.max(0, y0 - y1 - (acc > v ? 2 : 0));
                  return <rect key={s.key} x={xs(i)} y={y1} width={xs.bandwidth()} height={h} fill={s.color} rx={1.5} />;
                })}
                <rect x={xs(i)} y={0} width={xs.bandwidth()} height={ih} fill="transparent"
                      onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)} />
              </g>
            );
          })}
          <line className="baseline" x1={0} x2={iw} y1={ih} y2={ih} />
        </g>
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: m.left + (xs(hover) ?? 0) + xs.bandwidth() / 2, top: m.top + ys(totals[hover]) }}>
          <div className="t">{labels[hover]} · total {yFormat(totals[hover])}</div>
          {stacks.map((s) => (
            <div className="r" key={s.key}><i className="sw" style={{ background: s.color }} />{s.label}
              <b style={{ marginLeft: "auto", paddingLeft: 8 }}>{yFormat(s.values[hover] ?? 0)}</b></div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Barras horizontales con el valor al final (tinta de texto, no el color de la serie). */
export function HBars({ rows, max = 1, format, stacked }: {
  rows: { label: ReactNode; value: number; color: string; note?: string }[]; max?: number; format: (v: number) => string;
  stacked?: boolean;
}) {
  if (stacked) {
    // etiqueta encima de la barra: para columnas estrechas con etiquetas largas
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {rows.map((r, i) => (
          <div key={i}>
            <div className="small secondary">{r.label}</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 10, alignItems: "center" }}>
              <div style={{ height: 10, background: "var(--surface-2)", borderRadius: 4 }}>
                <div style={{ width: `${Math.max(0, Math.min(1, r.value / (max || 1))) * 100}%`, height: "100%",
                              background: r.color, borderRadius: 4 }} />
              </div>
              <span className="small tabnum">{format(r.value)}</span>
            </div>
          </div>
        ))}
      </div>
    );
  }
  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto", gap: "6px 10px", alignItems: "center" }}>
      {rows.map((r, i) => (
        <HBarRow key={i} {...r} max={max} format={format} />
      ))}
    </div>
  );
}

function HBarRow({ label, value, color, note, max, format }: {
  label: ReactNode; value: number; color: string; note?: string; max: number; format: (v: number) => string;
}) {
  const w = Math.max(0, Math.min(1, value / (max || 1)));
  return (
    <>
      <span className="small secondary">{label}</span>
      <div style={{ height: 12, background: "var(--surface-2)", borderRadius: 4 }} title={note}>
        <div style={{ width: `${w * 100}%`, height: "100%", background: color, borderRadius: 4, minWidth: value > 0 ? 3 : 0 }} />
      </div>
      <span className="small tabnum" style={{ textAlign: "right", minWidth: 52 }}>{format(value)}</span>
    </>
  );
}

/** Barra de proporciones (suman 1) con separación de 2 px y leyenda debajo. */
export function ShareBar({ parts }: { parts: { label: string; value: number; color: string }[] }) {
  const total = parts.reduce((a, p) => a + p.value, 0) || 1;
  const shown = useMemo(() => parts.filter((p) => p.value / total > 0.001), [parts, total]);
  return (
    <div>
      <div style={{ display: "flex", gap: 2, height: 14, borderRadius: 4, overflow: "hidden" }}>
        {shown.map((p) => (
          <div key={p.label} title={`${p.label}: ${Math.round((p.value / total) * 100)} %`}
               style={{ flex: `${p.value / total} 1 0`, background: p.color, minWidth: 3 }} />
        ))}
      </div>
      <div className="legend" style={{ marginTop: 6 }}>
        {parts.map((p) => (
          <span key={p.label} className="key"><i className="sw" style={{ background: p.color }} />{p.label}{" "}
            <b className="tabnum">{Math.round((p.value / total) * 100)} %</b></span>
        ))}
      </div>
    </div>
  );
}
