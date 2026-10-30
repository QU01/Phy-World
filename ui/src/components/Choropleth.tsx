import { useMemo, useState } from "react";
import { geoNaturalEarth1, geoPath } from "d3-geo";
import { COUNTRIES, countryName, iso3OfFeature } from "../lib/geo";
import { classOf, RAMP_VARS } from "../lib/colors";
import { useWidth } from "./charts";

/** Países coloreados por ``values[iso3]`` en 7 clases (``breaks``) de la rampa secuencial. */
export function Choropleth({ values, breaks, labels, format, title, selected, onSelect }: {
  values: Record<string, number>;
  breaks: number[];
  labels: string[];
  format: (v: number) => string;
  title: string;
  selected?: string | null;
  onSelect?: (iso3: string | null) => void;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const h = Math.round(width * 0.5);
  const [hover, setHover] = useState<{ iso: string; x: number; y: number } | null>(null);
  const path = useMemo(() => geoPath(geoNaturalEarth1().fitExtent([[4, 4], [width - 4, h - 4]], { type: "Sphere" })),
                       [width, h]);
  const shapes = useMemo(() => COUNTRIES.features.map((f) => ({ iso: iso3OfFeature(f), d: path(f) ?? "" })), [path]);
  return (
    <div className="mapwrap" ref={ref} style={{ position: "relative" }}>
      <svg width={width} height={h} role="img" aria-label={title}>
        <path d={path({ type: "Sphere" }) ?? ""} fill="var(--surface-1)" />
        {shapes.map(({ iso, d }, i) => {
          const v = iso ? values[iso] : undefined;
          const c = v === undefined ? -2 : classOf(v, breaks);
          const fill = c === -2 ? "var(--land)" : c < 0 ? "var(--surface-2)" : `var(${RAMP_VARS[c]})`;
          return (
            <path key={i} d={d} fill={fill} stroke={iso && iso === selected ? "var(--text-primary)" : "var(--land-stroke)"}
                  strokeWidth={iso && iso === selected ? 2 : 0.5} style={{ cursor: onSelect ? "pointer" : undefined }}
                  onPointerMove={(e) => {
                    const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                    if (iso) setHover({ iso, x: e.clientX - r.left, y: e.clientY - r.top });
                  }}
                  onPointerLeave={() => setHover(null)}
                  onClick={() => onSelect?.(iso ?? null)} />
          );
        })}
      </svg>
      <div className="maplegend">
        <div>{title}</div>
        <div className="ramp">
          <span><i style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }} />&lt; {labels[0]}</span>
          {labels.map((l, i) => (
            <span key={l}><i style={{ background: `var(${RAMP_VARS[i]})` }} />{l}</span>
          ))}
        </div>
      </div>
      {hover && (
        <div className="tooltip" style={{ left: hover.x, top: hover.y }}>
          <div className="t">{countryName(hover.iso)}</div>
          <div className="r">{values[hover.iso] === undefined ? "sin datos en el modelo" : format(values[hover.iso])}</div>
        </div>
      )}
    </div>
  );
}
