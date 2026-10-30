import { useMemo, useState } from "react";
import { getJSON, SNAPSHOT, type ConflictCell, type ConflictMeta, type ConflictPayload, type ConflictValidation, type Outcome } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, ErrorNotice, Field, Legend, Loading, Seg, Stat } from "../components/common";
import { CellMap, type CellLayer } from "../components/CellMap";
import { HBars, LineChart, ShareBar, StackedColumns } from "../components/charts";
import { classOf, magnitudeBreaks, PROB_BREAKS, PROB_LABELS, RAMP_VARS, TYPE_COLOR, TYPE_LABEL, TYPE_ORDER, TYPE_SHORT } from "../lib/colors";
import { countryName, iso3AtPoint } from "../lib/geo";
import { compact, latlon, monthName, num, pct, pctAxis } from "../lib/format";

type Metric = "window" | "month" | "events" | "deaths";
const METRICS: { value: Metric; label: string; title: string }[] = [
  { value: "window", label: "P en 12 meses", title: "Probabilidad de al menos un evento en los 12 meses" },
  { value: "month", label: "P por mes", title: "Probabilidad de al menos un evento en el mes elegido" },
  { value: "events", label: "Eventos esperados", title: "Número esperado de eventos en los 12 meses" },
  { value: "deaths", label: "Muertes estatales", title: "Muertes esperadas por violencia estatal en los 12 meses" },
];

export function ConflictView({ theme }: { theme: string }) {
  const meta = useApi((s) => getJSON<ConflictMeta>("/api/conflict/meta", undefined, s), []);
  const [origin, setOrigin] = useState<string | undefined>(undefined);
  const effOrigin = origin ?? meta.data?.last_month;
  const fc = useApi(effOrigin ? (s) => getJSON<ConflictPayload>("/api/conflict/forecast", { origin: effOrigin }, s) : null,
                    [effOrigin]);
  const [outcome, setOutcome] = useState<Outcome>("any");
  const [metric, setMetric] = useState<Metric>("window");
  const [month, setMonth] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [focusGw, setFocusGw] = useState<number | null>(null);

  // nombres en español: el país de Natural Earth que contiene la celda central de cada país de PRIO-GRID;
  // si no hay (Kosovo, territorios), el nombre que usa UCDP
  const names = useMemo(() => {
    const out: Record<string, string> = {};
    if (!meta.data) return out;
    for (const [gw, ged] of Object.entries(meta.data.country_names)) out[gw] = ged;
    for (const [gw, [lon, lat]] of Object.entries(meta.data.country_point)) {
      const iso = iso3AtPoint(lon, lat);
      if (iso) out[gw] = countryName(iso);
    }
    return out;
  }, [meta.data]);
  if (meta.error) return <ErrorNotice error={meta.error} what="el pronóstico por celdas" />;
  if (!meta.data) return <Loading what="Cargando la malla PRIO-GRID y los eventos de UCDP" />;
  const P = fc.data;
  return (
    <>
      <div className="viewhead">
        <h2>Conflicto por celdas</h2>
        <p>
          Probabilidad de violencia organizada en cada celda de 0,5° (~55 km) para los 12 meses siguientes al último
          dato, con un umbral logístico y un proceso de Hawkes espacio-temporal promediados (F11). Datos: UCDP GED {" "}
          {meta.data.versions[0]} hasta {monthName(meta.data.last_final_month)} y eventos candidatos hasta{" "}
          {monthName(meta.data.last_month)}.
        </p>
      </div>
      <div className="controls card" style={{ padding: "10px 14px" }}>
        <Field label="Origen (último mes con datos)">
          <select value={effOrigin} onChange={(e) => { setOrigin(e.target.value); setSelected(null); }}>
            {originOptions(meta.data).map((o) => (
              <option key={o} value={o}>{monthName(o)}{meta.data!.saved_origins.includes(o) ? "" : " (entrena ~6 min)"}</option>
            ))}
          </select>
        </Field>
        <Field label="Tipo de violencia">
          <Seg value={outcome} onChange={setOutcome} label="Tipo de violencia"
               options={(["any", ...TYPE_ORDER] as Outcome[]).map((o) => ({ value: o, label: TYPE_SHORT[o], title: TYPE_LABEL[o] }))} />
        </Field>
        <Field label="Qué pintar">
          <Seg value={metric} onChange={setMetric} label="Métrica" options={METRICS} />
        </Field>
        {metric === "month" && P && (
          <Field label={`Mes: ${monthName(P.months[month])}`}>
            <input type="range" min={0} max={P.months.length - 1} value={month} onChange={(e) => setMonth(+e.target.value)}
                   aria-label="Mes del horizonte" />
          </Field>
        )}
        <Field label="Ir a un país">
          <select value={focusGw ?? ""} onChange={(e) => setFocusGw(e.target.value ? +e.target.value : null)}>
            <option value="">Mundo</option>
            {P?.countries.slice(0, 80).map((c) => (
              <option key={c.gw} value={c.gw}>{names[String(c.gw)] ?? `GW ${c.gw}`}</option>
            ))}
          </select>
        </Field>
      </div>
      {fc.error && <ErrorNotice error={fc.error} what="el pronóstico por celdas" />}
      {!P && !fc.error && (
        <Loading what="Calculando el pronóstico de las 64 818 celdas"
                 slow="La primera vez para un origen entrena el modelo (unos 6 minutos); después tarda 1-2 minutos y queda guardado." />
      )}
      {P && (
        <div className="conflict-layout">
          <div style={{ display: "flex", flexDirection: "column", gap: 16, minWidth: 0 }}>
            <MapBlock P={P} outcome={outcome} metric={metric} month={month} selected={selected} onSelect={setSelected}
                      focus={focusGw !== null ? meta.data.country_bbox[String(focusGw)] ?? null : null}
                      theme={theme} names={names} />
            <CountryTable P={P} names={names} onFocus={(gw) => setFocusGw(gw)} focus={focusGw} />
            <ValidationCard P={P} />
          </div>
          <div className="side">
            {selected === null ? (
              <Summary P={P} names={names} onPick={(g) => setSelected(g)} />
            ) : (
              <CellPanel gid={selected} P={P} origin={P.origin} names={names} onClose={() => setSelected(null)} />
            )}
          </div>
        </div>
      )}
    </>
  );
}

function originOptions(m: ConflictMeta): string[] {
  if (SNAPSHOT) return [m.last_month];
  // los orígenes guardados y diciembre de cada año desde 2015, más el último mes
  const out = new Set<string>(m.saved_origins);
  const last = +m.last_month.slice(0, 4);
  for (let y = 2015; y < last; y++) out.add(`${y}-12`);
  out.add(m.last_month);
  return [...out].filter((o) => o >= m.first_origin && o <= m.last_month).sort().reverse();
}

function MapBlock({ P, outcome, metric, month, selected, onSelect, focus, theme, names }: {
  P: ConflictPayload; outcome: Outcome; metric: Metric; month: number; selected: number | null;
  onSelect: (g: number | null) => void; focus: [number, number, number, number] | null; theme: string;
  names: Record<string, string>;
}) {
  const c = P.cells;
  const { values, breaks, labels, fmt, title } = useMemo(() => {
    const n = c.gid.length;
    let values: number[];
    if (metric === "window") values = c[`p_window_${outcome}`];
    else if (metric === "month") values = c[`p_month_${outcome}`].map((r) => r[month]);
    else if (metric === "events")
      values = Array.from({ length: n }, (_, i) =>
        outcome === "any" ? c.events_expected_sb[i] + c.events_expected_ns[i] + c.events_expected_os[i]
                          : c[`events_expected_${outcome}`][i]);
    else values = c.deaths_sb_expected;
    if (metric === "window" || metric === "month") {
      return { values, breaks: PROB_BREAKS, labels: PROB_LABELS, fmt: (v: number) => pct(v),
               title: `${metric === "window" ? "P(al menos un evento en 12 meses)" : `P(evento en ${monthName(P.months[month])})`}, ${TYPE_LABEL[outcome].toLowerCase()}` };
    }
    const b = magnitudeBreaks(values, metric === "events" ? 0.1 : 1);
    return { values, breaks: b, labels: b.map((x) => compact(x)), fmt: (v: number) => num(v, v < 10 ? 1 : 0),
             title: metric === "events" ? `Eventos esperados en 12 meses, ${TYPE_LABEL[outcome].toLowerCase()}` : "Muertes estatales esperadas en 12 meses" };
  }, [c, outcome, metric, month, P.months]);
  const layer: CellLayer = useMemo(() => {
    const cls = new Int8Array(c.gid.length);
    values.forEach((v, i) => (cls[i] = classOf(v, breaks)));
    return { lon: c.lon, lat: c.lat, gid: c.gid, cls };
  }, [c, values, breaks]);
  return (
    <div style={{ position: "relative" }}>
      <CellMap layer={layer} selectedGid={selected} onSelect={onSelect} focus={focus} theme={theme}
               tooltip={(gid, i) => (
                 <>
                   <div className="t">{i !== undefined ? `${names[String(c.country_gw[i])] ?? "GW " + c.country_gw[i]} · ` : ""}celda {gid}</div>
                   {i === undefined ? (
                     <div className="r">P(algún evento en 12 meses) &lt; 1 %</div>
                   ) : (
                     <>
                       <div className="r">{title.split(",")[0]}: <b style={{ marginLeft: "auto" }}>{fmt(values[i])}</b></div>
                       <div className="r muted">Eventos en los últimos 12 meses: {c.events_last12[i]}</div>
                     </>
                   )}
                 </>
               )} />
      <div className="maplegend">
        <div>{title}</div>
        <div className="ramp">
          <span><i style={{ background: "var(--land)", border: "1px solid var(--border)" }} />&lt; {labels[0]}</span>
          {labels.map((l, i) => (
            <span key={i}><i style={{ background: `var(${RAMP_VARS[i]})` }} />{l}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function Summary({ P, names, onPick }: { P: ConflictPayload; names: Record<string, string>; onPick: (gid: number) => void }) {
  const e = P.summary.expected_events_window;
  const c = P.cells;
  return (
    <>
      <Card title={`${monthName(P.months[0])} – ${monthName(P.months[P.months.length - 1])}`} sub="Todo el mundo, esperado por el modelo">
        <div className="stats">
          {TYPE_ORDER.map((t) => <Stat key={t} value={compact(e[t])} label={`eventos · ${TYPE_SHORT[t].toLowerCase()}`} />)}
          <Stat value={compact(P.summary.expected_deaths_sb_window)} label="muertes estatales" />
        </div>
      </Card>
      <Card title="Celdas de mayor riesgo" sub="P(al menos un evento en 12 meses); clic para ver la ficha">
        <table>
          <tbody>
            {c.gid.slice(0, 12).map((g, i) => (
              <tr key={g} className="clickable" onClick={() => onPick(g)}>
                <td>{names[String(c.country_gw[i])] ?? `GW ${c.country_gw[i]}`}<div className="small muted">{latlon(c.lat[i], c.lon[i])}</div></td>
                <td className="num">{pct(c.p_window_any[i])}<div className="small muted">{num(c.deaths_sb_expected[i], 0)} muertes</div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <Card title="Cómo leerlo">
        <ul className="caveats" style={{ marginTop: 0 }}>
          <li>Haga clic en una celda para ver su historia, su pronóstico mes a mes y de dónde viene su riesgo.</li>
          <li>Sólo se pintan las celdas con P ≥ 1 % en 12 meses; el resto queda en gris.</li>
          {P.caveats.slice(0, 3).map((x) => <li key={x}>{x}</li>)}
        </ul>
      </Card>
    </>
  );
}

function CellPanel({ gid, P, origin, names, onClose }: { gid: number; P: ConflictPayload; origin: string;
  names: Record<string, string>; onClose: () => void }) {
  const detail = useApi((s) => getJSON<ConflictCell>("/api/conflict/cell", { gid, origin }, s), [gid, origin]);
  const c = P.cells;
  const i = useMemo(() => c.gid.indexOf(gid), [c, gid]);
  const d = detail.data;
  const hist = useMemo(() => {
    if (!d) return null;
    const years: string[] = [];
    const by: Record<string, number[]> = { sb: [], ns: [], os: [] };
    d.months.forEach((m, k) => {
      const y = m.slice(0, 4);
      if (years[years.length - 1] !== y) {
        years.push(y);
        for (const t of TYPE_ORDER) by[t].push(0);
      }
      for (const t of TYPE_ORDER) by[t][by[t].length - 1] += d.events[t][k];
    });
    return { years, by };
  }, [d]);
  const months = P.months.map((_, k) => k);
  return (
    <Card title={d ? names[String(d.country_gw)] ?? d.country : `Celda ${gid}`} sub={d ? `${latlon(d.lat, d.lon)} · celda PRIO-GRID ${gid}` : undefined}
          actions={<button className="btn small" onClick={onClose} aria-label="Cerrar ficha">✕</button>}>
      {detail.error && <ErrorNotice error={detail.error} />}
      {i < 0 ? (
        <p className="secondary">P(al menos un evento de cualquier tipo en 12 meses) &lt; 1 %: la celda no está entre las
          que devuelve el pronóstico.</p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <h3 style={{ marginBottom: 6 }}>Probabilidad de al menos un evento en 12 meses</h3>
            <HBars max={1} format={(v) => pct(v)}
                   rows={(["any", ...TYPE_ORDER] as Outcome[]).map((o) => ({ label: TYPE_LABEL[o], value: c[`p_window_${o}`][i], color: TYPE_COLOR[o] }))} />
          </div>
          <div>
            <h3>Mes a mes</h3>
            <Legend items={TYPE_ORDER.map((t) => ({ label: TYPE_SHORT[t], color: TYPE_COLOR[t], kind: "line" as const }))} />
            <LineChart x={months} height={160} yMin={0} yMax={pMax(c, i)}
                       yFormat={pctAxis(pMax(c, i))} xFormat={(k) => monthName(P.months[k]).slice(0, 3)}
                       series={TYPE_ORDER.map((t) => ({ key: t, label: TYPE_SHORT[t], color: TYPE_COLOR[t], values: c[`p_month_${t}`][i] }))}
                       ariaLabel="Probabilidad mensual de evento por tipo" />
          </div>
          <div>
            <h3>Muertes estatales por mes</h3>
            <p className="sub" style={{ margin: "2px 0 4px" }}>Mediana y percentil 90 de la simulación; esperadas en 12 meses: {num(c.deaths_sb_expected[i], 0)}</p>
            <LineChart x={months} height={130} yMin={0} xFormat={(k) => monthName(P.months[k]).slice(0, 3)}
                       series={[{ key: "p50", label: "Mediana", color: "var(--series-1)", values: c.deaths_sb_p50[i] },
                                { key: "p90", label: "Percentil 90", color: "var(--series-1)", dash: true, values: c.deaths_sb_p90[i] }]}
                       ariaLabel="Muertes estatales por mes" />
          </div>
          <div>
            <h3 style={{ marginBottom: 6 }}>De dónde viene el riesgo del mes siguiente</h3>
            <ShareBar parts={[
              { label: "Su propio pasado", value: c.driver_own[i], color: "var(--series-1)" },
              { label: "Celdas vecinas", value: c.driver_neighbours[i], color: "var(--series-2)" },
              { label: "Su país", value: c.driver_country[i], color: "var(--series-3)" },
              { label: "Fondo estructural", value: c.driver_background[i], color: "var(--series-4)" },
            ]} />
          </div>
        </div>
      )}
      <div style={{ marginTop: 14 }}>
        <h3>Eventos observados por año</h3>
        {!d && !detail.error && <Loading what="Leyendo la historia de la celda" />}
        {d && hist && (
          <>
            <p className="sub" style={{ margin: "2px 0 4px" }}>
              Desde 1989: {TYPE_ORDER.map((t) => `${d.events_total_since_1989[t]} ${TYPE_SHORT[t].toLowerCase()}`).join(" · ")}.
              Vecinas, últimos 12 meses: {d.neighbours_events_last12} eventos.
            </p>
            <StackedColumns labels={hist.years.map((y) => `'${y.slice(2)}`)} height={140}
                            stacks={TYPE_ORDER.map((t) => ({ key: t, label: TYPE_SHORT[t], color: TYPE_COLOR[t], values: hist.by[t] }))}
                            ariaLabel="Eventos observados por año y tipo" />
            <Legend items={TYPE_ORDER.map((t) => ({ label: TYPE_SHORT[t], color: TYPE_COLOR[t] }))} />
          </>
        )}
      </div>
    </Card>
  );
}

function pMax(c: ConflictPayload["cells"], i: number): number {
  return Math.min(1, Math.max(0.02, ...TYPE_ORDER.map((t) => Math.max(...c[`p_month_${t}`][i]))) * 1.1);
}

function CountryTable({ P, names, onFocus, focus }: { P: ConflictPayload; names: Record<string, string>;
  onFocus: (gw: number | null) => void; focus: number | null }) {
  const [all, setAll] = useState(false);
  const rows = all ? P.countries : P.countries.slice(0, 12);
  return (
    <Card title="Países" sub="Suma de sus celdas en los 12 meses; clic para acercar el mapa"
          actions={<button className="btn small" onClick={() => setAll(!all)}>{all ? "Ver menos" : `Ver los ${P.countries.length}`}</button>}>
      <div className="scroll" style={{ maxHeight: all ? 420 : undefined }}>
        <table>
          <thead>
            <tr><th>País</th><th className="num">Eventos esperados</th><th className="num">Muertes estatales</th>
              <th className="num">Celdas con P &gt; 50 %</th><th className="num">P máx. en una celda</th></tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.gw} className={`clickable ${focus === c.gw ? "selected" : ""}`} onClick={() => onFocus(focus === c.gw ? null : c.gw)}>
                <td>{names[String(c.gw)] ?? `GW ${c.gw}`}</td>
                <td className="num">{num(c.events_expected, 0)}</td>
                <td className="num">{num(c.deaths_sb_expected, 0)}</td>
                <td className="num">{c.cells_p_any_over_50}</td>
                <td className="num">{pct(c.max_p_any)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

const MODEL_NAMES: Record<string, string> = {
  ensemble: "Este modelo (combinación)", logistic: "Sólo logística", hawkes: "Sólo Hawkes",
  bm_neighbors12: "ViEWS conflictology neighbors12", bm_cell12: "Frecuencia 12 meses de la celda",
  bm_persistence: "Persistencia", bm_history: "Frecuencia histórica", bm_zero: "ViEWS exactly zero", bm_last: "ViEWS last historical",
};

function ValidationCard({ P }: { P: ConflictPayload }) {
  const [open, setOpen] = useState(false);
  const v = P.validation;
  const full = useApi(open ? (s) => getJSON<ConflictValidation>("/api/conflict/validation", undefined, s) : null, [open]);
  if (!v) return null;
  return (
    <Card title="¿Cuánto acierta?" sub={`Validación fuera de muestra: ${v.origins.length} orígenes (${v.origins[0]} … ${v.origins[v.origins.length - 1]}), 12 meses cada uno, las 64 818 celdas`}
          actions={<button className="btn small" onClick={() => setOpen(!open)}>{open ? "Ocultar detalle" : "Ver detalle"}</button>}>
      <div className="grid3">
        <div>
          <h3 style={{ marginBottom: 6 }}>AUPR, cualquier evento</h3>
          <p className="sub">Más alto es mejor; la tasa base es ~0,6 %</p>
          <HBars stacked max={0.6} format={(x) => num(x, 3)} rows={Object.entries(v.any_event_aupr).map(([k, x]) => ({ label: MODEL_NAMES[k] ?? k, value: x, color: k === "ensemble" ? "var(--series-1)" : "var(--ref)" }))} />
        </div>
        <div>
          <h3 style={{ marginBottom: 6 }}>Brier, cualquier evento</h3>
          <p className="sub">Más bajo es mejor</p>
          <HBars stacked max={0.01} format={(x) => num(x, 5)} rows={Object.entries(v.any_event_brier).map(([k, x]) => ({ label: MODEL_NAMES[k] ?? k, value: x, color: k === "ensemble" ? "var(--series-1)" : "var(--ref)" }))} />
        </div>
        <div>
          <h3 style={{ marginBottom: 6 }}>CRPS muertes, África y O. Medio</h3>
          <p className="sub">Más bajo es mejor</p>
          <HBars stacked max={0.4} format={(x) => num(x, 3)} rows={Object.entries(v.sb_deaths_crps_africa_me).map(([k, x]) => ({ label: MODEL_NAMES[k] ?? k, value: x, color: k === "ensemble" ? "var(--series-1)" : "var(--ref)" }))} />
        </div>
      </div>
      {open && full.loading && <Loading what="Cargando la validación" />}
      {open && full.error && <ErrorNotice error={full.error} />}
      {open && full.data && <ValidationDetail v={full.data} />}
    </Card>
  );
}

function ValidationDetail({ v }: { v: ConflictValidation }) {
  const by = v.aupr_by_origin.any;
  const origins = Object.keys(by.ensemble).sort();
  const x = origins.map((_, i) => i);
  const models: [string, string][] = [["ensemble", "var(--series-1)"], ["bm_cell12", "var(--series-2)"], ["bm_neighbors12", "var(--series-3)"]];
  const cal = v.calibration["ensemble|any|h1-3"].filter((b) => b[2] > 0);
  const edges = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1];
  return (
    <div className="grid2" style={{ marginTop: 16 }}>
      <div>
        <h3>AUPR por origen (cualquier evento)</h3>
        <Legend items={models.map(([m, c]) => ({ label: MODEL_NAMES[m], color: c, kind: "line" as const }))} />
        <LineChart x={x} height={190} yMin={0} xFormat={(i) => origins[i]?.slice(0, 4) ?? ""} yFormat={(y) => num(y, 2)}
                   series={models.map(([m, c]) => ({ key: m, label: MODEL_NAMES[m], color: c, values: origins.map((o) => by[m]?.[o] ?? null) }))} />
      </div>
      <div>
        <h3>Calibración (meses 1-3)</h3>
        <p className="sub">Probabilidad pronosticada frente a frecuencia observada, por intervalo</p>
        <table>
          <thead><tr><th>Intervalo</th><th className="num">P media</th><th className="num">Observada</th><th className="num">Celda-mes</th></tr></thead>
          <tbody>
            {cal.map((b) => {
              const k = v.calibration["ensemble|any|h1-3"].indexOf(b);
              return (
                <tr key={k}><td>{pct(edges[k], 1)}–{pct(edges[k + 1], 1)}</td><td className="num">{pct(b[0] / b[2], 2)}</td>
                  <td className="num">{pct(b[1] / b[2], 2)}</td><td className="num">{num(b[2], 0)}</td></tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
