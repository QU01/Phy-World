import { useMemo, useState } from "react";
import { getJSON, SNAPSHOT, type DensityCountry, type DensityPayload, type DensityScenario, type DensityStudy } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, ErrorNotice, Field, Legend, Loading, Seg, Stat } from "../components/common";
import { HBars, LineChart } from "../components/charts";
import { Choropleth } from "../components/Choropleth";
import { CellMap, type CellLayer } from "../components/CellMap";
import { classOf } from "../lib/colors";
import { countryName } from "../lib/geo";
import { compact, num, pct } from "../lib/format";

const BREAKS = [0.01, 0.05, 0.2, 1, 5, 20, 100];
const LABELS = ["0,01", "0,05", "0,2", "1", "5", "20", "100"];

export function DensityView({ theme }: { theme: string }) {
  const [origin, setOrigin] = useState(2025);
  const [h, setH] = useState(1);
  const [layer, setLayer] = useState<"pais" | "celdas">("pais");
  const [iso, setIso] = useState<string | null>(null);
  const r = useApi((s) => getJSON<DensityPayload>("/api/density", { origin, h, cells: true }, s), [origin, h]);
  const [lens, setLens] = useState<"nb" | "ppml">("nb");
  const study = useApi((s) => getJSON<DensityStudy>("/api/density/study", { lens }, s), [lens]);
  const d = r.data;
  const values = useMemo(() => Object.fromEntries((d?.countries ?? []).map((c) => [c.iso3, c.density])), [d]);
  const sel = d?.countries.find((c) => c.iso3 === iso) ?? null;
  return (
    <>
      <div className="viewhead">
        <h2>Densidad de eventos (Atlas)</h2>
        <p>
          Eventos de violencia organizada esperados por cada 10 000 km² de cada país, en el año que se elija, con
          regresiones sobre la historia del conflicto y las variables del Atlas elegidas fuera de muestra (F13).
          Desde cualquier año de 1990 a 2025; después de 2019 el Atlas queda en su último dato.
        </p>
      </div>
      <div className="controls card" style={{ padding: "10px 14px" }}>
        <Field label="Información hasta el año">
          <select value={origin} onChange={(e) => setOrigin(+e.target.value)}>
            {(SNAPSHOT ? [2025] : Array.from({ length: 31 }, (_, k) => 2025 - k)).map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </Field>
        <Field label="Años hacia delante">
          <Seg value={h} onChange={setH} options={[1, 2, 3, 4, 5].map((v) => ({ value: v, label: `${v}` }))} />
        </Field>
        <Field label="Mapa">
          <Seg value={layer} onChange={setLayer} options={[{ value: "pais", label: "Por país" }, { value: "celdas", label: "Por celda" }]} />
        </Field>
        {d && <span className="secondary">Pronóstico para <b>{d.target}</b> · modelo entrenado con objetivos ≤ {d.train_until}</span>}
      </div>
      {r.error && <ErrorNotice error={r.error} what="la densidad de eventos" />}
      {!d && !r.error && <Loading what="Ajustando las regresiones y prediciendo" />}
      {d && (
        <div className="conflict-layout">
          <div style={{ display: "flex", flexDirection: "column", gap: 16, minWidth: 0 }}>
            {layer === "pais" ? (
              <Choropleth values={values} breaks={BREAKS} labels={LABELS} format={(v) => `${num(v, v < 1 ? 3 : 1)} eventos / 10 000 km²`}
                          title={`Eventos esperados por 10 000 km², ${d.target}`} selected={iso} onSelect={setIso} />
            ) : (
              <CellsBlock d={d} theme={theme} />
            )}
            <CountryTable d={d} iso={iso} onPick={setIso} />
          </div>
          <div className="side">
            <Card title={`Mundo, ${d.target}`} sub="Suma de los eventos esperados de todos los países">
              <div className="stats">
                <Stat value={compact(d.world_events_expected)} label="eventos esperados" />
                <Stat value={compact(d.countries.reduce((a, c) => a + (c.observed ?? 0), 0) || NaN)} label={d.countries.some((c) => c.observed !== null) ? "observados (UCDP GED)" : "aún sin dato"} />
              </div>
            </Card>
            {sel ? <CountryCard c={sel} d={d} labels={study.data?.labels} /> : (
              <Card title="Un país"><p className="secondary">Seleccione un país en el mapa o en la tabla.</p></Card>
            )}
            {d.validation && <ValidationCard v={d.validation} h={d.horizon} />}
          </div>
        </div>
      )}
      <StudyCard study={study.data} error={study.error} h={h} lens={lens} onLens={setLens} />
      <ScenarioCard />
    </>
  );
}

function CellsBlock({ d, theme }: { d: DensityPayload; theme: string }) {
  const cells = d.cells!;
  const layer: CellLayer = useMemo(() => {
    const cls = new Int8Array(cells.gid.length);
    cells.density.forEach((v, i) => (cls[i] = classOf(v, BREAKS)));
    return { lon: cells.lon, lat: cells.lat, gid: cells.gid, cls };
  }, [cells]);
  const idx = useMemo(() => new Map(cells.gid.map((g, i) => [g, i])), [cells]);
  return (
    <div style={{ position: "relative" }}>
      <CellMap layer={layer} selectedGid={null} onSelect={() => {}} focus={null} theme={theme}
               tooltip={(gid) => {
                 const i = idx.get(gid);
                 return i === undefined ? <div className="r">sin eventos esperados</div> : (
                   <>
                     <div className="t">{countryName(cells.iso3[i])} · celda {gid}</div>
                     <div className="r">{num(cells.density[i], 2)} eventos / 10 000 km² · {num(cells.events_expected[i], 1)} eventos</div>
                   </>
                 );
               }} />
      <div className="maplegend">
        <div>Eventos esperados por 10 000 km², {d.target} (el país repartido por la historia de sus celdas)</div>
        <div className="ramp">
          {LABELS.map((l, i) => <span key={l}><i style={{ background: `var(--r${i})` }} />{l}</span>)}
        </div>
      </div>
    </div>
  );
}

function CountryTable({ d, iso, onPick }: { d: DensityPayload; iso: string | null; onPick: (i: string) => void }) {
  const [all, setAll] = useState(false);
  const rows = all ? d.countries : d.countries.slice(0, 15);
  return (
    <Card title="Países" sub="Ordenados por eventos esperados; p10-p90 es la banda de la distribución predictiva"
          actions={<button className="btn small" onClick={() => setAll(!all)}>{all ? "Ver menos" : `Ver los ${d.countries.length}`}</button>}>
      <div className="scroll" style={{ maxHeight: all ? 460 : undefined }}>
        <table>
          <thead>
            <tr><th>País</th><th className="num">Eventos esperados</th><th className="num">Densidad / 10 000 km²</th>
              <th className="num">P(algún evento)</th><th className="num">p10–p90</th><th className="num">Observados</th></tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.iso3} className={`clickable ${iso === c.iso3 ? "selected" : ""}`} onClick={() => onPick(c.iso3)}>
                <td>{countryName(c.iso3)}{c.in_atlas ? "" : " *"}</td>
                <td className="num">{num(c.events_expected, c.events_expected < 10 ? 1 : 0)}</td>
                <td className="num">{num(c.density, c.density < 1 ? 3 : 1)}</td>
                <td className="num">{pct(c.p_any)}</td>
                <td className="num">{num(c.p10, 0)}–{num(c.p90, 0)}</td>
                <td className="num">{c.observed === null ? "–" : num(c.observed, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">* País que el Atlas no tiene: su pronóstico se apoya en su historia de eventos.</p>
    </Card>
  );
}

function CountryCard({ c, d, labels }: { c: DensityCountry; d: DensityPayload; labels?: DensityStudy["labels"] }) {
  const name = (f: string) => {
    const base = f.replace(/__na$/, "");
    const l = labels?.[base]?.label ?? base;
    return f.endsWith("__na") ? `${l} (dato faltante)` : l;
  };
  return (
    <Card title={countryName(c.iso3)} sub={`${d.target} · ${num(c.area_km2 / 1000, 0)} mil km²${c.atlas_frozen_years ? ` · Atlas congelado hace ${c.atlas_frozen_years} años` : ""}`}>
      <div className="stats" style={{ marginBottom: 12 }}>
        <Stat value={num(c.events_expected, c.events_expected < 10 ? 1 : 0)} label="eventos esperados" />
        <Stat value={num(c.density, c.density < 1 ? 3 : 1)} label="por 10 000 km²" />
        <Stat value={pct(c.p_any)} label="P(algún evento)" />
      </div>
      <dl className="kv">
        <dt>p10 – p50 – p90</dt><dd>{num(c.p10, 0)} – {num(c.p50, 0)} – {num(c.p90, 0)} eventos</dd>
        <dt>Observados</dt><dd>{c.observed === null ? "aún sin dato" : `${num(c.observed, 0)} eventos`}</dd>
      </dl>
      <h3 style={{ margin: "12px 0 6px" }}>Lo que más pesa en su pronóstico</h3>
      <p className="sub">Multiplica los eventos esperados frente a un país promedio</p>
      <HBars max={Math.max(...c.drivers.map((x) => Math.abs(x.log_effect)), 0.1)} format={() => ""}
             rows={c.drivers.map((x) => ({ label: `${name(x.factor)} · ×${num(Math.exp(x.log_effect), 2)}`,
                                           value: Math.abs(x.log_effect),
                                           color: x.log_effect >= 0 ? "var(--series-2)" : "var(--series-1)" }))} />
      <Legend items={[{ label: "sube el pronóstico", color: "var(--series-2)" }, { label: "lo baja", color: "var(--series-1)" }]} />
    </Card>
  );
}

function ValidationCard({ v, h }: { v: NonNullable<DensityPayload["validation"]>; h: number }) {
  const eras: [string, string][] = [["atlas_observado", "2005–2019 (Atlas observado)"], ["atlas_congelado", "2020–2025 (Atlas congelado)"]];
  return (
    <Card title={`¿Cuánto acierta a ${h} ${h === 1 ? "año" : "años"}?`} sub="Validación con origen rodante; los años 2020-2025 no se usaron para elegir los factores">
      <table>
        <thead><tr><th>Período</th><th className="num">Mejora sobre la historia</th><th className="num">AUPR</th><th className="num">Esperado / observado</th></tr></thead>
        <tbody>
          {eras.filter(([k]) => v[k]).map(([k, label]) => (
            <tr key={k}><td>{label}</td><td className="num">{pct(v[k].skill_vs_history, 0)}</td>
              <td className="num">{num(v[k].aupr_any, 2)}</td><td className="num">{num(v[k].expected_over_observed, 2)}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">Mejora = reducción del RPS (órdenes de magnitud de eventos) frente al modelo que sólo usa la historia del conflicto.</p>
    </Card>
  );
}

function StudyCard({ study, error, h, lens, onLens }: { study?: DensityStudy; error?: import("../lib/api").ApiError; h: number;
  lens: "nb" | "ppml"; onLens: (l: "nb" | "ppml") => void }) {
  const key = h <= 2 ? "h1" : h <= 4 ? "h3" : "h5";
  if (error) return <ErrorNotice error={error} what="el estudio de factores" />;
  if (!study) return null;
  const s = study.horizons[key];
  if (!s) return null;
  const top = Object.entries(s.single).sort((a, b) => b[1].skill - a[1].skill).slice(0, 12);
  const worst = Object.entries(s.single).sort((a, b) => a[1].skill - b[1].skill).slice(0, 5);
  const groups = Object.entries(s.groups).sort((a, b) => b[1] - a[1]);
  return (
    <Card title={`¿Qué predice mejor los eventos? (a ${key.slice(1)} ${key === "h1" ? "año" : "años"})`}
          sub={lens === "nb"
            ? "Lente del orden de magnitud (binomial negativa): dónde habrá violencia y de qué escala. Cada factor añadido a la historia del conflicto, fuera de muestra (2005-2019): reducción del RPS"
            : "Lente del volumen (PPML, la del sistema): cuántos eventos habrá en total. Cada factor añadido a la historia del conflicto, fuera de muestra (2005-2019): reducción del RPS"}
          actions={<Seg value={lens} onChange={onLens} options={[{ value: "nb", label: "Orden de magnitud" }, { value: "ppml", label: "Volumen" }]} />}>
      <div className="grid3" style={{ alignItems: "start" }}>
        <div>
          <h3 style={{ marginBottom: 8 }}>Los 12 factores que más mejoran</h3>
          <HBars stacked max={Math.max(...top.map(([, v]) => v.skill))} format={(v) => pct(v, 1)}
                 rows={top.map(([f, v]) => ({ label: `${study.labels[f]?.label ?? f} · ${v.group}`, value: v.skill, color: "var(--series-1)" }))} />
        </div>
        <div>
          <h3 style={{ marginBottom: 8 }}>Por grupo</h3>
          <HBars stacked max={Math.max(...groups.map(([, v]) => Math.abs(v)))} format={(v) => pct(v, 1)}
                 rows={groups.map(([g, v]) => ({ label: g, value: Math.max(v, 0), color: g.includes("World3") ? "var(--series-2)" : "var(--series-1)",
                                                 note: pct(v, 1) }))} />
          <p className="small muted" style={{ marginTop: 8 }}>
            Todo el Atlas: {pct(s.todo_atlas, 1)}. Añadir World3 mundial a todo el Atlas lo deja en {pct(s.todo_atlas_mas_world3, 1)}.
            {groups.filter(([, v]) => v < 0).map(([g, v]) => ` ${g}: ${pct(v, 1)}.`)}
          </p>
        </div>
        <div>
          <h3 style={{ marginBottom: 8 }}>Los que empeoran la predicción</h3>
          <table>
            <tbody>
              {worst.map(([f, v]) => (
                <tr key={f}><td>{study.labels[f]?.label ?? f}<div className="small muted">{v.group}</div></td><td className="num">{pct(v.skill, 1)}</td></tr>
              ))}
            </tbody>
          </table>
          <h3 style={{ margin: "12px 0 6px" }}>Selección hacia delante</h3>
          <ol className="caveats">
            {s.forward.filter((p) => p.add).map((p) => (
              <li key={p.add!}>{study.labels[p.add!]?.label ?? p.add} <span className="muted">({pct(p.skill, 1)} acumulado)</span></li>
            ))}
          </ol>
        </div>
      </div>
    </Card>
  );
}

function ScenarioCard() {
  const [on, setOn] = useState(false);
  const r = useApi(on ? (s) => getJSON<DensityScenario>("/api/density/scenario", { origin: 2019, members: 32 }, s) : null, [on]);
  const d = r.data;
  const obsWorld = d ? d.years.map((_, k) => {
    const vals = d.observed.map((row) => row[k]);
    return vals.every((v) => v === null) ? null : vals.reduce((a: number, v) => a + (v ?? 0), 0);
  }) : [];
  return (
    <Card title="Escenario: la densidad sobre las trayectorias del Atlas (L1)"
          sub="Desde 2019, el modelo de 1 año aplicado año a año al ingreso, la deuda, la democracia, los golpes, las crisis y el conflicto civil que simula el L1, con la historia de eventos simulada. Exploratorio."
          actions={!on ? <button className="btn primary" onClick={() => setOn(true)}>Simular 2020-2030</button> : undefined}>
      {r.error && <ErrorNotice error={r.error} what="el escenario" />}
      {on && r.loading && <Loading what="Simulando 32 miembros del L1 y sus eventos" slow="Unos segundos." />}
      {d && (
        <>
          <Legend items={[{ label: "Media del escenario", color: "var(--series-1)", kind: "line" }, { label: "p10–p90", color: "var(--series-1)", kind: "band" },
                          { label: "Observado (UCDP GED)", color: "var(--series-2)", kind: "line" }]} />
          <LineChart x={d.years} height={240} yMin={0} yFormat={compact}
                     series={[{ key: "m", label: "Media", color: "var(--series-1)", values: d.world_events.mean, band: { lo: d.world_events.p10, hi: d.world_events.p90 } },
                              { key: "o", label: "Observado", color: "var(--series-2)", values: obsWorld }]}
                     ariaLabel="Eventos mundiales por año en el escenario frente a lo observado" />
          <p className="small muted">Variables del L1 que mueven el escenario: {d.l1_driven.join(", ")}. Quedan en su valor de 2019: {d.frozen.join(", ") || "ninguna"}.</p>
        </>
      )}
    </Card>
  );
}
