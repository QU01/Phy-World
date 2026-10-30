import { useEffect, useState } from "react";
import { getJSON, postJSON, ApiError, type L0RResult, type Levers, type PolicyResult } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, ErrorNotice, Field, Legend, Loading } from "../components/common";
import { LineChart } from "../components/charts";
import { compact, num, pct } from "../lib/format";

const LEVER_ES: Record<string, string> = {
  carbon_price: "Precio mundial del carbono", renewables: "Inversión pública en energía no fósil",
  efficiency: "Inversión en eficiencia energética", agri_rd: "I+D agrícola y extensión",
  aid: "Ayuda al desarrollo (a África subsahariana y Asia del Sur)", fertility: "Educación de niñas y planificación familiar",
  consolidation: "Consolidación fiscal",
};
const METRIC_ES: Record<string, string> = {
  welfare: "Bienestar (log del consumo per cápita descontado)", peak_gmst: "Pico de temperatura 2020-2100 (°C)",
  poorest: "Consumo de la región más pobre en 2100 (log)", food_ratio_min: "Cereal per cápita mínimo regional / 2019",
  land_ratio: "Tierras de cultivo 2100 / 2019", debt_world: "Deuda pública mundial / PIB en 2100",
  gmst_2050: "Temperatura 2050 (°C)", gmst_2100: "Temperatura 2100 (°C)", nonfossil_2050: "Cuota no fósil de la energía en 2050",
};
const CONSTRAINT_ES: Record<string, string> = {
  food_floor: "Piso alimentario (cereal per cápita ≥ 90 % de 2019 en toda región, p10)",
  land_cap: "Tope de tierras de cultivo (≤ +10 % en 2100, p90)", debt_cap: "Tope de deuda (≤ 150 % del PIB en 2100, p90)",
};
const UNIT_ES: Record<string, string> = {
  "USD/tCO2": "USD/tCO₂", "share of GDP": "del PIB", "share of donor GDP": "del PIB donante",
  "share of GDP in SSA, SAS, MEA": "del PIB en ASS, Asia del Sur y Oriente Medio", "fraction of the deficit": "del déficit",
};
const BAND_ES: Record<string, string> = {
  pop: "Población mundial", gdp: "PIB mundial", energy: "Energía primaria", cereal: "Cereales", cropland: "Tierras de cultivo",
  gmst: "Temperatura media global",
};

export function PolicyView() {
  const lv = useApi((s) => getJSON<Levers>("/api/levers", undefined, s), []);
  const [theta, setTheta] = useState<Record<string, number>>({});
  const [narrative, setNarrative] = useState("ssp2");
  const [res, setRes] = useState<{ data?: PolicyResult; error?: ApiError; loading: boolean }>({ loading: false });
  const bands = useApi((s) => getJSON<L0RResult>("/api/l0r", { posterior: true, members: 100 }, s), []);
  useEffect(() => {
    if (lv.data) setNarrative(lv.data.default_narrative);
  }, [lv.data]);
  const run = () => {
    setRes({ loading: true });
    postJSON<PolicyResult>("/api/policy", { theta, narrative, members: 100 })
      .then((data) => setRes({ data, loading: false }))
      .catch((e) => setRes({ error: e instanceof ApiError ? e : new ApiError(0, String(e)), loading: false }));
  };
  const p = res.data;
  return (
    <>
      <div className="viewhead">
        <h2>Políticas y regiones (F6, F8)</h2>
        <p>
          El L0 regional (10 regiones, calibrado con datos ≤ 2019 y con posterior bayesiano) y la evaluación robusta
          de una política sobre un ensemble de estados del mundo: objetivos en percentiles, límites tipo planetarios y
          comparación con no hacer nada. Exploratorio.
        </p>
      </div>
      <Card title="Mundo a 2100 sin políticas nuevas" sub="L0 regional con el posterior de F7: p10-p90 que incluyen la discrepancia declarada">
        {bands.error && <ErrorNotice error={bands.error} what="el L0 regional" />}
        {bands.loading && <Loading what="Corriendo 100 miembros del L0 regional" />}
        {bands.data && (
          <>
            <Legend items={[{ label: "Mediana", color: "var(--series-1)", kind: "line" }, { label: "p10–p90", color: "var(--series-1)", kind: "band" }]} />
            <div className="grid3">
              {Object.entries(bands.data.world).map(([k, b]) => (
                <div key={k}>
                  <h3>{BAND_ES[k] ?? k}</h3>
                  <p className="sub">{b.unit}</p>
                  <LineChart x={bands.data!.years} height={150} yFormat={compact}
                             series={[{ key: k, label: "Mediana", color: "var(--series-1)", values: b.p50, band: { lo: b.p10, hi: b.p90 } }]} />
                </div>
              ))}
            </div>
          </>
        )}
      </Card>
      <Card title="Evaluar una política" sub="Las palancas se pagan (salen de la inversión) y su eficacia es incierta en cada estado del mundo">
        {lv.error && <ErrorNotice error={lv.error} />}
        {lv.data && (
          <div className="grid2" style={{ alignItems: "start" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {lv.data.levers.map((l) => {
                const v = theta[l.name] ?? 0;
                return (
                  <Field key={l.name} label={`${LEVER_ES[l.name] ?? l.name}: ${l.unit.startsWith("USD") ? `${num(v, 0)} ${UNIT_ES[l.unit]}` : `${pct(v, 2)} ${UNIT_ES[l.unit] ?? l.unit}`}`}>
                    <input type="range" min={l.lo} max={l.hi} step={(l.hi - l.lo) / 100} value={v} title={l.what}
                           onChange={(e) => setTheta({ ...theta, [l.name]: +e.target.value })} />
                  </Field>
                );
              })}
              <div className="row">
                <Field label="Narrativa (SSP)">
                  <select value={narrative} onChange={(e) => setNarrative(e.target.value)}>
                    {lv.data.narratives.map((n) => <option key={n} value={n}>{n.toUpperCase()}</option>)}
                  </select>
                </Field>
                <button className="btn primary" onClick={run} disabled={res.loading}>Evaluar</button>
                <button className="btn" onClick={() => setTheta({})}>Sin política</button>
              </div>
            </div>
            <div>
              {res.loading && <Loading what="Evaluando la política sobre el ensemble" slow="Incluye la búsqueda de vulnerabilidades (PRIM): hasta un par de minutos." />}
              {res.error && <ErrorNotice error={res.error} what="evaluar políticas" />}
              {p && <PolicyOutcome p={p} />}
            </div>
          </div>
        )}
      </Card>
    </>
  );
}

function PolicyOutcome({ p }: { p: PolicyResult }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="notice" style={{ borderLeftColor: p.feasible ? "var(--good)" : "var(--critical)" }}>
        <div className="title">{p.feasible ? "✓ Dentro de los límites" : "✕ Viola al menos un límite"}</div>
        <div className="secondary small">
          Mejor que no hacer nada en bienestar en {pct(p.better_than_baseline_share.welfare, 0)} de los estados del mundo; con
          menor pico de temperatura en {pct(p.better_than_baseline_share.peak_gmst, 0)}.
        </div>
      </div>
      <table>
        <thead><tr><th>Límite</th><th className="num">Con la política</th><th className="num">Sin política</th></tr></thead>
        <tbody>
          {Object.entries(p.constraints).map(([k, g]) => (
            <tr key={k}><td>{CONSTRAINT_ES[k] ?? k}</td>
              <td className="num">{g <= 0 ? "✓" : "✕"} {num(g, 3)}</td>
              <td className="num">{(p.baseline_constraints[k] ?? 0) <= 0 ? "✓" : "✕"} {num(p.baseline_constraints[k], 3)}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">g ≤ 0 es dentro del límite.</p>
      <table>
        <thead><tr><th>Resultado (2020-2100)</th><th className="num">p10</th><th className="num">p50</th><th className="num">p90</th></tr></thead>
        <tbody>
          {Object.entries(p.percentiles).map(([k, q]) => (
            <tr key={k}><td>{METRIC_ES[k] ?? k}<div className="small muted">{p.units[k]}</div></td>
              <td className="num">{num(q["10"], 3)}</td><td className="num">{num(q["50"], 3)}</td><td className="num">{num(q["90"], 3)}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
