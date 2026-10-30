import { useMemo, useState } from "react";
import { getJSON, type SocialResult } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, ErrorNotice, Field, Legend, Loading, Seg } from "../components/common";
import { LineChart } from "../components/charts";
import { Choropleth } from "../components/Choropleth";
import { countryName } from "../lib/geo";
import { compact, num, pct, pctAxis } from "../lib/format";

const WORLD_KEYS = ["in_conflict", "conflict_onset", "coup_success", "in_crisis", "demo_pop", "loss"] as const;
const W3_KEYS = ["pop", "iopc", "fpc", "le"] as const;
type CountryMetric = "p_conflict" | "conflict" | "p_coup" | "crisis" | "dem";
const CM: { value: CountryMetric; label: string; breaks: number[]; labels: string[] }[] = [
  { value: "conflict", label: "En conflicto", breaks: [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8], labels: ["5 %", "10 %", "20 %", "30 %", "45 %", "60 %", "80 %"] },
  { value: "p_conflict", label: "P(inicio de conflicto)", breaks: [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12], labels: ["0,5 %", "1 %", "2 %", "3 %", "5 %", "8 %", "12 %"] },
  { value: "p_coup", label: "P(intento de golpe)", breaks: [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12], labels: ["0,5 %", "1 %", "2 %", "3 %", "5 %", "8 %", "12 %"] },
  { value: "crisis", label: "En crisis de deuda", breaks: [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8], labels: ["5 %", "10 %", "20 %", "30 %", "45 %", "60 %", "80 %"] },
  { value: "dem", label: "Democracia", breaks: [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95], labels: ["5 %", "15 %", "30 %", "50 %", "70 %", "85 %", "95 %"] },
];

export function SocialView() {
  const [scenario, setScenario] = useState(1);
  const [feedback, setFeedback] = useState(true);
  const [n, setN] = useState(40);
  const r = useApi((s) => getJSON<SocialResult>("/api/social", { scenario, feedback, n }, s), [scenario, feedback, n]);
  const [cm, setCm] = useState<CountryMetric>("conflict");
  const [yearIdx, setYearIdx] = useState(30);
  const [iso, setIso] = useState<string | null>(null);
  const d = r.data;
  const yi = d ? Math.min(yearIdx, d.years.length - 1) : 0;
  const values = useMemo(() => {
    if (!d) return {};
    const out: Record<string, number> = {};
    d.iso3.forEach((c, k) => (out[c] = d.country[cm][yi][k]));
    return out;
  }, [d, cm, yi]);
  const spec = CM.find((x) => x.value === cm)!;
  return (
    <>
      <div className="viewhead">
        <h2>Sociedad ↔ World3</h2>
        <p>
          World3-03 acoplado a 180 países en las dos direcciones: el ingreso de World3 mueve los riesgos de conflicto,
          golpes, crisis de deuda y democracia (logits calibrados con 1950-2019), y esos eventos destruyen capital y
          capacidad de política en World3. Es un modo de comportamiento a 2100, no un pronóstico.
        </p>
      </div>
      <div className="controls card" style={{ padding: "10px 14px" }}>
        <Field label="Escenario de World3">
          <select value={scenario} onChange={(e) => setScenario(+e.target.value)}>
            {Object.entries(d?.scenarios ?? { 1: "Negocios como de costumbre" }).map(([k, v]) => (
              <option key={k} value={k}>{k}. {v}</option>
            ))}
          </select>
        </Field>
        <Field label="La sociedad afecta a World3">
          <Seg value={feedback ? "1" : "0"} onChange={(v) => setFeedback(v === "1")}
               options={[{ value: "1", label: "Sí (acople)" }, { value: "0", label: "No (sólo ida)" }]} />
        </Field>
        <Field label="Miembros del ensemble">
          <Seg value={n} onChange={setN} options={[20, 40, 100].map((v) => ({ value: v, label: String(v) }))} />
        </Field>
      </div>
      {r.error && <ErrorNotice error={r.error} />}
      {!d && !r.error && <Loading what="Corriendo World3 con la sociedad" />}
      {d && (
        <>
          <div className="grid3">
            {WORLD_KEYS.filter((k) => d.world[k]).map((k) => {
              const b = d.world[k];
              const share = b.unit.startsWith("fracción");
              return (
                <Card key={k} title={b.label} sub={b.unit}>
                  <LineChart x={d.years} height={170} yMin={0} yFormat={share ? pctAxis(Math.max(...b.p90)) : (v) => num(v, v < 10 ? 1 : 0)}
                             series={[{ key: k, label: "Mediana", color: "var(--series-1)", values: b.p50, band: { lo: b.p10, hi: b.p90 } }]}
                             ariaLabel={b.label} />
                </Card>
              );
            })}
          </div>
          <Legend items={[{ label: "Mediana de los miembros", color: "var(--series-1)", kind: "line" },
                          { label: "p10–p90", color: "var(--series-1)", kind: "band" }]} />
          <Card title="Por país" sub="Fracción de los miembros del ensemble (o probabilidad media) en el año elegido; clic en un país para ver su trayectoria">
            <div className="controls" style={{ marginBottom: 10 }}>
              <Seg value={cm} onChange={setCm} options={CM.map((x) => ({ value: x.value, label: x.label }))} />
              <Field label={`Año: ${d.years[yi]}`}>
                <input type="range" min={0} max={d.years.length - 1} value={yi} onChange={(e) => setYearIdx(+e.target.value)} />
              </Field>
            </div>
            <div className="grid2" style={{ alignItems: "start" }}>
              <Choropleth values={values} breaks={spec.breaks} labels={spec.labels} format={(v) => pct(v, 1)}
                          title={`${spec.label}, ${d.years[yi]}`} selected={iso} onSelect={setIso} />
              {iso && d.iso3.includes(iso) ? (
                <div>
                  <h3>{countryName(iso)}</h3>
                  <Legend items={[{ label: "En conflicto", color: "var(--series-1)", kind: "line" },
                                  { label: "En crisis de deuda", color: "var(--series-2)", kind: "line" },
                                  { label: "Democracia", color: "var(--series-3)", kind: "line" }]} />
                  <LineChart x={d.years} height={220} yMin={0} yMax={1} yFormat={(v) => pct(v, 0)}
                             marker={{ x: d.years[yi], label: String(d.years[yi]) }}
                             series={[
                               { key: "c", label: "En conflicto", color: "var(--series-1)", values: d.country.conflict.map((r) => r[d.iso3.indexOf(iso)]) },
                               { key: "k", label: "En crisis de deuda", color: "var(--series-2)", values: d.country.crisis.map((r) => r[d.iso3.indexOf(iso)]) },
                               { key: "d", label: "Democracia", color: "var(--series-3)", values: d.country.dem.map((r) => r[d.iso3.indexOf(iso)]) },
                             ]} ariaLabel={`Trayectoria de ${countryName(iso)}`} />
                  <p className="small muted">Fracción de los {d.members} miembros en cada estado.</p>
                </div>
              ) : (
                <p className="secondary">Seleccione un país en el mapa.</p>
              )}
            </div>
          </Card>
          <h3>World3 en esta corrida</h3>
          <div className="grid2">
            {W3_KEYS.filter((k) => d.world3[k]).map((k) => {
              const b = d.world3[k];
              return (
                <Card key={k} title={b.label} sub={b.unit}>
                  <LineChart x={d.w3_years} height={160} yFormat={compact}
                             series={[{ key: k, label: "Mediana", color: "var(--series-1)", values: b.p50, band: { lo: b.p10, hi: b.p90 } }]}
                             ariaLabel={b.label} />
                </Card>
              );
            })}
          </div>
        </>
      )}
    </>
  );
}
