import { useState } from "react";
import { getJSON, type Catalog, type L1Path } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, ErrorNotice, Field, Legend, Loading } from "../components/common";
import { LineChart } from "../components/charts";
import { countryName } from "../lib/geo";
import { compact, num } from "../lib/format";

const VAR_ES: Record<string, string> = {
  gdp_pc: "PIB per cápita", log_gdp_pc: "log del PIB per cápita", conflict_hazard: "Riesgo de conflicto civil",
  debt_gdp: "Deuda pública / PIB", resource_rents_gdp: "Rentas de recursos / PIB", democracy: "Probabilidad de democracia",
  gmst: "Temperatura media global", co2_emissions_world: "Emisiones fósiles de CO₂ (mundo)",
  co2_concentration: "Concentración de CO₂", cereal_production: "Producción de cereales", cropland: "Tierras de cultivo",
  food_supply: "Suministro de calorías", resource_rent_share_world: "Rentas de recursos / PIB mundial",
  oil_price: "Precio real del petróleo", primary_energy: "Energía primaria", population: "Población",
  human_capital: "Capital humano",
};
const WORLD_VARS = new Set(["gmst", "co2_emissions_world", "co2_concentration", "resource_rent_share_world", "oil_price"]);

export function CountryView() {
  const cat = useApi((s) => getJSON<Catalog>("/api/catalog", undefined, s), []);
  const [variable, setVariable] = useState("gdp_pc");
  const [region, setRegion] = useState("COL");
  const [asOf, setAsOf] = useState(2009);
  const [members, setMembers] = useState(32);
  const [req, setReq] = useState<{ variable: string; region: string; as_of: number; members: number } | null>(null);
  const r = useApi(req ? (s) => getJSON<L1Path>("/api/l1", req, s) : null, [JSON.stringify(req)]);
  const world = WORLD_VARS.has(variable);
  const regions = cat.data?.regions ?? [];
  const d = r.data;
  return (
    <>
      <div className="viewhead">
        <h2>Países (L1)</h2>
        <p>
          El Atlas del Sistema-Mundo (180 países) con población, capital humano, recursos y energía endógenos
          (F1-F3): pronóstico desde un año de origen con la calibración de corte ≤ origen. Horizonte medido fuera de
          muestra: 1-10 años. Las bandas p10-p90 son demasiado estrechas (cubren 60-70 % en vez de 80 %).
        </p>
      </div>
      {cat.error && <ErrorNotice error={cat.error} what="el L1" />}
      <div className="controls card" style={{ padding: "10px 14px" }}>
        <Field label="Variable">
          <select value={variable} onChange={(e) => setVariable(e.target.value)}>
            {(cat.data?.observables ?? []).map((o) => (
              <option key={o.variable} value={o.variable}>{VAR_ES[o.variable] ?? o.variable}</option>
            ))}
          </select>
        </Field>
        <Field label="País">
          <select value={world ? "WLD" : region} disabled={world} onChange={(e) => setRegion(e.target.value)}>
            {world ? <option value="WLD">Mundo</option> : regions.map((c) => <option key={c} value={c}>{countryName(c)}</option>)}
          </select>
        </Field>
        <Field label={`Año de origen: ${asOf}`}>
          <input type="range" min={cat.data?.first_origin ?? 1970} max={(cat.data?.last_year ?? 2019) - 1} value={asOf}
                 onChange={(e) => setAsOf(+e.target.value)} />
        </Field>
        <Field label="Miembros">
          <select value={members} onChange={(e) => setMembers(+e.target.value)}>
            {[16, 32, 64].map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </Field>
        <button className="btn primary" disabled={!cat.data}
                onClick={() => setReq({ variable, region: world ? "WLD" : region, as_of: asOf, members })}>
          Pronosticar
        </button>
      </div>
      {r.error && <ErrorNotice error={r.error} what="el L1" />}
      {r.loading && <Loading what="Corriendo el ensemble del L1" slow="Unos segundos por consulta." />}
      {d && !r.loading && (
        <Card title={`${VAR_ES[d.variable] ?? d.variable} · ${d.region === "WLD" ? "Mundo" : countryName(d.region)}`}
              sub={`${d.unit} · origen ${d.as_of} (calibración ≤ ${d.params_cutoff}) · ${d.members} miembros · se resuelve con ${d.resolution_source}`}>
          <Legend items={[{ label: "Mediana", color: "var(--series-1)", kind: "line" }, { label: "p10–p90", color: "var(--series-1)", kind: "band" }]} />
          <LineChart x={d.years} height={260} yFormat={compact} marker={{ x: d.as_of, label: "origen" }}
                     series={[{ key: "p50", label: "Mediana", color: "var(--series-1)", values: d.p50, band: { lo: d.p10, hi: d.p90 } }]}
                     ariaLabel="Trayectoria pronosticada" />
          <table style={{ marginTop: 10 }}>
            <thead><tr><th>Año</th><th className="num">p10</th><th className="num">p50</th><th className="num">p90</th></tr></thead>
            <tbody>
              {d.years.map((y, i) => (
                <tr key={y}><td>{y}</td><td className="num">{num(d.p10[i], 3)}</td><td className="num">{num(d.p50[i], 3)}</td><td className="num">{num(d.p90[i], 3)}</td></tr>
              ))}
            </tbody>
          </table>
          <details style={{ marginTop: 10 }}>
            <summary>Defectos declarados ({d.caveats.length})</summary>
            <ul className="caveats">{d.caveats.map((c) => <li key={c}>{c}</li>)}</ul>
          </details>
        </Card>
      )}
    </>
  );
}
