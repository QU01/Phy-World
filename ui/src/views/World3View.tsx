import { useState } from "react";
import { getJSON, type L0Result } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, ErrorNotice, Field, Legend, Loading } from "../components/common";
import { LineChart } from "../components/charts";
import { compact } from "../lib/format";

const KEYS = ["pop", "le", "fpc", "iopc", "sopc", "nrfr", "ppolx", "al", "hwi", "hef", "cbr", "cdr"];

export function World3View() {
  const [scenario, setScenario] = useState(2);
  const r = useApi((s) => getJSON<L0Result>("/api/l0", { scenario }, s), [scenario]);
  const base = useApi((s) => getJSON<L0Result>("/api/l0", { scenario: 1 }, s), []);
  const d = r.data;
  const b = base.data;
  const compare = scenario !== 1 && b;
  return (
    <>
      <div className="viewhead">
        <h2>World3-03 (L0)</h2>
        <p>
          El núcleo biofísico: World3-03 reimplementado en NumPy y anclado a Vensim (error máximo 6,9e-6). Los diez
          escenarios de <i>Limits to Growth: The 30-Year Update</i>. Son modos de comportamiento del modelo de 2004, no
          pronósticos.
        </p>
      </div>
      <div className="controls card" style={{ padding: "10px 14px" }}>
        <Field label="Escenario">
          <select value={scenario} onChange={(e) => setScenario(+e.target.value)}>
            {Object.entries(d?.scenarios ?? b?.scenarios ?? {}).map(([k, v]) => (
              <option key={k} value={k}>{k}. {v}</option>
            ))}
          </select>
        </Field>
        {compare && (
          <Legend items={[{ label: `Escenario ${scenario}`, color: "var(--series-1)", kind: "line" },
                          { label: "Escenario 1 (referencia)", color: "var(--ref)", kind: "dash" }]} />
        )}
      </div>
      {r.error && <ErrorNotice error={r.error} />}
      {!d && !r.error && <Loading what="Corriendo World3" />}
      {d && (
        <div className="grid3">
          {KEYS.filter((k) => d.series[k]).map((k) => (
            <Card key={k} title={d.labels[k]} sub={d.units[k]}>
              <LineChart x={d.years} height={160} yFormat={compact} marker={{ x: 2026, label: "hoy" }}
                         series={[
                           ...(compare && b!.series[k] ? [{ key: "ref", label: "Escenario 1", color: "var(--ref)", dash: true, values: b!.series[k] }] : []),
                           { key: k, label: `Escenario ${scenario}`, color: "var(--series-1)", values: d.series[k] },
                         ]} ariaLabel={d.labels[k]} />
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
