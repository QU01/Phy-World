import { getJSON, type Sources, type Status } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, ErrorNotice, Loading } from "../components/common";

const DATA_ES: Record<string, string> = {
  atlas: "Atlas del Sistema-Mundo (panel de 180 países)", wpp: "UN WPP 2024 por edad y sexo (F1)",
  energy: "Energía por país y fuente, OWID (F3)", climate: "FaIR: calibración 1.6.0 y HadCRUT5 (F4)",
  food: "FAOSTAT y agua del WDI (F5)", conflict_grid: "UCDP GED + Candidate y PRIO-GRID (F11)",
};
const CAP_ES: Record<string, string> = {
  conflict: "Conflicto por celdas", social: "Sociedad ↔ World3", l0: "World3-03", l1: "Países (L1)",
  l0r: "L0 regional (F6)", policy: "Políticas (F8)",
};

export function DataView({ status }: { status?: Status }) {
  const src = useApi((s) => getJSON<Sources>("/api/sources", undefined, s), []);
  return (
    <>
      <div className="viewhead">
        <h2>Datos y fuentes</h2>
        <p>
          World4 no incluye datos de terceros: cada conjunto se instala con su comando y se comprueba por SHA-256. Las
          vistas que necesitan algo que falta lo dicen, con el comando que lo instala.
        </p>
      </div>
      {!status && <Loading what="Leyendo el estado" />}
      {status && (
        <div className="grid2" style={{ alignItems: "start" }}>
          <Card title="Conjuntos de datos" sub={`Revisión del modelo ${status.model_revision}`}>
            <table>
              <tbody>
                {Object.entries(status.data).map(([k, ok]) => (
                  <tr key={k}>
                    <td>{DATA_ES[k] ?? k}</td>
                    <td><span className={`pill ${ok ? "ok" : "no"}`}>{ok ? "instalado" : "falta"}</span></td>
                    <td>{!ok && <code>{status.remedy[k]}</code>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <Card title="Vistas">
            <table>
              <tbody>
                {Object.entries(status.capabilities).map(([k, c]) => (
                  <tr key={k}>
                    <td>{CAP_ES[k] ?? k}</td>
                    <td><span className={`pill ${c.ok ? "ok" : "no"}`}>{c.ok ? "disponible" : "falta " + c.missing.join(", ")}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
      <Card title="Registro de fuentes (W12)" sub="Licencia, marca de verificación ([V] leída, [S] supuesta, [?] desconocida) y si se puede redistribuir">
        {src.error && <ErrorNotice error={src.error} />}
        {src.loading && <Loading what="Leyendo sources.json" />}
        {src.data && (
          <div className="scroll">
            <table>
              <thead><tr><th>Fuente</th><th>Licencia</th><th>Marca</th><th>Redistribuible</th><th>Versión</th><th>Se usa en</th></tr></thead>
              <tbody>
                {Object.entries(src.data.sources).map(([k, s]) => (
                  <tr key={k}>
                    <td>{s.name}</td><td>{s.license ?? "desconocida"}</td><td>{s.mark}</td>
                    <td>{s.redistributable === null ? "?" : s.redistributable ? "sí" : "no"}</td>
                    <td className="small">{s.vintage}</td><td className="small">{(s.used_in ?? []).join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}
