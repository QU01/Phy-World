import { useEffect, useState } from "react";
import { getJSON, SNAPSHOT, type Status } from "./lib/api";
import { useApi } from "./lib/useApi";
import { ConflictView } from "./views/ConflictView";
import { SocialView } from "./views/SocialView";
import { World3View } from "./views/World3View";
import { CountryView } from "./views/CountryView";
import { PolicyView } from "./views/PolicyView";
import { DataView } from "./views/DataView";
import { ErrorNotice } from "./components/common";

const VIEWS = [
  { id: "conflicto", label: "Conflicto por celdas", hint: "F11 · 64 818 celdas, 12 meses", cap: "conflict" },
  { id: "sociedad", label: "Sociedad ↔ World3", hint: "180 países a 2100", cap: "social" },
  { id: "world3", label: "World3", hint: "L0 · 5 escenarios", cap: "l0" },
  { id: "paises", label: "Países", hint: "L1 · 1-10 años", cap: "l1" },
  { id: "politicas", label: "Políticas y regiones", hint: "F6 · F8", cap: "policy" },
  { id: "datos", label: "Datos y fuentes", hint: "instalados, licencias", cap: null },
] as const;
type ViewId = (typeof VIEWS)[number]["id"];

function readHash(): ViewId {
  const h = window.location.hash.replace(/^#\/?/, "");  // #conflicto (y el antiguo #/conflicto)
  return (VIEWS.find((v) => v.id === h)?.id ?? "conflicto") as ViewId;
}

function storedTheme(): string {
  try {
    return localStorage.getItem("w4-theme") ?? "auto";
  } catch {
    return "auto";
  }
}

export function App() {
  const [view, setView] = useState<ViewId>(readHash);
  const [theme, setTheme] = useState<string>(storedTheme);
  const status = useApi((s) => getJSON<Status>("/api/status", undefined, s), []);
  useEffect(() => {
    const on = () => setView(readHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  useEffect(() => {
    if (SNAPSHOT) return;                     // publicada: el tema lo pone el visor (data-theme), no la página
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("w4-theme", theme);
    } catch {
      /* almacenamiento no disponible: el tema vale para esta visita */
    }
  }, [theme]);
  // el canvas del mapa lee los colores del tema: un cambio de esquema del sistema también debe repintar
  const [scheme, setScheme] = useState(() => window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "d" : "l");
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const on = () => setScheme(mq.matches ? "d" : "l");
    mq?.addEventListener("change", on);
    return () => mq?.removeEventListener("change", on);
  }, []);
  const themeKey = `${theme}-${scheme}`;
  const caps = status.data?.capabilities;
  return (
    <div className="app">
      <nav className="nav" aria-label="Vistas">
        <div className="brand">
          <b>Phy-World · World4</b>
          <span>QUASAR · modelo integrado del sistema-mundo</span>
        </div>
        {VIEWS.map((v) => (
          <button key={v.id} aria-current={view === v.id ? "page" : undefined}
                  onClick={() => { window.location.hash = v.id; }}>
            <span>{v.label}{v.cap && caps && !caps[v.cap]?.ok ? " ⚠" : ""}</span>
            <small>{v.hint}</small>
          </button>
        ))}
        <div className="spacer" />
        <div className="foot">
          {!SNAPSHOT && <label className="field">
            <span>Tema</span>
            <select value={theme} onChange={(e) => setTheme(e.target.value)}>
              <option value="auto">Del sistema</option>
              <option value="light">Claro</option>
              <option value="dark">Oscuro</option>
            </select>
          </label>}
          {status.data && <span>Revisión del modelo {status.data.model_revision}</span>}
          <span>Exploratorio salvo donde se indica el horizonte medido.</span>
        </div>
      </nav>
      <main className="main">
        {SNAPSHOT && (
          <div className="notice">
            <div className="title">Instantánea · resultados ya calculados</div>
            <div className="secondary small">
              Pronóstico por celdas con origen en septiembre de 2026, validación, World3, Sociedad ↔ World3 y una selección
              de consultas del L1. Cambiar el origen, otras consultas y evaluar políticas se calculan en vivo con{" "}
              <code>python world4_cli.py --serve</code>.
            </div>
          </div>
        )}
        {status.error && <ErrorNotice error={status.error} what="conectar con World4" />}
        {view === "conflicto" && <ConflictView theme={themeKey} />}
        {view === "sociedad" && <SocialView />}
        {view === "world3" && <World3View />}
        {view === "paises" && <CountryView />}
        {view === "politicas" && <PolicyView />}
        {view === "datos" && <DataView status={status.data} />}
      </main>
    </div>
  );
}
