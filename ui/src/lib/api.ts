// Cliente de la API de `python world4_cli.py --serve` (world4/server.py).
//
// Modo instantánea (``VITE_SNAPSHOT=1`` al compilar): en vez de la API lee ``data/<clave>.json`` publicados junto
// a la página (``ui/scripts/snapshot.py`` los escribe con la misma clave); lo que no está precalculado responde
// que necesita el servidor en vivo.

export const SNAPSHOT = import.meta.env.VITE_SNAPSHOT === "1";
const LIVE_REMEDY = "python world4_cli.py --serve";

/** Clave de archivo de una petición: ruta sin /api/, parámetros ordenados (igual que snapshot.py). */
export function snapshotKey(path: string, params: Record<string, string>): string {
  const base = path.replace(/^\/api\//, "").replace(/\//g, "_");
  const qs = Object.keys(params).sort().map((k) => `${k}-${params[k]}`).join("_");
  return (qs ? `${base}__${qs}` : base).replace(/[^A-Za-z0-9_.-]/g, "-");
}

async function fromSnapshot<T>(path: string, params: Record<string, string>, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`data/${snapshotKey(path, params)}.json`, { signal });
  if (!res.ok) {
    throw new ApiError(503, "Esta consulta no está en la instantánea: esta página muestra resultados ya calculados y " +
                       "para calcular otros hace falta el servidor de World4 en vivo.", LIVE_REMEDY);
  }
  const body = (await res.json()) as { __status?: number; error?: string; remedy?: string };
  if (body && typeof body === "object" && body.__status) throw new ApiError(body.__status, body.error ?? "", body.remedy);
  return body as T;
}

export class ApiError extends Error {
  status: number;
  remedy?: string;
  constructor(status: number, message: string, remedy?: string) {
    super(message);
    this.status = status;
    this.remedy = remedy;
  }
}

async function handle<T>(res: Response): Promise<T> {
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    throw new ApiError(res.status, text.slice(0, 300) || res.statusText);
  }
  if (!res.ok) {
    const b = (body ?? {}) as { error?: string; remedy?: string };
    throw new ApiError(res.status, b.error ?? res.statusText, b.remedy);
  }
  return body as T;
}

export async function getJSON<T>(path: string, params?: Record<string, string | number | boolean | undefined>,
                                 signal?: AbortSignal): Promise<T> {
  const q = new URLSearchParams();
  const flat: Record<string, string> = {};
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== "") {
      flat[k] = typeof v === "boolean" ? (v ? "1" : "0") : String(v);
      q.set(k, flat[k]);
    }
  }
  if (SNAPSHOT) return fromSnapshot<T>(path, flat, signal);
  const url = q.toString() ? `${path}?${q}` : path;
  let res: Response;
  try {
    res = await fetch(url, { signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "No se pudo contactar con el servidor de World4.",
                       "python world4_cli.py --serve");
  }
  return handle<T>(res);
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  if (SNAPSHOT) {
    throw new ApiError(503, "Evaluar una política se calcula en vivo: esta página es una instantánea.", LIVE_REMEDY);
  }
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify(body) });
  return handle<T>(res);
}

// ------------------------------------------------------------------ tipos
export type Capability = { ok: boolean; needs: string[]; missing: string[]; remedy: string[] };
export type Status = {
  model_revision: number;
  data: Record<string, boolean>;
  remedy: Record<string, string>;
  capabilities: Record<string, Capability>;
  ui_built: boolean;
};

export type Band = { p10: number[]; p50: number[]; p90: number[] };
export type LabelledBand = Band & { label: string; unit: string };

export type L0Result = {
  scenario: number;
  scenarios: Record<string, string>;
  years: number[];
  series: Record<string, number[]>;
  labels: Record<string, string>;
  units: Record<string, string>;
};

export type SocialResult = {
  scenario: number;
  scenarios: Record<string, string>;
  members: number;
  feedback: boolean;
  years: number[];
  world: Record<string, LabelledBand>;
  w3_years: number[];
  world3: Record<string, LabelledBand>;
  iso3: string[];
  country: Record<string, number[][]>;
  country_units: Record<string, string>;
  groups: Record<string, Record<string, Band>>;
};

export type Observable = { variable: string; unit: string; resolution_source: string; regions: string };
export type Catalog = { observables: Observable[]; regions: string[]; first_origin: number; last_year: number; horizon: number };
export type L1Path = Band & {
  variable: string; region: string; as_of: number; unit: string; years: number[]; members: number;
  caveats: string[]; resolution_source: string; params_cutoff: number;
};

export type Lever = { name: string; lo: number; hi: number; unit: string; what: string; target: string; sign: number };
export type Levers = { levers: Lever[]; narratives: string[]; default_narrative: string };
export type PolicyResult = {
  theta: Record<string, number>;
  narrative: string;
  members: number;
  objectives: Record<string, number>;
  constraints: Record<string, number>;
  feasible: boolean;
  baseline_objectives: Record<string, number>;
  baseline_constraints: Record<string, number>;
  percentiles: Record<string, Record<string, number>>;
  better_than_baseline_share: Record<string, number>;
  units: Record<string, string>;
};
export type L0RResult = {
  years: number[];
  world: Record<string, Band & { unit: string }>;
  members: number;
  posterior_cutoff: number;
  band: string;
};

export type ConflictMeta = {
  first_origin: string;
  last_month: string;
  last_final_month: string;
  saved_origins: string[];
  versions: string[];
  country_names: Record<string, string>;
  country_bbox: Record<string, [number, number, number, number]>;
  country_point: Record<string, [number, number]>;
  rules: Record<string, string>;
};

export type Outcome = "any" | "sb" | "ns" | "os";
export type ConflictCells = {
  gid: number[];
  lon: number[];
  lat: number[];
  country_gw: number[];
  events_last12: number[];
  deaths_last12: number[];
  deaths_sb_expected: number[];
  deaths_sb_p50: number[][];
  deaths_sb_p90: number[][];
  driver_background: number[];
  driver_own: number[];
  driver_neighbours: number[];
  driver_country: number[];
} & Record<`p_window_${Outcome}`, number[]>
  & Record<`p_month_${Outcome}`, number[][]>
  & Record<`events_expected_${"sb" | "ns" | "os"}`, number[]>;

export type ConflictPayload = {
  schema: string;
  origin: string;
  months: string[];
  data: { ged_version: string; candidate_version: string; last_final_month: string; last_month: string };
  cells: ConflictCells;
  countries: { gw: number; events_expected: number; deaths_sb_expected: number; cells_p_any_over_50: number; max_p_any: number }[];
  summary: { cells_returned: number; min_p: number; expected_events_window: Record<string, number>;
             expected_deaths_sb_window: number; branching_ratio: Record<string, Record<string, number>> };
  validation: null | {
    origins: string[];
    any_event_aupr: Record<string, number>;
    any_event_brier: Record<string, number>;
    sb_deaths_crps_africa_me: Record<string, number>;
  };
  caveats: string[];
  sources: string[];
};

export type ConflictCell = {
  gid: number; lon: number; lat: number; country_gw: number; country: string; origin: string;
  months: string[]; events: Record<string, number[]>; deaths: Record<string, number[]>;
  events_total_since_1989: Record<string, number>; neighbours_events_last12: number;
};

type Scores = { brier?: number; log_loss?: number; auroc?: number; aupr?: number; crps?: number; events?: number; expected?: number };
export type ConflictValidation = {
  meta: { origins: string[]; run_date: string; horizon: number; samples: number };
  summary: Record<string, Record<string, Record<string, { all_h: Scores; h1: Scores; "h4-12": Scores; "2026_candidate": Scores }>>>;
  window12: Record<string, Record<string, Record<string, Scores>>>;
  calibration: Record<string, [number, number, number][]>;
  hawkes: Record<string, { branching: Record<string, Record<string, number>>; theta: Record<string, number> }>;
  aupr_by_origin: Record<string, Record<string, Record<string, number>>>;
};

export type SourceEntry = {
  name: string; license: string | null; mark: string; redistributable: boolean | null; vintage?: string;
  lag_years?: number; used_in?: string[]; note?: string;
};
export type Sources = { sources: Record<string, SourceEntry> };
