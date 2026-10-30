// Color por función (dataviz): categórico para los tipos de violencia (slots 1-3, validados con todos los
// pares en claro y oscuro), secuencial de un solo tono para el riesgo (rampa naranja de 7 pasos, validada
// como ordinal en claro y oscuro: --r0 ... --r6 en theme.css).

export const TYPE_LABEL: Record<string, string> = {
  any: "Cualquier violencia",
  sb: "Estatal",
  ns: "No estatal",
  os: "Unilateral (contra civiles)",
};
export const TYPE_COLOR: Record<string, string> = {
  sb: "var(--series-1)",
  ns: "var(--series-2)",
  os: "var(--series-3)",
  any: "var(--text-primary)",
};
export const TYPE_ORDER = ["sb", "ns", "os"] as const;
export const TYPE_SHORT: Record<string, string> = { any: "Cualquiera", sb: "Estatal", ns: "No estatal", os: "Contra civiles" };

export const RAMP_VARS = ["--r0", "--r1", "--r2", "--r3", "--r4", "--r5", "--r6"];

/** Lee los colores efectivos de la rampa (cambian con el tema) para pintar en canvas. */
export function readRamp(): string[] {
  const cs = getComputedStyle(document.documentElement);
  return RAMP_VARS.map((v) => cs.getPropertyValue(v).trim());
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Clases de probabilidad: eventos raros, así que la escala es casi logarítmica. */
export const PROB_BREAKS = [0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8];
export const PROB_LABELS = ["1 %", "3 %", "10 %", "20 %", "40 %", "60 %", "80 %"];

export function classOf(v: number, breaks: number[]): number {
  if (!(v >= breaks[0])) return -1;
  let k = 0;
  while (k + 1 < breaks.length && v >= breaks[k + 1]) k++;
  return k;
}

/** Cortes "bonitos" para magnitudes (eventos, muertes) con 7 clases desde ``lo``. */
export function magnitudeBreaks(values: number[], lo: number): number[] {
  const vs = values.filter((v) => v >= lo).sort((a, b) => a - b);
  if (vs.length === 0) return [lo, lo * 3, lo * 10, lo * 30, lo * 100, lo * 300, lo * 1000];
  const qs = [0, 0.3, 0.55, 0.75, 0.88, 0.95, 0.985].map((q) => vs[Math.min(vs.length - 1, Math.floor(q * vs.length))]);
  const nice = (x: number) => {
    const p = Math.pow(10, Math.floor(Math.log10(x)));
    const m = x / p;
    return (m < 1.5 ? 1 : m < 2.5 ? 2 : m < 4 ? 3 : m < 7.5 ? 5 : 10) * p;
  };
  const out: number[] = [];
  for (const q of qs) {
    const n = Math.max(nice(Math.max(q, lo)), lo);
    if (!out.length || n > out[out.length - 1]) out.push(n);
  }
  while (out.length < 7) out.push(out[out.length - 1] * 3);
  return out;
}
