const nf = (d: number) => new Intl.NumberFormat("es-ES", { maximumFractionDigits: d, minimumFractionDigits: 0 });
const cache = new Map<number, Intl.NumberFormat>();

export function num(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  let f = cache.get(digits);
  if (!f) {
    f = nf(digits);
    cache.set(digits, f);
  }
  return f.format(x);
}

/** Número compacto con sufijo (mil, M, mil M) para ejes y estadísticas. */
export function compact(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  const a = Math.abs(x);
  if (a >= 1e12) return `${num(x / 1e12, 1)} B`;
  if (a >= 1e9) return `${num(x / 1e9, 1)} mil M`;
  if (a >= 1e6) return `${num(x / 1e6, 1)} M`;
  if (a >= 1e4) return `${num(x / 1e3, 0)} mil`;
  if (a >= 100) return num(x, 0);
  if (a >= 1) return num(x, 2);
  return num(x, 3);
}

export function pct(p: number | null | undefined, digits?: number): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "–";
  const v = p * 100;
  const d = digits ?? (v < 1 ? 2 : v < 10 ? 1 : 0);
  return `${num(v, d)} %`;
}

/** Formateador de porcentajes para un eje cuyo máximo es ``max`` (más decimales si el rango es pequeño). */
export function pctAxis(max: number): (v: number) => string {
  const d = max <= 0.02 ? 2 : max <= 0.1 ? 1 : 0;
  return (v: number) => pct(v, d);
}

export function monthName(label: string): string {
  const [y, m] = label.split("-").map(Number);
  const names = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];
  return `${names[m - 1]} ${y}`;
}

export function latlon(lat: number, lon: number): string {
  return `${num(Math.abs(lat), 2)}° ${lat >= 0 ? "N" : "S"}, ${num(Math.abs(lon), 2)}° ${lon >= 0 ? "E" : "O"}`;
}
