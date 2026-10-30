import { geoContains } from "d3-geo";
import { feature } from "topojson-client";
import type { Topology, GeometryCollection } from "topojson-specification";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import countries from "i18n-iso-countries";
import es from "i18n-iso-countries/langs/es.json";
import world110 from "world-atlas/countries-110m.json";

countries.registerLocale(es);

// Natural Earth 1:110m (dominio público) vía world-atlas; los ids son ISO 3166-1 numéricos.
const topo = world110 as unknown as Topology<{ countries: GeometryCollection; land: GeometryCollection }>;
export const COUNTRIES = feature(topo, topo.objects.countries) as FeatureCollection<Geometry, { name: string }>;
export const LAND = feature(topo, topo.objects.land) as unknown as FeatureCollection;

export function countryFeatureByIso3(iso3: string): Feature | undefined {
  const n = countries.alpha3ToNumeric(iso3);
  return n ? COUNTRIES.features.find((f) => String(f.id) === n) : undefined;
}

/** iso numérico (como en world-atlas) -> ISO3. */
export function iso3OfFeature(f: Feature): string | undefined {
  return f.id === undefined ? undefined : countries.numericToAlpha3(String(f.id).padStart(3, "0"));
}

/** ISO3 del país de Natural Earth que contiene el punto, si lo hay. */
export function iso3AtPoint(lon: number, lat: number): string | undefined {
  const f = COUNTRIES.features.find((g) => geoContains(g, [lon, lat]));
  return f ? iso3OfFeature(f) : undefined;
}

export function countryName(iso3: string): string {
  return countries.getName(iso3, "es") ?? iso3;
}
