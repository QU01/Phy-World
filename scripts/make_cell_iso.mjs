// País ISO 3166-1 numérico de cada celda PRIO-GRID 2.0, por geometría: el polígono de Natural Earth 1:50m
// (dominio público, world-atlas 2.0.2) que contiene el centro de la celda; si ninguno lo contiene (costas), el de
// la esquina o el punto medio de un lado que sí caiga en uno. Lo usa world4.conflict.country_panel para sumar los
// eventos de UCDP GED por país del Atlas (que usa ISO3), en vez de por código Gleditsch-Ward.
//
//   cd ui && npm install && cd .. && node scripts/make_cell_iso.mjs <gid.json> <out.json>
import fs from "fs";
import { createRequire } from "module";

const require = createRequire(new URL("../ui/package.json", import.meta.url));
const d3 = require("d3-geo");
const topo = require("topojson-client");
const w50 = require("world-atlas/countries-50m.json");

const [gidFile, outFile] = process.argv.slice(2);
const gids = JSON.parse(fs.readFileSync(gidFile, "utf8"));
const feats = topo.feature(w50, w50.objects.countries).features.filter((f) => f.id !== undefined);
const boxes = feats.map((f) => d3.geoBounds(f));
const inBox = (b, lon, lat) => lat >= b[0][1] - 0.01 && lat <= b[1][1] + 0.01 &&
  (b[0][0] <= b[1][0] ? lon >= b[0][0] - 0.01 && lon <= b[1][0] + 0.01 : lon >= b[0][0] - 0.01 || lon <= b[1][0] + 0.01);
const find = (lon, lat) => {
  for (let k = 0; k < feats.length; k++) if (inBox(boxes[k], lon, lat) && d3.geoContains(feats[k], [lon, lat])) return +feats[k].id;
  return 0;
};
const out = new Array(gids.length).fill(0);
gids.forEach((g, i) => {
  const row = Math.floor((g - 1) / 720) + 1, col = ((g - 1) % 720) + 1;
  const lon = -180 + (col - 0.5) * 0.5, lat = -90 + (row - 0.5) * 0.5;
  const tries = [[0, 0], [-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2], [0, -0.24], [0, 0.24], [-0.24, 0], [0.24, 0]];
  for (const [dx, dy] of tries) {
    const v = find(lon + dx, lat + dy);
    if (v) { out[i] = v; break; }
  }
});
fs.writeFileSync(outFile, JSON.stringify({ source: "Natural Earth 1:50m via world-atlas 2.0.2 (public domain)", gid: gids, isonum: out }));
console.log("cells", gids.length, "assigned", out.filter((v) => v).length);
