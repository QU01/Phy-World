"""Decode the Natural Earth 110m TopoJSON (world-atlas) into compact GeoJSON features."""
import json
from pathlib import Path
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT


def load_countries(path=ROOT / "data" / "raw" / "countries-110m.json", digits=2):
    topo = json.load(open(path))
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append([round(x * sx + tx, digits), round(y * sy + ty, digits)])
        arcs.append(pts)

    def ring(idx):
        out = []
        for i in idx:
            pts = arcs[i] if i >= 0 else arcs[~i][::-1]
            out.extend(pts if not out else pts[1:])
        return out

    feats = []
    for g in topo["objects"]["countries"]["geometries"]:
        if g["type"] == "Polygon":
            coords = [ring(r) for r in g["arcs"]]
        elif g["type"] == "MultiPolygon":
            coords = [[ring(r) for r in poly] for poly in g["arcs"]]
        else:
            continue
        feats.append({"type": "Feature", "id": int(g["id"]) if g.get("id", "").isdigit() else -1,
                      "properties": {"name": g["properties"]["name"]},
                      "geometry": {"type": g["type"], "coordinates": coords}})
    return {"type": "FeatureCollection", "features": feats}


if __name__ == "__main__":
    fc = load_countries()
    print(len(fc["features"]), len(json.dumps(fc, separators=(",", ":"))))
