"""Merge the country-year source panels (data/processed/sources/*.csv) into the model panel.

Output: data/processed/panel_full.csv  (panel.csv + every source variable, left-joined on
iso3/year; name collisions get a "<source>__" prefix) and data/processed/world_full.csv
(world-level series from *_world.csv files, prefixed by source).
Run the source builders first:  python -c "from sources import X; X.build()".
"""
from pathlib import Path

import pandas as pd
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
SRC = ROOT / "data" / "processed" / "sources"
SOURCES = ["acciones", "elites", "finanzas", "comercio", "recursos", "geopolitica", "demografia", "clima"]


def main():
    p = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    added = {}
    for s in SOURCES:
        f = SRC / f"{s}.csv"
        if not f.exists():
            print("missing", s)
            continue
        d = pd.read_csv(f)
        d = d[d.year.between(1950, 2019)]
        d = d.groupby(["iso3", "year"], as_index=False).first()
        d = d[[c for c in d.columns if c in ("iso3", "year") or pd.api.types.is_numeric_dtype(d[c])]]
        ren = {c: f"{s}__{c}" for c in d.columns if c not in ("iso3", "year") and c in p.columns}
        d = d.rename(columns=ren)
        p = p.merge(d, on=["iso3", "year"], how="left")
        added[s] = [c for c in d.columns if c not in ("iso3", "year")]
    p.to_csv(ROOT / "data" / "processed" / "panel_full.csv", index=False)
    w = None
    for s in SOURCES:
        f = SRC / f"{s}_world.csv"
        if f.exists():
            d = pd.read_csv(f)
            d = d.rename(columns={c: f"{s}__{c}" for c in d.columns if c != "year"})
            w = d if w is None else w.merge(d, on="year", how="outer")
    if w is not None:
        w.sort_values("year").to_csv(ROOT / "data" / "processed" / "world_full.csv", index=False)
    print("panel_full", p.shape, {k: len(v) for k, v in added.items()})
    return added


if __name__ == "__main__":
    main()
