"""Instantánea de la API para publicar la interfaz sin servidor (``VITE_SNAPSHOT=1``).

Llama a las rutas de ``world4.server`` con los parámetros que la interfaz pide por omisión y escribe cada
respuesta en ``<out>/data/<clave>.json``, con la misma clave que ``snapshotKey`` de ``src/lib/api.ts``. Una ruta
que necesita datos que faltan se guarda como ``{"__status": 503, "error", "remedy"}``.

    .venv/bin/python ui/scripts/snapshot.py <out>
"""
import json
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from world4 import server as S  # noqa: E402

L1 = [("gdp_pc", "COL"), ("conflict_hazard", "COL"), ("democracy", "COL"), ("gdp_pc", "MEX"),
      ("conflict_hazard", "NGA"), ("population", "NGA"), ("debt_gdp", "ARG"), ("gdp_pc", "CHN")]
SOCIAL = [(1, "1"), (1, "0"), (2, "1"), (3, "1"), (5, "1")]
N_CELLS = 200


def key(path: str, params: dict) -> str:
    base = path.removeprefix("/api/").replace("/", "_")
    qs = "_".join(f"{k}-{params[k]}" for k in sorted(params))
    return re.sub(r"[^A-Za-z0-9_.-]", "-", f"{base}__{qs}" if qs else base)


def call(path: str, params: dict | None = None):
    params = {k: str(v) for k, v in (params or {}).items()}
    try:
        code, payload = S.route("GET", path, {k: [v] for k, v in params.items()}, None)
        if isinstance(payload, bytes):
            import gzip
            payload = json.loads(gzip.decompress(payload))
    except S.DataMissing as exc:
        code, payload = 503, {"__status": 503, "error": str(exc), "remedy": exc.remedy}
    return key(path, params), code, payload


def main(out: Path) -> None:
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    reqs = [("/api/status", {}), ("/api/sources", {}), ("/api/catalog", {}), ("/api/levers", {}),
            ("/api/l0r", {"posterior": "1", "members": "100"}), ("/api/conflict/meta", {}),
            ("/api/conflict/validation", {})]
    reqs += [("/api/l0", {"scenario": s}) for s in range(1, 6)]
    reqs += [("/api/social", {"scenario": s, "feedback": f, "n": 40}) for s, f in SOCIAL]
    reqs += [("/api/l1", {"variable": v, "region": r, "as_of": 2009, "members": 32}) for v, r in L1]
    reqs += [("/api/density", {"origin": 2025, "h": h, "cells": "1"}) for h in range(1, 6)]
    reqs += [("/api/density/study", {"lens": lens}) for lens in ("nb", "ppml")]
    reqs.append(("/api/density/scenario", {"origin": 2019, "members": 32}))
    meta = S.conflict_meta()
    origin = meta["last_month"]
    reqs.append(("/api/conflict/forecast", {"origin": origin}))
    for path, params in reqs:
        k, code, payload = call(path, params)
        (data / f"{k}.json").write_text(json.dumps(S._clean(payload), ensure_ascii=False, default=S._jsonable,
                                                   separators=(",", ":")), encoding="utf-8")
        print(f"{code} {k}")
    fc = json.loads((data / f"{key('/api/conflict/forecast', {'origin': origin})}.json").read_text(encoding="utf-8"))
    for gid in fc["cells"]["gid"][:N_CELLS]:
        k, code, payload = call("/api/conflict/cell", {"gid": gid, "origin": origin})
        (data / f"{k}.json").write_text(json.dumps(S._clean(payload), ensure_ascii=False, separators=(",", ":")),
                                        encoding="utf-8")
    print(f"{N_CELLS} cells")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
