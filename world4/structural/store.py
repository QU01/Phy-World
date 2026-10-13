"""Almacén de ensembles del L1: un pronóstico por origen (1970-2018) más el origen vivo 2019->2030.

Lo usan el generador de preguntas históricas (W8, W9a), los recalibradores y el estimador vivo.
Se construye una vez (~1-3 min con procesos 'spawn') y vive en ``data_cache/structural/`` junto
a los datos del Atlas: deriva de ellos y no se versiona.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from world4 import MODEL_REVISION
from world4.structural import engine

ORIGINS = tuple(range(1970, engine.L1_LAST_DATA_YEAR))       # 1970..2018
LIVE_ORIGIN = engine.L1_LAST_DATA_YEAR
R_DEFAULT = 32
STORE_VERSION = 1
ARRAYS = ("conflict_real", "hazard", "coup", "crisis", "demo", "ly", "debt")


def path() -> Path:
    from world4.data import atlas
    return atlas.root().parent / "structural" / f"store_v{STORE_VERSION}.npz"


def _one(args):
    origin, R, live = args
    return engine.ensemble(origin, R=R, live=live)


def build(R: int = R_DEFAULT, workers: int = 6, log=print) -> Path:
    t = time.time()
    jobs = [(o, R, False) for o in ORIGINS] + [(LIVE_ORIGIN, R, True)]
    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(_one, jobs)
    blobs, meta = {}, {"model_revision": MODEL_REVISION, "R": R, "k_max": engine.K_MAX, "origins": {}}
    from world4.data import atlas
    meta["atlas_commit"] = atlas.ATLAS_COMMIT
    for r in results:
        tag = "live" if r["live"] else str(r["origin"])
        for k in ARRAYS:
            blobs[f"{tag}/{k}"] = r[k]
        meta["origins"][tag] = {"origin": r["origin"], "params_cutoff": r["params_cutoff"], "seed": r["seed"]}
    meta["seconds"] = time.time() - t
    out = path()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, __meta__=np.frombuffer(json.dumps(meta).encode(), np.uint8), **blobs)
    load.cache_clear()
    log(f"store: {len(results)} ensembles in {meta['seconds']:.0f} s -> {out}")
    return out


class StoreMissing(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load() -> dict:
    p = path()
    if not p.is_file():
        raise StoreMissing(f"no structural store at {p}: python world4_cli.py --build-store")
    z = np.load(p)
    meta = json.loads(bytes(z["__meta__"]).decode())
    out = {"meta": meta, "ens": {}}
    for tag, m in meta["origins"].items():
        out["ens"][tag] = {**m, "live": tag == "live", **{k: z[f"{tag}/{k}"] for k in ARRAYS}}
    return out


def get(origin: int, live: bool = False) -> dict:
    s = load()
    tag = "live" if live else str(origin)
    if tag not in s["ens"]:
        raise KeyError(f"origin {origin} not in the store")
    return s["ens"][tag]
