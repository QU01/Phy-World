"""``world4/models/posterior.npz``: el posterior de F7 que viaja con el producto (Studio no calibra).

Contiene, para el L0 regional (corte 2019) y, si se calculó, para el L1 en el origen vivo:

* las partículas equiponderadas del ABC-SMC (``theta``) y los nombres y cajas de las perillas;
* los pares de la discrepancia AR1 (no el GP: se reajusta al cargar, determinista con semilla 0);
* las varianzas declaradas (dato y discrepancia) y las marcas;
* metadatos: revisión del modelo, commit del Atlas, fecha, corte.

Deriva SÓLO de datos <= corte y de código del repo: no lleva datos de terceros (W12).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from world4 import home

# se lee del hogar si hay uno recalculado allí; si no, el que viaja con el paquete
_HOME_POST = home.home() / "models" / "posterior.npz"
PATH = _HOME_POST if _HOME_POST.is_file() else home.PACKAGE_MODELS / "posterior.npz"
WRITE_PATH = home.models() / "posterior.npz"


class PosteriorMissing(RuntimeError):
    pass


def _pack(prefix: str, post: dict, blobs: dict) -> dict:
    blobs[f"{prefix}/theta"] = np.asarray(post["theta"], float)
    meta = {k: post[k] for k in ("names", "box") if k in post}
    for k in ("cutoff", "start", "origin", "windows", "obs_sd", "p2_bands", "disc_bands", "marks", "explore"):
        if k in post:
            meta[k] = post[k]
    disc = {}
    for v, d in (post.get("discrepancy") or {}).items():
        for f in ("h", "model", "obs"):
            blobs[f"{prefix}/disc/{v}/{f}"] = np.asarray(d[f], float)
        disc[v] = d["mark"]
    meta["discrepancy_marks"] = disc
    meta["posterior_summary"] = post.get("posterior_summary")
    return meta


def save(l0r_post: dict, l1_post: dict | None = None, path: Path = WRITE_PATH) -> Path:
    from world4 import MODEL_REVISION
    from world4.data import atlas
    blobs = {}
    meta = {"model_revision": MODEL_REVISION, "atlas_commit": atlas.ATLAS_COMMIT, "created": date.today().isoformat(),
            "l0r": _pack("l0r", l0r_post, blobs)}
    if l1_post is not None:
        meta["l1"] = _pack("l1", l1_post, blobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, __meta__=np.frombuffer(json.dumps(meta).encode(), np.uint8), **blobs)
    return path


def _unpack(prefix: str, meta: dict, z) -> dict:
    out = dict(meta)
    out["theta"] = z[f"{prefix}/theta"]
    out["discrepancy"] = {v: {"h": z[f"{prefix}/disc/{v}/h"], "model": z[f"{prefix}/disc/{v}/model"],
                              "obs": z[f"{prefix}/disc/{v}/obs"], "mark": mk}
                          for v, mk in meta.get("discrepancy_marks", {}).items()}
    return out


def load(path: Path = PATH) -> dict:
    if not Path(path).is_file():
        raise PosteriorMissing(f"no posterior at {path}: python world4_cli.py --calibrate")
    z = np.load(path)
    meta = json.loads(bytes(z["__meta__"]).decode())
    out = {k: meta[k] for k in ("model_revision", "atlas_commit", "created")}
    out["l0r"] = _unpack("l0r", meta["l0r"], z)
    if "l1" in meta:
        out["l1"] = _unpack("l1", meta["l1"], z)
    return out


def draw_l0r(post: dict, n: int, rng, explore: bool = False) -> dict:
    """n juegos de perillas del L0: las identificadas del posterior; las no identificadas (``explore``) de su
    caja de previa uniforme si ``explore``, si no en su valor de F6 (1)."""
    th = post["theta"][rng.choice(len(post["theta"]), n, replace=len(post["theta"]) < n)]
    mult = {name: th[:, j] for j, name in enumerate(post["names"])}
    if explore:
        for name, (lo, hi, _) in post["explore"].items():
            mult[name] = rng.uniform(lo, hi, n)
    return mult
