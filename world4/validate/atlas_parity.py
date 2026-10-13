"""Sonda de paridad W2: el L1 portado debe dar EXACTAMENTE lo mismo que el Atlas original.

``probe(model, calibrate, validate)`` recibe los tres módulos (los originales del clon o los
portados) y ejecuta los caminos que usan la validación y la calibración: corrida histórica,
pronóstico desde 1990 con la regla de conducta estimada, con política congelada y con una
ablación, las referencias de comparación y la pérdida de calibración en el corte de 1990.

Cada array se resume en su SHA-256 (bit a bit) más tres estadísticos. En la misma plataforma
se exige el hash; en otra (BLAS distinto) se cae a los estadísticos con rtol 1e-10 y se dice.
"""
from __future__ import annotations

import hashlib
import json
import platform

import numpy as np

STAT_RTOL = 1e-10


def fingerprint(a) -> dict:
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    canon = np.where(np.isnan(a), np.nan, a)          # un único patrón de bits para NaN
    finite = a[np.isfinite(a)]
    return {
        "sha256": hashlib.sha256(canon.tobytes() + str(a.shape).encode()).hexdigest(),
        "shape": list(a.shape),
        "stats": [float(finite.sum()) if finite.size else 0.0,
                  float(np.abs(finite).sum()) if finite.size else 0.0,
                  int(np.isnan(a).sum())],
    }


def platform_id() -> str:
    return f"{platform.system()}-{platform.machine()}-numpy{np.__version__}"


def probe(model, calibrate, validate) -> dict:
    out: dict[str, np.ndarray] = {}
    W = calibrate.W
    d = json.load(open(validate.RES / "calibration.json"))["params"]
    P = model.Params().with_vector(list(d), list(d.values()))

    o = model.Simulator(W, P, R=4, seed=3).run(0, W.T)
    out.update({f"hist/{k}": v for k, v in o.items()})

    p90, cut = validate.params_for(1990)
    assert cut == 1990, cut
    variants = {
        "bc1990": {},
        "frozen1990": {"policy": "frozen"},
        "sinclima1990": {"overrides": validate.ABLATIONS["sin_clima_ecologia"]},
    }
    for tag, kw in variants.items():
        f = validate.forecast(p90, 40, R=4, seed=40, **kw)
        out.update({f"{tag}/{k}": v for k, v in f.items()})

    for h in (1, 5, 10):
        out.update({f"baseline_h{h}/{k}": v for k, v in validate.baselines(40, h).items()})
        out.update({f"conflict_baseline_h{h}/{k}": v
                    for k, v in validate.conflict_baselines(40, h).items()})

    # Pérdida de calibración con los parámetros libres del corte 1990 (muta estado global: al final)
    roll = json.load(open(validate.ROLL / "params_1990.json"))
    calibrate.set_cutoff(40)
    det = calibrate.loss([roll["libres"][n] for n in calibrate.NAMES], R=4, seed=1, detail=True)
    out["loss1990"] = np.array([det[k] for k in sorted(det)])
    return out


def fingerprints(arrays: dict) -> dict:
    return {"platform": platform_id(), "arrays": {k: fingerprint(v) for k, v in sorted(arrays.items())}}


def compare(golden: dict, current: dict) -> dict:
    """{'mode', 'exact', 'failed': [...], 'missing': [...]} comparando huellas."""
    same_platform = golden["platform"] == current["platform"]
    failed, missing = [], []
    for k, g in golden["arrays"].items():
        c = current["arrays"].get(k)
        if c is None:
            missing.append(k)
            continue
        if same_platform:
            if g["sha256"] != c["sha256"]:
                failed.append(k)
        else:
            ok = g["shape"] == c["shape"] and g["stats"][2] == c["stats"][2] and all(
                abs(x - y) <= STAT_RTOL * max(abs(x), 1e-300) for x, y in zip(g["stats"][:2], c["stats"][:2]))
            if not ok:
                failed.append(k)
    extra = sorted(set(current["arrays"]) - set(golden["arrays"]))
    return {"mode": "sha256" if same_platform else f"stats rtol {STAT_RTOL:g}",
            "n": len(golden["arrays"]), "failed": failed, "missing": missing, "extra": extra,
            "exact": same_platform and not failed and not missing}
