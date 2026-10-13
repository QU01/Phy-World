"""W10: rendimiento en la máquina de referencia (Ryzen 7 5700X).

* L0: una corrida del L0 regional de 2019 a 2100 (81 años, 112 stocks), mediana de 50 corridas, < 10 ms.
  Se informa también World3-03 (n = 1 y por miembro con n = 1000).
* L1: 64 miembros de 2019 a 2100 con el modelo vigente (F1-F5), < 60 s. Se mide en un proceso y repartido
  en 8 procesos de 8 miembros (los 8 núcleos de la máquina de referencia; semillas distintas por proceso:
  es el mismo ensemble de 64). Cuenta la segunda, y se informan las dos.
"""
from __future__ import annotations

import platform
import time

import numpy as np

L0_MS, L1_S = 10.0, 60.0


def cpu() -> str:
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out or platform.processor()
    except Exception:  # noqa: BLE001
        return platform.processor()


def _chunk(seed):
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.core.l1_atlas import model as amodel
    W2100 = amodel.build_world(horizon=2100)
    f = coupling.forecast(69, R=8, seed=seed, world=W2100, horizon=81, f5=coupling.F5Config())
    return float(f["Tg"][:, -1].mean())


def l1_parallel(workers: int = 8) -> float:
    from multiprocessing import get_context
    t = time.perf_counter()
    with get_context("spawn").Pool(workers) as pool:
        pool.map(_chunk, [69 + 1000 * k for k in range(workers)])
    return time.perf_counter() - t


def run(l1: bool = True, log=print) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import l0
    from world4.core.l0r import model
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    m = model.L0R(model.calibrate(W, 2019), model.initial(W, 2019))
    m.run(81)
    ts = []
    for _ in range(50):
        t = time.perf_counter()
        m.run(81)
        ts.append(time.perf_counter() - t)
    t = time.perf_counter()
    l0.run()
    w3 = time.perf_counter() - t
    t = time.perf_counter()
    l0.run(n=1000)
    w3n = (time.perf_counter() - t) / 1000
    out = {"cpu": cpu(), "l0r_ms_median": float(np.median(ts) * 1e3), "l0r_ms_p90": float(np.percentile(ts, 90) * 1e3),
           "world3_ms_single": w3 * 1e3, "world3_ms_per_member_n1000": w3n * 1e3}
    out["l0_ok"] = out["l0r_ms_median"] < L0_MS
    if l1:
        from world4.core import coupling
        from world4.core.l1_atlas import model as amodel
        W2100 = amodel.build_world(horizon=2100)
        t = time.perf_counter()
        coupling.forecast(69, R=64, world=W2100, horizon=81, f5=coupling.F5Config())
        out["l1_64_2019_2100_s"] = time.perf_counter() - t
        out["l1_64_parallel_8x8_s"] = l1_parallel()
        out["l1_ok"] = out["l1_64_parallel_8x8_s"] < L1_S
    log(f"W10 on {out['cpu']}: L0 regional {out['l0r_ms_median']:.2f} ms per run to 2100 (World3-03 "
        f"{out['world3_ms_single']:.0f} ms single, {out['world3_ms_per_member_n1000']:.2f} ms per member in a batch of "
        f"1000)" + (f"; L1 64 x 2019->2100 {out['l1_64_2019_2100_s']:.0f} s in one process, "
                    f"{out['l1_64_parallel_8x8_s']:.0f} s in 8 x 8" if l1 else ""))
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    r = run()
    Path("runs").mkdir(exist_ok=True)
    Path("runs/w10.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
