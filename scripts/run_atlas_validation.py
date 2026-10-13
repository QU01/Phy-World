"""Validación completa del L1 portado (= validate.main() del Atlas) con procesos 'spawn'.

Uso: python scripts/run_atlas_validation.py <out.json> [workers]
Las semillas sólo dependen del origen, así que el reparto entre procesos no cambia el
resultado. No escribe en data_cache/atlas/results (no pisa el validation.json del Atlas).
"""
import json
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def chunk(args):
    warnings.simplefilter("ignore")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world4.core.l1_atlas import validate
    origins, variants, R = args
    return validate.evaluate(origins=list(origins), variants=variants, R=R, log=lambda *a: None)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    from world4.data import atlas
    atlas.require()
    from world4.core.l1_atlas import validate
    out, workers = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 6
    t = time.time()
    origins = list(range(20, 69))
    variants = tuple(validate.ABLATIONS)
    with get_context("spawn").Pool(workers) as pool:
        parts = pool.map(chunk, [(origins[k::workers], variants, 32) for k in range(workers)])
    summ = validate.summarise([r for p in parts for r in p])
    summ["meta"] = dict(origins=[1970, 2018], cutoffs=validate.CUTOFFS, horizons=validate.HORIZONS, R=32,
                        seconds=time.time() - t)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("done", round(time.time() - t), "s ->", out)
