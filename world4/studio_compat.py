"""F10: el shim que el contrato genérico de Quasar Studio espera del módulo de física (hoja de ruta §5.5).

Studio (``backend/app/services/phy_models``) importa este módulo bajo su aislamiento y lee de él ``Fidelity``,
``FAMILIES``, ``VAR_NAMES``, ``BOUNDS_LO``/``BOUNDS_HI``, ``denormalize`` y ``evaluate``. Las corridas las
lanza su adaptador (``models/world4.py`` de Studio) llamando a ``run_studio``: así la lógica vive y se prueba
aquí, en el paquete, y Studio sólo traduce.

Qué significa cada pieza en World4:

* **θ** = las 7 palancas de política de F8 (``scenarios.levers``), con su caja y unidades.
* **Familias** = los dos modos de la CLI que Studio ofrece: ``scenario`` (una política sobre el ensemble,
  ``--theta``) y ``design`` (la búsqueda robusta, ``--design``).
* **Fidelidad:** L0 = el L0 regional a 2100 (la política y la búsqueda viven ahí); L1 = lo mismo más los 180
  países como agentes 2019-2029 (F1-F5), que es lo que pinta el globo; L2 no se ofrece: sus emuladores (FaIR,
  cultivos) ya están acoplados en el L1 desde F4/F5 (la CLI la degrada con aviso, Studio la declara).

Nada aquí escribe fuera del ``outdir`` que recibe ``run_studio``.
"""
from __future__ import annotations

import enum
import json
import time
from pathlib import Path

import numpy as np

from world4.scenarios import levers


class Fidelity(enum.Enum):
    L0 = 0
    L1 = 1
    L2 = 2


FAMILIES = ("scenario", "design")
VAR_NAMES = {f: list(levers.NAMES) for f in FAMILIES}
NDIM = {f: len(levers.NAMES) for f in FAMILIES}
BOUNDS_LO = {f: levers.BOX[:, 0].tolist() for f in FAMILIES}
BOUNDS_HI = {f: levers.BOX[:, 1].tolist() for f in FAMILIES}
UNITS = {lv.name: lv.unit for lv in levers.LEVERS}
OUT_KEYS = ["neg_welfare_p10", "peak_gmst_p90", "neg_poorest_p10", "food_floor", "land_cap", "debt_cap"]
SOURCE = {Fidelity.L0: "world4(L0)", Fidelity.L1: "world4(L0+L1)", Fidelity.L2: "world4(L0+L1)"}

# miembros por defecto, los de la CLI: --sow 200 en --theta, 100 en --design; 32 miembros del L1 (los del dashboard)
MEMBERS = {"scenario": 200, "design": 100}
L1_MEMBERS = 32


def denormalize(u, family: str | None = None) -> np.ndarray:
    """[0, 1]^7 -> la caja de las palancas."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return levers.BOX[:, 0] + u * (levers.BOX[:, 1] - levers.BOX[:, 0])


def _fid(fidelity) -> Fidelity:
    if isinstance(fidelity, Fidelity):
        return fidelity
    if isinstance(fidelity, str):
        return Fidelity[fidelity]
    return Fidelity(int(fidelity))


def evaluate(theta, fidelity=Fidelity.L0, family: str | None = None, *, members: int = 100,
             narrative: str = "ssp2", seed: int = 0) -> dict:
    """Una política sobre el ensemble del L0 (sin PRIM: es la evaluación puntual barata). Registro plano."""
    from world4.scenarios import robust
    th = np.asarray(theta, float)
    levers.to_policy(th)
    r = robust.evaluate_policy(th, members=members, narrative=narrative, seed=seed, vulnerability=False)
    rec = {**r["objectives"], **r["constraints"], "feasible": bool(r["feasible"]),
           "better_welfare_share": r["better_than_baseline_share"]["welfare"],
           "better_peak_gmst_share": r["better_than_baseline_share"]["peak_gmst"],
           "gmst_2100_p50": r["percentiles"]["gmst_2100"][50], "nonfossil_2050_p50": r["percentiles"]["nonfossil_2050"][50],
           "members": members, "narrative": narrative, "exploratory": True,
           # la evaluación puntual es el L0 aunque se pida L1: el L1 no tiene palancas (declarado)
           "source": "world4(L0)" if _fid(fidelity) is Fidelity.L0 else "world4(L0; L1 has no levers)"}
    return rec


def _dump(path: Path, obj) -> str:
    path.write_text(json.dumps(obj, separators=(",", ":"), default=lambda o: o.tolist() if hasattr(o, "tolist")
                               else str(o)), encoding="utf-8")
    return path.name


def run_studio(cfg: dict, outdir, log=print) -> dict:
    """Una corrida de Studio: ``scenario`` (evaluate_policy + PRIM) o ``design`` (búsqueda robusta), y en las dos
    los datos del explorador (``world.json``, ``geo.json``, ``graph.json``) y el informe HTML de F9.

    ``cfg``: ``family``, ``fidelity`` (L0/L1), ``narrative``, ``members`` (estados del mundo), ``l1_members``,
    las palancas por nombre (scenario), ``rounds``/``n_init`` (design), ``land_reported`` y ``seed``. Devuelve el resultado en la forma que el genérico de Studio resume: ``record``, ``theta``,
    ``pareto_front`` y ``dataset`` (cada uno con ``source``), más ``deliverables``."""
    from world4 import explorer, report
    from world4.scenarios import narratives, robust
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    family = str(cfg.get("family") or "scenario")
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; families: {', '.join(FAMILIES)}")
    fid = _fid(cfg.get("fidelity") or "L0")
    if fid is Fidelity.L2:
        log("L2 is not separate in World4: its emulators (FaIR, crops) are already coupled in the L1; running L1")
        fid = Fidelity.L1
    narrative = str(cfg.get("narrative") or narratives.DEFAULT)
    if narrative not in narratives.NARRATIVES:
        raise ValueError(f"unknown narrative {narrative!r}; narratives: {', '.join(narratives.NARRATIVES)}")
    members = int(cfg.get("members") or MEMBERS[family])
    l1_members = int(cfg.get("l1_members") or L1_MEMBERS)
    seed = int(cfg.get("seed") or 0)
    files: dict[str, str] = {}
    source = SOURCE[fid]

    if family == "scenario":
        th = np.array([float(cfg.get(n) or 0.0) for n in levers.NAMES])
        levers.to_policy(th)
        log(f"round: 1/3 policy over {members} states of the world, narrative {narrative} (L0 to 2100)")
        pol = robust.evaluate_policy(th, members=members, narrative=narrative, seed=seed, vulnerability=True)
        files["policy"] = _dump(out / "policy.json", pol)
        front, dataset = [], [{**pol["objectives"], **pol["constraints"], "feasible": pol["feasible"],
                               "theta": pol["theta"], "source": source}]
        search = None
        record = {**pol["objectives"], **pol["constraints"], "feasible": bool(pol["feasible"]),
                  "better_welfare_share": pol["better_than_baseline_share"]["welfare"],
                  "better_peak_gmst_share": pol["better_than_baseline_share"]["peak_gmst"]}
    else:
        rounds = int(cfg.get("rounds") or 2)
        n_init = int(cfg.get("n_init") or 40)
        report_only = ("land_cap",) if cfg.get("land_reported", True) else ()
        log(f"round: 1/3 robust search, narrative {narrative}, {members} states of the world, {rounds} rounds"
            + (" (land reported, not constraining: no lever moves it in the L0)" if report_only else ""))
        search = robust.design(members=members, narrative=narrative, seed=seed, n_init=n_init, rounds=rounds,
                               report_only=report_only, log=log)
        files["design"] = _dump(out / "design.json", search)
        reps = search.get("representatives") or {}
        chosen = (reps.get("compromise") or {}).get("theta") or {}
        th = np.array([float(chosen.get(n, 0.0)) for n in levers.NAMES])
        log("round: 2/3 the compromise policy over the ensemble, with its vulnerabilities (PRIM)")
        pol = robust.evaluate_policy(th, members=max(members, 100), narrative=narrative, seed=seed)
        files["policy"] = _dump(out / "policy.json", pol)
        front = [{**r["objectives"], **r["constraints"], "feasible": r["feasible"], "theta": r["theta"],
                  "source": source} for r in search.get("front") or []]
        dataset = front
        record = {**pol["objectives"], **pol["constraints"], "feasible": bool(pol["feasible"]),
                  "front_size": search.get("front_size"), "feasible_on_front": search.get("feasible_on_front"),
                  "evaluations": search.get("evaluations")}
    log("round: 3/3 explorer data (agents, flows, L0 regions, graph)" + ("" if fid is Fidelity.L0 else
                                                                         ", with the L1 country agents"))
    world = explorer.world_payload(theta=th, narrative=narrative, members=members, R=l1_members,
                                   seed=seed, with_l1=fid is not Fidelity.L0, log=log)
    files["world"] = _dump(out / "world.json", world)
    files["geo"] = _dump(out / "geo.json", explorer.geo_payload())
    loops = explorer.loops_payload()
    files["graph"] = _dump(out / "graph.json", explorer.graph_payload(world, policy=pol, search=search, loops=loops))
    try:
        r = report.write(out / "world4_report.html", th, narrative=narrative, members=members, campaign=False,
                         log=lambda *a: None)
        files["report"] = Path(r["html"]).name
    except Exception as exc:                                    # noqa: BLE001 (un informe roto no tumba la corrida)
        log(f"report skipped: {type(exc).__name__}: {exc}")
    record = {k: v for k, v in record.items() if v is not None}
    return {"family": family, "theta": th.tolist(), "record": {**record, "source": source},
            "pareto_front": front, "dataset": dataset, "deliverables": files, "narrative": narrative,
            "fidelity": fid.name, "members": members, "seconds": round(time.time() - t_start, 1),
            "exploratory": True}
