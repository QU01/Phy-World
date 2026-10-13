"""Instala y verifica los datos del Atlas del Sistema-Mundo que necesita el L1.

World4 no los incorpora: varios derivan de fuentes que no se pueden redistribuir (EM-DAT) o
cuya licencia no está verificada (ver ``sources.json``). Se copian desde el repo del Atlas en
el commit fijado y cada fichero se comprueba contra ``atlas_manifest.json`` (SHA-256).

``panel_full.csv`` no está en el repo del Atlas (lo genera ``merge_sources.py``) y además
``model.build_world`` cae EN SILENCIO a ``panel.csv`` si falta, con otro resultado. Por eso
``install`` lo genera y ``verify`` exige su SHA-256: sin él no hay corrida del L1.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ATLAS_REPO = "https://github.com/QU01/Reproducible-Research-Project-2"
ATLAS_COMMIT = "9e96f6b9f470ca990213cd977762da6f440b283b"
MANIFEST = Path(__file__).with_name("atlas_manifest.json")

# Lo que el L1 necesita para construir el mundo, calibrar y validar. data/raw sólo hace falta
# para reconstruir el panel desde las fuentes, que no es parte de F0.
COPY = ("data/processed/panel.csv", "data/processed/world_full.csv",
        "data/processed/sources", "results/calibration.json", "results/rolling",
        "results/irl.json", "results/validation.json", "results/summary.json")
GENERATED = ("data/processed/panel_full.csv",)


class AtlasDataError(RuntimeError):
    pass


def root() -> Path:
    from world4.core.l1_atlas._paths import ATLAS_ROOT
    return ATLAS_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(base: Path, rels) -> dict[str, Path]:
    out = {}
    for rel in rels:
        p = base / rel
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    out[f.relative_to(base).as_posix()] = f
        elif p.is_file():
            out[rel] = p
    return out


def make_manifest(src_checkout: Path) -> dict:
    """Lo usa ``scripts/make_atlas_manifest.py``: hashes de los ficheros copiados y generados."""
    files = _files(src_checkout, COPY + GENERATED)
    missing = [g for g in GENERATED if g not in files]
    if missing:
        raise AtlasDataError(f"faltan ficheros generados en el clon: {missing} (correr merge_sources)")
    return {"repo": ATLAS_REPO, "commit": ATLAS_COMMIT,
            "files": {rel: {"sha256": sha256(p), "bytes": p.stat().st_size,
                            "generated": rel in GENERATED} for rel, p in files.items()}}


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify(base: Path | None = None) -> dict:
    """Comprueba cada fichero del manifiesto. Devuelve {ok, missing, mismatched, root}."""
    base = Path(base) if base else root()
    man = manifest()
    missing, bad = [], []
    for rel, meta in man["files"].items():
        p = base / rel
        if not p.is_file():
            missing.append(rel)
        elif sha256(p) != meta["sha256"]:
            bad.append(rel)
    return {"ok": not missing and not bad, "missing": missing, "mismatched": bad, "root": str(base),
            "commit": man["commit"]}


def require() -> Path:
    """Para quien va a correr el L1: o los datos están y son los del manifiesto, o error con remedio."""
    v = verify()
    if not v["ok"]:
        raise AtlasDataError(
            f"datos del Atlas incompletos en {v['root']} (faltan {len(v['missing'])}, "
            f"distintos {len(v['mismatched'])}). Remedio: python world4_cli.py --install-atlas "
            f"[--atlas-src <clon local en {ATLAS_COMMIT[:7]}>]")
    return Path(v["root"])


def install(src: str | Path | None = None, dest: Path | None = None) -> dict:
    """Copia los datos desde un clon local del Atlas (o clona el repo) y genera panel_full.csv."""
    dest = Path(dest) if dest else root()
    tmp = None
    if src is None:
        tmp = Path(tempfile.mkdtemp(prefix="w4atlas_"))
        subprocess.run(["git", "clone", "-q", ATLAS_REPO, str(tmp / "atlas")], check=True)
        subprocess.run(["git", "-C", str(tmp / "atlas"), "checkout", "-q", ATLAS_COMMIT], check=True)
        src = tmp / "atlas"
    src = Path(src)
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if head != ATLAS_COMMIT:
        raise AtlasDataError(f"el clon {src} está en {head[:7] or '?'}, World4 fija {ATLAS_COMMIT[:7]}")
    try:
        for rel, p in _files(src, COPY).items():
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
        _generate_panel_full()
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return verify(dest)


def _generate_panel_full():
    """merge_sources.main() del Atlas portado, sin tocar: escribe panel_full.csv y world_full.csv."""
    import contextlib
    import io
    import warnings

    from world4.core.l1_atlas import merge_sources
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        merge_sources.main()
