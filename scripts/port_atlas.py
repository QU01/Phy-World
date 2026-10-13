"""Porta el código del Atlas del Sistema-Mundo a ``world4/core/l1_atlas`` SIN cambiar comportamiento.

Uso: python scripts/port_atlas.py <clon del Atlas en 9e96f6b>

Las únicas transformaciones son mecánicas y se listan aquí; cualquier otra diferencia es un
error del port y la prueba W2 la detecta:

  1. ``from model import X`` (y behavior, calibrate, submodels, rl, geo, irl) → import relativo.
  2. ``ROOT = Path(__file__).resolve().parents[k]`` → ``ROOT = ATLAS_ROOT`` (la carpeta de datos
     y resultados vive fuera del paquete, ver ``world4/data/atlas.py``).
  3. ``Path(__file__).resolve().parents[1] / "data" / "raw"`` de fetch_data.py → ``ATLAS_ROOT / ...``.

Al final imprime el diff de líneas cambiadas por fichero para revisarlo a ojo.
"""
import difflib
import subprocess
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) / "src"
DST = Path(__file__).resolve().parents[1] / "world4" / "core" / "l1_atlas"
EXPECTED = "9e96f6b9f470ca990213cd977762da6f440b283b"
LOCAL = ("model", "behavior", "calibrate", "submodels", "rl", "geo", "irl")

head = subprocess.run(["git", "-C", str(SRC.parent), "rev-parse", "HEAD"], capture_output=True,
                      text=True).stdout.strip()
if head != EXPECTED:
    sys.exit(f"el clon está en {head}, se esperaba {EXPECTED}")


def transform(line: str, depth: int) -> str:
    s = line.lstrip()
    indent = line[: len(line) - len(s)]
    for mod in LOCAL:
        if s.startswith(f"from {mod} import "):
            return f"{indent}from {'.' * depth}{mod} import " + s[len(f"from {mod} import "):]
    if s.startswith("from sources import "):
        return indent + "from . import " + s[len("from sources import "):]
    for k in (1, 2):
        old = f"ROOT = Path(__file__).resolve().parents[{k}]"
        if s.startswith(old):
            return indent + "ROOT = ATLAS_ROOT" + s[len(old):]
    old = 'RAW = Path(__file__).resolve().parents[1] / "data" / "raw"'
    if s.startswith(old):
        return indent + 'RAW = ATLAS_ROOT / "data" / "raw"' + s[len(old):]
    return line


def port(src: Path, dst: Path, depth: int):
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out = [transform(ln, depth) for ln in lines]
    if any("ATLAS_ROOT" in ln for ln in out):
        # el import de la ruta va justo después de la docstring/futuros: se inserta tras el
        # último import de nivel superior para no cambiar el orden de ejecución de nada más
        last = max(i for i, ln in enumerate(out) if ln.startswith(("import ", "from ")))
        out.insert(last + 1, f"from {'.' * depth}_paths import ATLAS_ROOT\n")
    dst.write_text("".join(out), encoding="utf-8", newline="")
    return [d for d in difflib.unified_diff(lines, out, n=0) if d.startswith(("+", "-"))
            and not d.startswith(("+++", "---"))]


DST.mkdir(parents=True, exist_ok=True)
(DST / "sources").mkdir(exist_ok=True)
report = {}
for f in sorted(SRC.glob("*.py")):
    report[f.name] = port(f, DST / f.name, 1)
for f in sorted((SRC / "sources").glob("*.py")):
    report["sources/" + f.name] = port(f, DST / "sources" / f.name, 2)
for name, diff in report.items():
    if diff:
        print(f"== {name}")
        for d in diff:
            print("   " + d.rstrip())
print(f"portados {len(report)} ficheros desde {EXPECTED[:7]}")
