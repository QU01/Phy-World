"""Extrae la salida del Vensim World3-03 que WorldDynamics.jl guarda en ``test/world3_03.jl``.

Uso: python scripts/make_world3_reference.py <ruta a WorldDynamics.jl> 
Escribe tests/reference/world3_03_vensim.csv (time + 5 variables, 1900-2100 cada 0,5 años).
WorldDynamics.jl es MIT (Natale, Crescenzi, Serafim 2022); los números son la salida del
modelo Vensim World3-03, no de WorldDynamics.
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "tests" / "reference" / "world3_03_vensim.csv"
NAMES = {"food": "f", "industrial_output": "io", "nonrenewable_resources": "nr",
         "persistent_pollution": "ppolx", "population": "pop"}

text = (SRC / "test" / "world3_03.jl").read_text()
cols = {}
for jl, ours in NAMES.items():
    m = re.search(rf"^{jl} = \[(.*?)\]", text, re.S | re.M)
    cols[ours] = np.array([float(v) for v in m.group(1).split()])
n = {len(v) for v in cols.values()}
assert n == {401}, n
commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
sha = hashlib.sha256((SRC / "test" / "world3_03.jl").read_bytes()).hexdigest()
t = 1900 + 0.5 * np.arange(401)
with open(OUT, "w", newline="\n") as fh:
    fh.write(f"# Vensim World3-03 (scenario 1), via WorldDynamics.jl test/world3_03.jl @ {commit}\n")
    fh.write(f"# sha256(test/world3_03.jl) = {sha}; license of the file: MIT\n")
    fh.write("time," + ",".join(cols) + "\n")
    for i in range(401):
        fh.write(f"{t[i]:g}," + ",".join(repr(float(cols[k][i])) for k in cols) + "\n")
print("wrote", OUT)
