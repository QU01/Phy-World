"""Huellas del Atlas ORIGINAL para el candado W2 (tests/golden/atlas_9e96f6b.json).

Uso: python scripts/make_atlas_golden.py <clon del Atlas en 9e96f6b, con panel_full.csv>

Importa los módulos del clon (``src/`` en sys.path, nombres planos ``model``/``calibrate``…),
no los del paquete world4, y les pasa la misma sonda que usa la prueba del port.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLONE = Path(sys.argv[1])
head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
sys.path.insert(0, str(CLONE / "src"))
sys.path.insert(0, str(REPO))

import warnings  # noqa: E402

warnings.simplefilter("ignore")
import calibrate  # noqa: E402  (módulos ORIGINALES)
import model  # noqa: E402
import validate  # noqa: E402

from world4.validate import atlas_parity  # noqa: E402

assert Path(model.__file__).resolve().is_relative_to(CLONE.resolve()), model.__file__
fp = atlas_parity.fingerprints(atlas_parity.probe(model, calibrate, validate))
fp["atlas_commit"] = head
out = REPO / "tests" / "golden" / f"atlas_{head[:7]}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(fp, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print(len(fp["arrays"]), "huellas ->", out)
