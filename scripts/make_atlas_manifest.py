"""Escribe world4/data/atlas_manifest.json desde un clon del Atlas en el commit fijado.

Uso: python scripts/make_atlas_manifest.py <clon> 
El clon debe tener panel_full.csv generado con su propio merge_sources.py (src/).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world4.data import atlas  # noqa: E402

man = atlas.make_manifest(Path(sys.argv[1]))
atlas.MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print(len(man["files"]), "ficheros ->", atlas.MANIFEST)
