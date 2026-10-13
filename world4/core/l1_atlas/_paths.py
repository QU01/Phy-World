"""Dónde vive la carpeta de datos y resultados del Atlas (``data/``, ``results/``).

No viaja dentro del paquete: varios de sus ficheros derivan de fuentes cuya licencia no permite
redistribuirlos o no está verificada (EM-DAT, entre otras; ver ``world4/data/sources.json``).
``world4.data.atlas.install()`` la copia desde el repo del Atlas en el commit fijado y
verifica cada SHA-256. Se cambia con la variable de entorno ``W4_ATLAS_ROOT``.
"""
import os
from pathlib import Path

from world4 import home as _home  # noqa: E402

DEFAULT = _home.data_cache() / "atlas"
ATLAS_ROOT = Path(os.environ.get("W4_ATLAS_ROOT", DEFAULT))
