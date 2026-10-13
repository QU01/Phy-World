"""Dónde vive lo que World4 descarga, calcula y escribe (F9: el paquete se instala con ``pip``).

* **Checkout del repo** (``pyproject.toml`` y ``world4_cli.py`` junto al paquete): la raíz del repo, como
  en F0-F8 (``data_cache/``, ``runs/``).
* **Instalado** (``pip install``): ``W4_HOME`` si está definida, si no ``~/.world4``.

Lo que VIAJA con el paquete (``package-data``): el posterior de F7 (``world4/models/posterior.npz``) y las
referencias de W1 y W2 (``world4/data/reference/``). Lo que no viaja: los datos de terceros, que se instalan
con ``--install-*`` y se verifican por SHA-256 (W12).
"""
from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
_REPO = PKG.parent
SOURCE_CHECKOUT = (_REPO / "pyproject.toml").is_file() and (_REPO / "world4_cli.py").is_file()
REFERENCE = PKG / "data" / "reference"
PACKAGE_MODELS = PKG / "models"


def home() -> Path:
    env = os.environ.get("W4_HOME")
    if env:
        return Path(env)
    return _REPO if SOURCE_CHECKOUT else Path.home() / ".world4"


def data_cache() -> Path:
    return home() / "data_cache"


def runs() -> Path:
    return home() / "runs"


def models() -> Path:
    """Donde se ESCRIBE un posterior nuevo: en un checkout, el del paquete (se versiona); instalado, el hogar."""
    return PACKAGE_MODELS if SOURCE_CHECKOUT and not os.environ.get("W4_HOME") else home() / "models"
