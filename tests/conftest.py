import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(scope="session")
def atlas_ready():
    """Las pruebas del L1 necesitan los datos del Atlas instalados y verificados. Si faltan, se
    saltan CON el remedio en el motivo; la campaña marca W2 como BLOQUEADO, nunca verde."""
    from world4.data import atlas
    v = atlas.verify()
    if not v["ok"]:
        pytest.skip("datos del Atlas sin instalar: python world4_cli.py --install-atlas")
    return v
