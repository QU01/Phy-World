"""QUASAR Phy-World - command line. The CLI lives in ``world4.cli`` (installed as ``world4``); this file keeps
``python world4_cli.py ...`` working from a checkout of the repo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from world4.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
