"""Inserta ``data.json`` en ``plantilla.html`` y escribe el informe autocontenido ``informe.html``."""
from pathlib import Path

HERE = Path(__file__).parent
data = (HERE / "data.json").read_text(encoding="utf-8").replace("</", "<\\/")
page = (HERE / "plantilla.html").read_text(encoding="utf-8").replace("__DATA__", data, 1)
(HERE / "informe.html").write_text(page, encoding="utf-8")
print(f"informe.html: {len(page) / 1e6:.2f} MB")
