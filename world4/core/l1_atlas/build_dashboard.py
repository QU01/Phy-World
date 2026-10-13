"""Build the dashboard: dashboard/index.html plus the files it fetches next to it.

  index.html        template with the narrative text inlined
  data.json         results/dashboard_data.json (model scenarios, validation, RL, recommendations)
  geo.json          Natural Earth 1:110m countries (GeoJSON)
  osint/*.json      open-source layers built by src/osint.py
  img/*.jpg         globe textures (NASA Blue Marble / Black Marble, via the three-globe package)
"""
import json
import shutil
from pathlib import Path

from .geo import load_countries
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
DASH = ROOT / "dashboard"


def main():
    tpl = (DASH / "template.html").read_text()
    text = json.load(open(DASH / "findings.json"))
    html = (tpl.replace("__FINDINGS__", json.dumps(text["findings"]))
            .replace("__LIMITS__", json.dumps(text["limits"]))
            .replace("__CLIMA__", json.dumps(text.get("clima", ""))))
    (DASH / "index.html").write_text(html)
    shutil.copy(ROOT / "results" / "dashboard_data.json", DASH / "data.json")
    json.dump(load_countries(), open(DASH / "geo.json", "w"), separators=(",", ":"))
    for f in ["index.html", "data.json", "geo.json"] + sorted(str(p.relative_to(DASH)) for p in (DASH / "osint").glob("*.json")):
        print(f, round((DASH / f).stat().st_size / 1e6, 2), "MB")


def publish_files():
    """Map of published path -> local path for the Artifact tool."""
    files = {"data.json": DASH / "data.json", "geo.json": DASH / "geo.json"}
    for p in list((DASH / "osint").glob("*.json")) + list((DASH / "img").glob("*")):
        files[str(p.relative_to(DASH))] = p
    return {k: str(v.relative_to(ROOT)) for k, v in files.items()}


if __name__ == "__main__":
    main()
    print(json.dumps(publish_files(), indent=1))
