"""Descarga fuentes tipográficas (OFL) y series observadas recientes (2015-2025)
para contrastar la capa L1 (Atlas del Sistema-Mundo, panel hasta 2019) contra
los datos más recientes disponibles.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
FONTS = HERE / "fonts"
OBS = HERE / "recent_obs"
FONTS.mkdir(parents=True, exist_ok=True)
OBS.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Phy-World-L1-Report/1.0"}


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 500:
        print(f"  [ok] {dest.name} (cached)")
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  [ok] {dest.name} ({len(data):,} B)")
        return True
    except Exception as e:
        print(f"  [FAIL] {dest.name}: {e}")
        return False


def fetch_fonts():
    print("== Fuentes tipográficas ==")
    urls = {
        # Source Serif 4 (editorial book serif, bold & italic) + fallback Instrument Serif
        "Newsreader-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/newsreader/Newsreader%5Bopsz%2Cwght%5D.ttf",
        "Newsreader-Italic.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/newsreader/Newsreader-Italic%5Bopsz%2Cwght%5D.ttf",
        "Lora-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/static/Lora-Bold.ttf",
        "Lora-BoldItalic.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/static/Lora-BoldItalic.ttf",
        "Lora-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/static/Lora-Regular.ttf",
        "Lora-Italic.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/static/Lora-Italic.ttf",
        "IBMPlexSans-Regular.ttf": "https://raw.githubusercontent.com/IBM/plex/master/packages/plex-sans/fonts/complete/ttf/IBMPlexSans-Regular.ttf",
        "IBMPlexSans-Italic.ttf": "https://raw.githubusercontent.com/IBM/plex/master/packages/plex-sans/fonts/complete/ttf/IBMPlexSans-Italic.ttf",
        "IBMPlexSans-Medium.ttf": "https://raw.githubusercontent.com/IBM/plex/master/packages/plex-sans/fonts/complete/ttf/IBMPlexSans-Medium.ttf",
        "IBMPlexSans-SemiBold.ttf": "https://raw.githubusercontent.com/IBM/plex/master/packages/plex-sans/fonts/complete/ttf/IBMPlexSans-SemiBold.ttf",
        "IBMPlexSans-Bold.ttf": "https://raw.githubusercontent.com/IBM/plex/master/packages/plex-sans/fonts/complete/ttf/IBMPlexSans-Bold.ttf",
        "IBMPlexMono-Regular.ttf": "https://raw.githubusercontent.com/IBM/plex/master/packages/plex-mono/fonts/complete/ttf/IBMPlexMono-Regular.ttf",
        "IBMPlexMono-Medium.ttf": "https://raw.githubusercontent.com/IBM/plex/master/packages/plex-mono/fonts/complete/ttf/IBMPlexMono-Medium.ttf",
    }
    for name, url in urls.items():
        download(url, FONTS / name)


def fetch_wdi(indicator: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [ok] {dest.name} (cached)")
        return True
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000&date=2010:2025"
    return download(url, dest)


def fetch_recent_obs():
    print("== Datos observados recientes (2010-2025) ==")
    # 1. World Bank WDI: Real GDP per capita PPP (constant 2021 int$) & constant 2015 USD
    fetch_wdi("NY.GDP.PCAP.PP.KD", OBS / "wdi_gdp_pc_ppp.json")
    fetch_wdi("NY.GDP.PCAP.KD", OBS / "wdi_gdp_pc_const.json")
    # 2. World Bank WDI: Natural resource rents (% of GDP) - same series as Atlas WDI
    fetch_wdi("NY.GDP.TOTL.RT.ZS", OBS / "wdi_res_rents.json")
    # 3. OWID: V-Dem Political Regime (RoW 0-3, >=2 is democracy) & Electoral Democracy Index
    download("https://ourworldindata.org/grapher/political-regime.csv?v=1&csvType=full&useColumnShortNames=true",
             OBS / "owid_political_regime.csv")
    download("https://ourworldindata.org/grapher/electoral-democracy-index.csv?v=1&csvType=full&useColumnShortNames=true",
             OBS / "owid_vdem_polyarchy.csv")
    # 4. UCDP/PRIO Armed Conflict Dataset (v24.1 / v25.1) or OWID armed conflict deaths by country
    download("https://ourworldindata.org/grapher/deaths-in-armed-conflicts-by-country.csv?v=1&csvType=full&useColumnShortNames=true",
             OBS / "owid_ucdp_conflict.csv")
    download("https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-241-csv.zip",
             OBS / "ucdp_prio_acd_241.zip")


if __name__ == "__main__":
    fetch_fonts()
    fetch_recent_obs()
