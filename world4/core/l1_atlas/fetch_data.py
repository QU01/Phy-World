"""Download raw data (CRAN mirrors on GitHub + world-atlas) and convert .rda -> .csv."""
import subprocess
import urllib.request
from pathlib import Path

import rdata
from ._paths import ATLAS_ROOT

RAW = ATLAS_ROOT / "data" / "raw"
SOURCES = {
    "pwt10.01.rda": "https://raw.githubusercontent.com/cran/pwt10/master/data/pwt10.01.rda",
    "ucdp_acd.rda": "https://raw.githubusercontent.com/cran/peacesciencer/master/data/ucdp_acd.rda",
    "ucdp_onsets.rda": "https://raw.githubusercontent.com/cran/peacesciencer/master/data/ucdp_onsets.rda",
    "gwcode_democracy.rda": "https://raw.githubusercontent.com/cran/peacesciencer/master/data/gwcode_democracy.rda",
}


WDI_RENTS = ("https://raw.githubusercontent.com/hdesaioecd/oecd-sof-2022-public/HEAD/data/"
             "2020%20sfr%20model%20data/API_NY-2/API_NY.GDP.TOTL.RT.ZS_DS2_en_csv_v2_3470550.csv")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dst = RAW / name
        if not dst.exists():
            urllib.request.urlretrieve(url, dst)
        obj = rdata.read_rda(dst)
        df = obj[list(obj)[0]]
        csv = "pwt1001.csv" if name.startswith("pwt") else name.replace(".rda", ".csv")
        df.to_csv(RAW / csv, index=False)
        print(name, df.shape)
    if not (RAW / "wdi_resource_rents.csv").exists():
        # World Bank WDI "Total natural resources rents (% of GDP)", bulk CSV (Dec 2021 release)
        urllib.request.urlretrieve(WDI_RENTS, RAW / "wdi_resource_rents.csv")
    if not (RAW / "countries-110m.json").exists():
        # Natural Earth 110m boundaries from the world-atlas npm package
        subprocess.run(["npm", "pack", "world-atlas@2", "--pack-destination", str(RAW)], check=True)
        subprocess.run(["tar", "xzf", str(next(RAW.glob("world-atlas-*.tgz"))), "-C", str(RAW),
                        "package/countries-110m.json", "--strip-components=1"], check=True)


if __name__ == "__main__":
    main()
