"""
Coffee Weather — One-Time Backfill
===================================
Fetches 2016-2026 + normals for all 5 origins (Brazil, Colombia, Honduras,
Super 4, Vietnam) and saves one parquet file per origin.

Run once:
    python backfill.py

After this, use daily_update.py for incremental refreshes.
"""

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
PARQUET_DIR = Path(__file__).parent / "data"
API_URL     = "https://api.weatherdesk.xweather.com/2e621a7f-2b1e-4f3e-af6a-5a986a68b398/services/gwi/v1/timeseries"
MAX_WORKERS = 20

# API keys to pull — "normals" is the API's own key for climate normals
FETCH_YEARS = [
    "2016", "2017", "2018", "2019", "2020",
    "2021", "2022", "2023", "2024", "2025", "2026",
    "normals",
]

# -------------------------------------------------------
# ORIGINS
# -------------------------------------------------------
ORIGINS = {
    "Brazil": {
        "file": "brazil.parquet",
        "stations": {
            "86827": "Espirito Santo", "86828": "Espirito Santo", "86829": "Espirito Santo",
            "86805": "Espirito Santo", "86785": "Espirito Santo", "86853": "Espirito Santo",
            "86804": "Espirito Santo", "83550": "Espirito Santo",
            "83595": "Minas Gerais",  "86743": "Minas Gerais",  "83442": "Minas Gerais",
            "83579": "Minas Gerais",  "83384": "Minas Gerais",  "83582": "Minas Gerais",
            "86850": "Minas Gerais",  "86800": "Minas Gerais",  "86799": "Minas Gerais",
            "86718": "Minas Gerais",  "86846": "Minas Gerais",  "86761": "Minas Gerais",
            "83592": "Minas Gerais",  "86719": "Minas Gerais",  "86794": "Minas Gerais",
            "83566": "Minas Gerais",  "86822": "Minas Gerais",  "86780": "Minas Gerais",
            "83538": "Minas Gerais",  "86797": "Minas Gerais",  "86798": "Minas Gerais",
            "86820": "Minas Gerais",  "83574": "Minas Gerais",  "86783": "Minas Gerais",
            "86782": "Minas Gerais",  "86757": "Minas Gerais",  "86821": "Minas Gerais",
            "86742": "Minas Gerais",  "86758": "Minas Gerais",  "83692": "Minas Gerais",
            "83687": "Minas Gerais",  "86825": "Minas Gerais",  "86784": "Minas Gerais",
            "86871": "Minas Gerais",  "83437": "Minas Gerais",  "86852": "Minas Gerais",
            "86823": "Minas Gerais",  "83479": "Minas Gerais",  "86873": "Minas Gerais",
            "86819": "Minas Gerais",  "83531": "Minas Gerais",  "86778": "Minas Gerais",
            "83393": "Minas Gerais",  "83483": "Minas Gerais",  "86795": "Minas Gerais",
            "86741": "Minas Gerais",  "86849": "Minas Gerais",  "86763": "Minas Gerais",
            "83492": "Minas Gerais",  "86801": "Minas Gerais",  "86779": "Minas Gerais",
            "86776": "Minas Gerais",  "86738": "Minas Gerais",  "86848": "Minas Gerais",
            "86824": "Minas Gerais",
            "86816": "Sao Paulo", "86865": "Sao Paulo", "86844": "Sao Paulo",
            "83630": "Sao Paulo", "86869": "Sao Paulo", "86817": "Sao Paulo",
            "86839": "Sao Paulo", "86866": "Sao Paulo", "86868": "Sao Paulo",
            "86842": "Sao Paulo", "83716": "Sao Paulo", "86864": "Sao Paulo",
            "83726": "Sao Paulo", "86838": "Sao Paulo", "86815": "Sao Paulo",
        },
    },
    "Colombia": {
        "file": "colombia.parquet",
        "stations": {
            "80009": "Colombia", "80036": "Colombia", "80063": "Colombia",
            "80091": "Colombia", "80110": "Colombia", "80112": "Colombia",
            "80210": "Colombia", "80211": "Colombia", "80214": "Colombia",
            "80222": "Colombia",
        },
    },
    "Honduras": {
        "file": "honduras.parquet",
        "stations": {
            "78708": "Honduras", "78714": "Honduras", "78717": "Honduras",
            "78718": "Honduras", "78719": "Honduras", "78720": "Honduras",
        },
    },
    "Super 4": {
        "file": "super4.parquet",
        "stations": {
            "84050": "Super 4", "84105": "Super 4", "84135": "Super 4",
            "84143": "Super 4", "84226": "Super 4",
        },
    },
    "Vietnam": {
        "file": "vietnam.parquet",
        "stations": {
            "48875": "Vietnam", "48866": "Vietnam", "48900": "Vietnam",
        },
    },
}

# -------------------------------------------------------
# FETCH
# -------------------------------------------------------
def _fetch_station(station: str, parameter: str) -> list:
    """Fetch one station / parameter. Returns list of row dicts."""
    params = {
        "station": station, "parameter": parameter,
        "start": "01-01", "end": "12-31", "model": "0", "metric": "1",
    }
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("output", {})

    records = []
    for api_year in FETCH_YEARS:
        if api_year not in data:
            continue
        # Rename API "normals" key to the display label used in parquet
        label = "Normal (Maxar)" if api_year == "normals" else api_year
        for d in data[api_year]:
            rec = {"station": station, "year": label, "date": d["date"]}
            if parameter == "PRCP":
                rec["prcp"]     = d.get("prcp")
                rec["prcp_sum"] = d.get("prcp_sum")
            elif parameter == "TAVG":
                rec["tavg"] = d.get("tavg")
            elif parameter == "TMIN":
                rec["tmin"] = d.get("tmin")
            else:
                rec["tmax"] = d.get("tmax")
            records.append(rec)
    return records


def _fetch_origin(origin_name: str, cfg: dict) -> pd.DataFrame:
    """Fetch all stations for one origin — PRCP, TAVG, TMIN, TMAX."""
    from functools import reduce
    station_region = cfg["stations"]
    stations       = list(station_region.keys())
    buckets        = {"PRCP": [], "TAVG": [], "TMIN": [], "TMAX": []}
    errors         = []

    tasks = [(s, p) for s in stations for p in buckets]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_station, s, p): (s, p) for s, p in tasks}
        for fut in as_completed(futures):
            stn, param = futures[fut]
            try:
                buckets[param].extend(fut.result())
            except Exception as e:
                errors.append(f"{stn}/{param}: {e}")

    if errors:
        print(f"  {len(errors)} error(s) (first 3): {errors[:3]}")

    frames = {p: pd.DataFrame(rows) for p, rows in buckets.items() if rows}
    if not frames:
        return pd.DataFrame()

    df = reduce(lambda l, r: l.merge(r, on=["station", "year", "date"], how="outer"),
                frames.values())
    for col in ["prcp", "prcp_sum", "tavg", "tmin", "tmax"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["region"] = df["station"].map(station_region)
    return df[["station", "region", "year", "date", "prcp", "prcp_sum", "tavg", "tmin", "tmax"]]


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {PARQUET_DIR}\n")

    for origin_name, cfg in ORIGINS.items():
        n_stations = len(cfg["stations"])
        print(f"[{origin_name}]  {n_stations} stations × 2 params × {len(FETCH_YEARS)} years ...")
        df = _fetch_origin(origin_name, cfg)
        if df.empty:
            print(f"  No data returned — skipping.\n")
            continue
        out = PARQUET_DIR / cfg["file"]
        df.to_parquet(out, index=False)
        years_found = sorted(df["year"].unique())
        print(f"  {len(df):,} rows saved -> {cfg['file']}")
        print(f"  Years: {years_found}\n")

    print("Backfill complete.")


if __name__ == "__main__":
    main()
