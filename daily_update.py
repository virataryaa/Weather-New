"""
Coffee Weather — Daily Update
==============================
Refreshes the current calendar year's data in all parquet files.
Historical years (2016 to last year) are never re-fetched.

Schedule via Windows Task Scheduler — run once daily (e.g., 07:00 AM):
    Program : python
    Arguments: "C:\...\Weather\Coffee\daily_update.py"
"""

import requests
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
PARQUET_DIR  = Path(__file__).parent / "data"
API_URL      = "https://api.weatherdesk.xweather.com/2e621a7f-2b1e-4f3e-af6a-5a986a68b398/services/gwi/v1/timeseries"
MAX_WORKERS  = 20
CURRENT_YEAR = str(datetime.date.today().year)

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
    """Fetch current-year data for one station / parameter."""
    params = {
        "station": station, "parameter": parameter,
        "start": "01-01", "end": "12-31", "model": "0", "metric": "1",
    }
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("output", {})

    records = []
    if CURRENT_YEAR in data:
        for d in data[CURRENT_YEAR]:
            rec = {"station": station, "year": CURRENT_YEAR, "date": d["date"]}
            if parameter == "PRCP":
                rec["prcp"]     = d.get("prcp")
                rec["prcp_sum"] = d.get("prcp_sum")
            else:
                rec["tavg"] = d.get("tavg")
            records.append(rec)
    return records


def _update_origin(origin_name: str, cfg: dict):
    """Fetch current year → drop old current-year rows → append → save."""
    parquet_path   = PARQUET_DIR / cfg["file"]
    station_region = cfg["stations"]
    stations       = list(station_region.keys())

    if not parquet_path.exists():
        print(f"  Parquet not found. Run backfill.py first.")
        return

    prcp_rows, tavg_rows, errors = [], [], []
    tasks = [(s, p) for s in stations for p in ("PRCP", "TAVG")]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_station, s, p): (s, p) for s, p in tasks}
        for fut in as_completed(futures):
            stn, param = futures[fut]
            try:
                rows = fut.result()
                (prcp_rows if param == "PRCP" else tavg_rows).extend(rows)
            except Exception as e:
                errors.append(f"{stn}/{param}: {e}")

    if errors:
        print(f"  {len(errors)} error(s): {errors[:3]}")

    df_prcp = pd.DataFrame(prcp_rows)
    df_tavg = pd.DataFrame(tavg_rows)

    if df_prcp.empty and df_tavg.empty:
        print(f"  No data returned for {CURRENT_YEAR}.")
        return

    if df_prcp.empty:
        new_df = df_tavg.copy()
        new_df["prcp"] = pd.NA
        new_df["prcp_sum"] = pd.NA
    elif df_tavg.empty:
        new_df = df_prcp.copy()
        new_df["tavg"] = pd.NA
    else:
        new_df = df_prcp.merge(
            df_tavg[["station", "year", "date", "tavg"]],
            on=["station", "year", "date"],
            how="outer",
        )

    new_df["region"] = new_df["station"].map(station_region)
    new_df = new_df[["station", "region", "year", "date", "prcp", "prcp_sum", "tavg"]]

    # Load existing, remove stale current-year rows, append fresh data
    existing = pd.read_parquet(parquet_path)
    existing = existing[existing["year"] != CURRENT_YEAR]
    updated  = pd.concat([existing, new_df], ignore_index=True)
    updated.to_parquet(parquet_path, index=False)

    print(f"  {len(new_df):,} rows updated for {CURRENT_YEAR} → {cfg['file']}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    today = datetime.date.today()
    print(f"Daily update — {today}  (refreshing year {CURRENT_YEAR})\n")
    for origin_name, cfg in ORIGINS.items():
        print(f"[{origin_name}]")
        _update_origin(origin_name, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
