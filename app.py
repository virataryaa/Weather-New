import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Coffee Weather Dashboard", layout="wide")

# -------------------------------------------------------
# THEME
# -------------------------------------------------------
BG      = "#fafafa"
SURFACE = "#ffffff"
INK     = "#1d1d1f"
INK_2   = "#424245"
INK_3   = "#6e6e73"
INK_4   = "#aeaeb2"
BORDER  = "rgba(0,0,0,0.08)"
GRID    = "rgba(0,0,0,0.06)"
RED     = "#c0392b"
NAVY    = "#0a2463"
AVG_5Y_COLOR  = "#f39c12"   # amber  — 5-year average line
AVG_10Y_COLOR = "#9b59b6"   # purple — 10-year average line

ALL_YEAR_COLORS = {
    "2026": RED,        "2025": INK,        "2024": "#2980b9",
    "2023": "#27ae60",  "2022": "#8e44ad",  "2021": "#e67e22",
    "2020": "#16a085",  "2019": "#d35400",  "2018": "#7f8c8d",
    "2017": "#2c3e50",  "2016": "#a93226",
    "Normal (Maxar)": INK_4,
}

CROP_COLOR_PALETTE = [
    RED, INK, "#f1948a", "#82e0aa", INK_4,
    "#2980b9", "#e67e22", "#16a085", "#8e44ad", "#7f8c8d",
]

st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
      background:{BG} !important; color:{INK} !important;
  }}
  [data-testid="stSidebar"] {{
      background:{SURFACE} !important;
      border-right:1px solid {BORDER} !important;
  }}
  [data-testid="stSidebar"] * {{ color:{INK} !important; }}
  .block-container {{ padding-top:2rem !important; }}
  h2.section-header {{
      font-family: Helvetica Neue, sans-serif;
      font-size: 0.65rem; font-weight: 600;
      letter-spacing: .18em; text-transform: uppercase;
      color: {INK_3}; border-bottom: 1px solid {BORDER};
      padding-bottom: .5rem; margin: 2rem 0 0.2rem 0;
  }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# PARQUET CONFIG
# -------------------------------------------------------
PARQUET_DIR = Path(__file__).parent / "data"

FILE_MAP = {
    "Brazil":   "brazil.parquet",
    "Colombia": "colombia.parquet",
    "Honduras": "honduras.parquet",
    "Super 4":  "super4.parquet",
    "Vietnam":  "vietnam.parquet",
}

ALL_CAL_YEARS = [str(y) for y in range(2016, 2027)] + ["Normal (Maxar)"]

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_origin_data(origin_name: str, parameter: str) -> pd.DataFrame:
    parquet_path = PARQUET_DIR / FILE_MAP[origin_name]
    if not parquet_path.exists():
        st.error(f"Parquet file missing for {origin_name}. Run backfill.py first.")
        return pd.DataFrame()
    path_str = str(parquet_path).replace("\\", "/")
    if parameter == "PRCP":
        sql = f"SELECT station, region, year, date, prcp, prcp_sum FROM read_parquet('{path_str}') WHERE prcp IS NOT NULL OR prcp_sum IS NOT NULL"
    else:
        sql = f"SELECT station, region, year, date, tavg, tmin, tmax FROM read_parquet('{path_str}') WHERE tavg IS NOT NULL OR tmin IS NOT NULL OR tmax IS NOT NULL"
    return duckdb.query(sql).df()


# -------------------------------------------------------
# CROP YEAR HELPERS  (dynamic on start_month — all origins)
# -------------------------------------------------------
def crop_label(dt, sm):
    if sm == 1: return str(dt.year)
    if dt.month >= sm: return f"{dt.year % 100:02d}/{(dt.year+1) % 100:02d}"
    return f"{(dt.year-1) % 100:02d}/{dt.year % 100:02d}"

def _cy_sort_key(cy, sm):
    return int(cy) if sm == 1 else int(cy.split("/")[1])

def _min_cy(sm):
    return "2016" if sm == 1 else "15/16"

def crop_xdate(dt, sm):
    return pd.Timestamp(2000 if dt.month >= sm else 2001, dt.month, dt.day)

def crop_xaxis_dict(sm):
    start = pd.Timestamp(2000, sm, 1)
    end   = (start + pd.DateOffset(years=1)) - pd.Timedelta(days=1)
    return dict(tickformat="%b", dtick="M1", tick0=start.strftime("%Y-%m-%d"),
                range=[start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
                gridcolor=GRID, tickfont=dict(size=11, color=INK_3),
                linecolor=BORDER, zerolinecolor=BORDER)

def crop_month_order(sm):
    return [(sm - 1 + i) % 12 + 1 for i in range(12)]

def normals_xdate(month_int, day_int, sm):
    return pd.Timestamp(2000 if month_int >= sm else 2001, month_int, day_int)


# -------------------------------------------------------
# PROCESSING — ALL ORIGINS (Cocoa-style crop year)
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def process_prcp(raw: pd.DataFrame, today: pd.Timestamp, sm: int):
    df = raw[raw["date"] != "02-29"].copy()
    df_real    = df[df["year"] != "Normal (Maxar)"].copy()
    df_normals = df[df["year"] == "Normal (Maxar)"].copy()

    df_real["year_int"]  = df_real["year"].astype(int)
    df_real["full_date"] = pd.to_datetime(
        df_real["year_int"].astype(str) + "-" + df_real["date"], errors="coerce")
    df_real = df_real[df_real["full_date"].notna()].copy()
    df_real["crop_year"] = df_real["full_date"].apply(lambda d: crop_label(d, sm))
    df_real["xdate"]     = df_real["full_date"].apply(lambda d: crop_xdate(d, sm))
    df_real["tag"]       = df_real["full_date"].apply(
        lambda d: "realized" if d <= today else "forecast")

    real_daily = (
        df_real.groupby(["region", "crop_year", "xdate", "tag"], as_index=False)
        .agg(prcp_avg=("prcp", "mean"))
        .sort_values("xdate")
    )
    real_daily["cumulative_prcp"] = real_daily.groupby(
        ["region", "crop_year"])["prcp_avg"].cumsum()
    real_daily = real_daily[real_daily["crop_year"] >= _min_cy(sm)].copy()

    df_normals["month"] = df_normals["date"].str[:2].astype(int)
    df_normals["day"]   = df_normals["date"].str[3:].astype(int)
    df_normals["xdate"] = df_normals.apply(
        lambda r: normals_xdate(r["month"], r["day"], sm), axis=1)
    normals_daily = (
        df_normals.groupby(["region", "xdate"], as_index=False)
        .agg(prcp_avg=("prcp", "mean"))
        .sort_values("xdate")
    )
    normals_daily["cumulative_prcp"] = normals_daily.groupby("region")["prcp_avg"].cumsum()

    cys_sorted = sorted(real_daily["crop_year"].unique(), key=lambda cy: _cy_sort_key(cy, sm))
    cy_colors  = {cy: CROP_COLOR_PALETTE[i] if i < len(CROP_COLOR_PALETTE) else INK_4
                  for i, cy in enumerate(reversed(cys_sorted))}
    latest_cy  = cys_sorted[-1] if cys_sorted else None
    return real_daily, normals_daily, cys_sorted, cy_colors, latest_cy


@st.cache_data(show_spinner=False)
def process_temp(raw: pd.DataFrame, today: pd.Timestamp, sm: int):
    df = raw[raw["date"] != "02-29"].copy()
    df_real    = df[df["year"] != "Normal (Maxar)"].copy()
    df_normals = df[df["year"] == "Normal (Maxar)"].copy()

    df_real["year_int"]  = df_real["year"].astype(int)
    df_real["full_date"] = pd.to_datetime(
        df_real["year_int"].astype(str) + "-" + df_real["date"], errors="coerce")
    df_real = df_real[df_real["full_date"].notna()].copy()
    df_real["crop_year"] = df_real["full_date"].apply(lambda d: crop_label(d, sm))
    df_real["xdate"]     = df_real["full_date"].apply(lambda d: crop_xdate(d, sm))
    df_real["tag"]       = df_real["full_date"].apply(
        lambda d: "realized" if d <= today else "forecast")

    agg = {"tavg_avg": ("tavg", "mean")}
    if "tmin" in df_real.columns: agg["tmin_avg"] = ("tmin", "mean")
    if "tmax" in df_real.columns: agg["tmax_avg"] = ("tmax", "mean")
    real_daily = (
        df_real.groupby(["region", "crop_year", "xdate", "tag"], as_index=False)
        .agg(**agg).sort_values("xdate")
    )
    for col in ["tmin_avg", "tmax_avg"]:
        if col not in real_daily.columns: real_daily[col] = pd.NA
    real_daily = real_daily[real_daily["crop_year"] >= _min_cy(sm)].copy()

    df_normals["month"] = df_normals["date"].str[:2].astype(int)
    df_normals["day"]   = df_normals["date"].str[3:].astype(int)
    df_normals["xdate"] = df_normals.apply(
        lambda r: normals_xdate(r["month"], r["day"], sm), axis=1)
    n_agg = {"tavg_avg": ("tavg", "mean")}
    if "tmin" in df_normals.columns: n_agg["tmin_avg"] = ("tmin", "mean")
    if "tmax" in df_normals.columns: n_agg["tmax_avg"] = ("tmax", "mean")
    normals_daily = (
        df_normals.groupby(["region", "xdate"], as_index=False)
        .agg(**n_agg).sort_values("xdate")
    )
    for col in ["tmin_avg", "tmax_avg"]:
        if col not in normals_daily.columns: normals_daily[col] = pd.NA
    return real_daily, normals_daily


@st.cache_data(show_spinner=False)
def process_rolling(real_daily: pd.DataFrame, normals_daily: pd.DataFrame):
    parts = []
    for _, grp in real_daily.groupby(["region", "crop_year"]):
        grp = grp.sort_values("xdate").copy()
        grp["prcp_30d"] = grp.rolling("30D", on="xdate", min_periods=1)["prcp_avg"].sum()
        parts.append(grp)
    real_rolled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    nparts = []
    for _, grp in normals_daily.groupby("region"):
        grp = grp.sort_values("xdate").copy()
        grp["prcp_30d"] = grp.rolling("30D", on="xdate", min_periods=1)["prcp_avg"].sum()
        nparts.append(grp)
    normals_rolled = pd.concat(nparts, ignore_index=True) if nparts else pd.DataFrame()
    return real_rolled, normals_rolled


# -------------------------------------------------------
# PROCESSING — BRAZIL
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def process_brazil(raw_prcp: pd.DataFrame, today: pd.Timestamp, sm: int = 9):
    df = raw_prcp[raw_prcp["date"] != "02-29"].copy().reset_index(drop=True)
    df_real    = df[df["year"] != "Normal (Maxar)"].copy()
    df_normals = df[df["year"] == "Normal (Maxar)"].copy()

    df_real["year_int"]  = df_real["year"].astype(int)
    df_real["full_date"] = pd.to_datetime(
        df_real["year_int"].astype(str) + "-" + df_real["date"], errors="coerce"
    )
    df_real = df_real[df_real["full_date"].notna()].copy()

    df_real["crop_year"] = df_real["full_date"].apply(lambda dt: _brazil_crop_label(dt, sm))
    df_real["xdate"] = df_real["full_date"].apply(
        lambda dt: pd.Timestamp(2000 if dt.month >= sm else 2001, dt.month, dt.day)
    )
    df_real["tag"] = df_real["full_date"].apply(
        lambda d: "realized" if d <= today else "forecast"
    )
    real_daily = (
        df_real.groupby(["region", "crop_year", "xdate", "tag"], as_index=False)
        .agg(prcp_avg=("prcp", "mean"))
        .sort_values("xdate")
    )
    real_daily["cumulative_prcp"] = real_daily.groupby(["region", "crop_year"])["prcp_avg"].cumsum()
    real_daily = real_daily[real_daily["crop_year"] >= _brazil_min_cy(sm)].copy()

    crop_years_sorted = sorted(real_daily["crop_year"].unique(), key=lambda cy: _brazil_cy_sort_key(cy, sm))
    crop_year_colors  = {
        cy: CROP_COLOR_PALETTE[i] if i < len(CROP_COLOR_PALETTE) else INK_4
        for i, cy in enumerate(reversed(crop_years_sorted))
    }
    latest_crop_year = crop_years_sorted[-1] if crop_years_sorted else None

    df_normals["month"] = df_normals["date"].str[:2].astype(int)
    df_normals["day"]   = df_normals["date"].str[3:].astype(int)
    df_normals["xdate"] = df_normals.apply(
        lambda r: pd.Timestamp(2000 if r["month"] >= sm else 2001, r["month"], r["day"]), axis=1,
    )
    normals_daily = (
        df_normals.groupby(["region", "xdate"], as_index=False)
        .agg(prcp_avg=("prcp", "mean"))
        .sort_values("xdate")
    )
    normals_daily["cumulative_prcp"] = normals_daily.groupby("region")["prcp_avg"].cumsum()
    return real_daily, normals_daily, crop_years_sorted, crop_year_colors, latest_crop_year


@st.cache_data(show_spinner=False)
def process_brazil_temp(raw_temp: pd.DataFrame, today: pd.Timestamp, sm: int = 9):
    df = raw_temp[raw_temp["date"] != "02-29"].copy().reset_index(drop=True)
    df_real    = df[df["year"] != "Normal (Maxar)"].copy()
    df_normals = df[df["year"] == "Normal (Maxar)"].copy()

    df_real["year_int"]  = df_real["year"].astype(int)
    df_real["full_date"] = pd.to_datetime(
        df_real["year_int"].astype(str) + "-" + df_real["date"], errors="coerce"
    )
    df_real = df_real[df_real["full_date"].notna()].copy()

    df_real["crop_year"] = df_real["full_date"].apply(lambda dt: _brazil_crop_label(dt, sm))
    df_real["xdate"] = df_real["full_date"].apply(
        lambda dt: pd.Timestamp(2000 if dt.month >= sm else 2001, dt.month, dt.day)
    )
    df_real["tag"] = df_real["full_date"].apply(
        lambda d: "realized" if d <= today else "forecast"
    )
    agg_cols = {"tavg_avg": ("tavg", "mean")}
    if "tmin" in df_real.columns:
        agg_cols["tmin_avg"] = ("tmin", "mean")
    if "tmax" in df_real.columns:
        agg_cols["tmax_avg"] = ("tmax", "mean")
    real_daily = (
        df_real.groupby(["region", "crop_year", "xdate", "tag"], as_index=False)
        .agg(**agg_cols)
        .sort_values("xdate")
    )
    for col in ["tmin_avg", "tmax_avg"]:
        if col not in real_daily.columns:
            real_daily[col] = pd.NA
    real_daily = real_daily[real_daily["crop_year"] >= _brazil_min_cy(sm)].copy()

    df_normals["month"] = df_normals["date"].str[:2].astype(int)
    df_normals["day"]   = df_normals["date"].str[3:].astype(int)
    df_normals["xdate"] = df_normals.apply(
        lambda r: pd.Timestamp(2000 if r["month"] >= sm else 2001, r["month"], r["day"]), axis=1,
    )
    n_agg = {"tavg_avg": ("tavg", "mean")}
    if "tmin" in df_normals.columns:
        n_agg["tmin_avg"] = ("tmin", "mean")
    if "tmax" in df_normals.columns:
        n_agg["tmax_avg"] = ("tmax", "mean")
    normals_daily = (
        df_normals.groupby(["region", "xdate"], as_index=False)
        .agg(**n_agg)
        .sort_values("xdate")
    )
    for col in ["tmin_avg", "tmax_avg"]:
        if col not in normals_daily.columns:
            normals_daily[col] = pd.NA
    return real_daily, normals_daily


@st.cache_data(show_spinner=False)
def process_brazil_rolling(real_daily: pd.DataFrame, normals_daily: pd.DataFrame):
    real = real_daily.groupby(["region", "crop_year", "xdate"], as_index=False).agg(prcp_avg=("prcp_avg", "mean"))
    parts = []
    for _, grp in real.groupby(["region", "crop_year"]):
        grp = grp.sort_values("xdate").copy()
        grp["prcp_30d"] = grp.rolling("30D", on="xdate", min_periods=1)["prcp_avg"].sum()
        parts.append(grp)
    real_rolled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(real.columns) + ["prcp_30d"])
    nparts = []
    for _, grp in normals_daily.groupby("region"):
        grp = grp.sort_values("xdate").copy()
        grp["prcp_30d"] = grp.rolling("30D", on="xdate", min_periods=1)["prcp_avg"].sum()
        nparts.append(grp)
    normals_rolled = pd.concat(nparts, ignore_index=True) if nparts else pd.DataFrame(columns=list(normals_daily.columns) + ["prcp_30d"])
    return real_rolled, normals_rolled


# -------------------------------------------------------
# AVERAGE LINE COMPUTATION
# -------------------------------------------------------
def compute_brazil_precip_avg(real_daily: pd.DataFrame, n: int, crop_years_sorted: list):
    avg_cys = crop_years_sorted[-n:] if len(crop_years_sorted) >= n else crop_years_sorted
    avg_df  = (
        real_daily[real_daily["crop_year"].isin(avg_cys)]
        .groupby(["region", "xdate"], as_index=False)
        .agg(prcp_avg=("prcp_avg", "mean"))
        .sort_values("xdate")
    )
    avg_df["cumulative_prcp"] = avg_df.groupby("region")["prcp_avg"].cumsum()
    return avg_df


def compute_brazil_rolling_avg(real_rolled: pd.DataFrame, n: int, crop_years_sorted: list):
    avg_cys = crop_years_sorted[-n:] if len(crop_years_sorted) >= n else crop_years_sorted
    avg_df  = (
        real_rolled[real_rolled["crop_year"].isin(avg_cys)]
        .groupby(["region", "xdate"], as_index=False)
        .agg(prcp_30d=("prcp_30d", "mean"))
    )
    return avg_df


def compute_brazil_temp_avg(real_daily_temp: pd.DataFrame, n: int, crop_years_sorted: list):
    avg_cys = crop_years_sorted[-n:] if len(crop_years_sorted) >= n else crop_years_sorted
    avg_df  = (
        real_daily_temp[real_daily_temp["crop_year"].isin(avg_cys)]
        .groupby(["region", "xdate"], as_index=False)
        .agg(tavg_avg=("tavg_avg", "mean"))
    )
    return avg_df


def _avg_cys(cys_sorted, n):
    return cys_sorted[-n:] if len(cys_sorted) >= n else cys_sorted

def compute_precip_avg(real_daily, n, cys_sorted):
    avg = (real_daily[real_daily["crop_year"].isin(_avg_cys(cys_sorted, n))]
           .groupby(["region", "xdate"], as_index=False)
           .agg(prcp_avg=("prcp_avg", "mean")).sort_values("xdate"))
    avg["cumulative_prcp"] = avg.groupby("region")["prcp_avg"].cumsum()
    return avg

def compute_rolling_avg(real_rolled, n, cys_sorted):
    return (real_rolled[real_rolled["crop_year"].isin(_avg_cys(cys_sorted, n))]
            .groupby(["region", "xdate"], as_index=False)
            .agg(prcp_30d=("prcp_30d", "mean")))

def compute_temp_avg(real_daily_temp, n, cys_sorted):
    return (real_daily_temp[real_daily_temp["crop_year"].isin(_avg_cys(cys_sorted, n))]
            .groupby(["region", "xdate"], as_index=False)
            .agg(tavg_avg=("tavg_avg", "mean")))


# -------------------------------------------------------
# CHART HELPERS
# -------------------------------------------------------
def _base_layout(title: str, y_title: str, height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=INK, family="Helvetica Neue, sans-serif"),
                   x=0, xanchor="left"),
        xaxis=dict(gridcolor=GRID, tickfont=dict(size=11, color=INK_3),
                   linecolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(title=dict(text=y_title, font=dict(color=INK_2, size=12)),
                   gridcolor=GRID, tickfont=dict(size=11, color=INK_3),
                   linecolor=BORDER, zerolinecolor=BORDER),
        legend=dict(title=dict(text="Year", font=dict(color=INK_3, size=11)),
                    bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(size=11, color=INK_2)),
        plot_bgcolor=SURFACE, paper_bgcolor=BG,
        font=dict(color=INK, family="Helvetica Neue, sans-serif"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=INK, size=12)),
        height=height,
        margin=dict(l=60, r=20, t=50, b=55),
    )

_MONTH_NAMES_LIST = ["January","February","March","April","May","June",
                     "July","August","September","October","November","December"]
_MNUM_MAP = {n: i+1 for i, n in enumerate(_MONTH_NAMES_LIST)}

def _brazil_crop_label(dt, sm):
    if sm == 1:
        return str(dt.year)
    if dt.month >= sm:
        return f"{dt.year % 100:02d}/{(dt.year+1) % 100:02d}"
    return f"{(dt.year-1) % 100:02d}/{dt.year % 100:02d}"

def _brazil_cy_sort_key(cy, sm):
    return int(cy) if sm == 1 else int(cy.split("/")[1])

def _brazil_min_cy(sm):
    return "2016" if sm == 1 else "16/17"

def _brazil_crop_xaxis(sm):
    start = pd.Timestamp(2000, sm, 1)
    end   = (start + pd.DateOffset(years=1)) - pd.Timedelta(days=1)
    return dict(
        tickformat="%b", dtick="M1", tick0=start.strftime("%Y-%m-%d"),
        range=[start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        gridcolor=GRID, tickfont=dict(size=11, color=INK_3),
        linecolor=BORDER, zerolinecolor=BORDER,
    )

def _brazil_month_order(sm):
    return [(sm - 1 + i) % 12 + 1 for i in range(12)]

_BOX_EXPLANATION = (
    "<p style='font-size:.71rem;color:#6e6e73;font-style:italic;margin-top:.3rem'>"
    "<b>Reading the box:</b> The box spans Q1–Q3 (middle 50% of monthly totals across all crop years). "
    "The line inside is the median. Whiskers extend to 1.5× the interquartile range. "
    "<b>Dots</b> are outlier years that fall outside the whiskers — unusually wet or dry months."
    "</p>"
)


# -------------------------------------------------------
# CALENDAR-YEAR CHART BUILDERS  (crop-year / xdate architecture)
# -------------------------------------------------------
def build_cumulative(real_daily, normals_daily, region, cys_sorted, cy_colors,
                     latest_cy, selected_cys, sm, avg_df=None, avg_label="", avg_color=AVG_5Y_COLOR):
    df_r = real_daily[real_daily["region"] == region].copy()
    df_n = normals_daily[normals_daily["region"] == region].sort_values("xdate")
    fig  = go.Figure()
    for cy in cys_sorted:
        if cy not in selected_cys: continue
        color = cy_colors.get(cy, INK_4)
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        if cy == latest_cy:
            realized = cy_df[cy_df["tag"] == "realized"]
            forecast = cy_df[cy_df["tag"] == "forecast"]
            if not realized.empty:
                fig.add_trace(go.Scatter(x=realized["xdate"], y=realized["cumulative_prcp"],
                    mode="lines", name=cy, legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2.5, dash="solid"), connectgaps=True,
                    hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
            if not forecast.empty:
                lead = pd.concat([realized.iloc[[-1]], forecast]) if not realized.empty else forecast
                fig.add_trace(go.Scatter(x=lead["xdate"], y=lead["cumulative_prcp"],
                    mode="lines", name=f"{cy} fcst", legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2, dash="dot"), connectgaps=True,
                    hovertemplate=f"<b>{cy} fcst</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(x=cy_df["xdate"], y=cy_df["cumulative_prcp"],
                mode="lines", name=cy, line=dict(color=color, width=1.5), connectgaps=True,
                hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
    if not df_n.empty:
        fig.add_trace(go.Scatter(x=df_n["xdate"], y=df_n["cumulative_prcp"],
            mode="lines", name="Normal (Maxar)",
            line=dict(color=INK_4, width=2, dash="dash"), connectgaps=True,
            hovertemplate="<b>Normal</b>  %{x|%b %d}  %{y:.1f} mm<extra></extra>"))
    if avg_df is not None:
        d = avg_df[avg_df["region"] == region].sort_values("xdate")
        if not d.empty:
            fig.add_trace(go.Scatter(x=d["xdate"], y=d["cumulative_prcp"], mode="lines",
                name=avg_label, line=dict(color=avg_color, width=2.5, dash="dashdot"),
                hovertemplate=f"<b>{avg_label}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
    layout = _base_layout(f"Cumulative Precipitation  —  {region}", "mm")
    layout["xaxis"] = crop_xaxis_dict(sm)
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_rolling(real_rolled, normals_rolled, region, cys_sorted, cy_colors,
                  selected_cys, sm, avg_df=None, avg_label="", avg_color=AVG_5Y_COLOR):
    df_r = real_rolled[real_rolled["region"] == region].copy()
    df_n = normals_rolled[normals_rolled["region"] == region].sort_values("xdate")
    fig  = go.Figure()
    for cy in cys_sorted:
        if cy not in selected_cys: continue
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        fig.add_trace(go.Scatter(x=cy_df["xdate"], y=cy_df["prcp_30d"],
            mode="lines", name=cy, line=dict(color=cy_colors.get(cy, INK_4), width=1.8),
            connectgaps=True,
            hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
    if not df_n.empty:
        fig.add_trace(go.Scatter(x=df_n["xdate"], y=df_n["prcp_30d"],
            mode="lines", name="Normal (Maxar)",
            line=dict(color=INK_4, width=2, dash="dash"), connectgaps=True,
            hovertemplate="<b>Normal</b>  %{x|%b %d}  %{y:.1f} mm<extra></extra>"))
    if avg_df is not None:
        d = avg_df[avg_df["region"] == region].sort_values("xdate")
        if not d.empty:
            fig.add_trace(go.Scatter(x=d["xdate"], y=d["prcp_30d"], mode="lines",
                name=avg_label, line=dict(color=avg_color, width=2.5, dash="dashdot"),
                hovertemplate=f"<b>{avg_label}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
    layout = _base_layout(f"30-Day Rolling Precipitation  —  {region}", "Rolling Sum (mm)")
    layout["xaxis"] = crop_xaxis_dict(sm)
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_temperature(real_daily_temp, normals_daily_temp, region, cys_sorted, cy_colors,
                      latest_cy, selected_cys, sm, avg_df=None, avg_label="", avg_color=AVG_5Y_COLOR):
    df_r = real_daily_temp[real_daily_temp["region"] == region].copy()
    df_n = normals_daily_temp[normals_daily_temp["region"] == region].sort_values("xdate")
    fig  = go.Figure()
    hist = df_r[df_r["crop_year"] != latest_cy]
    if not hist.empty:
        mm = hist.groupby("xdate", as_index=False).agg(lo=("tavg_avg","min"), hi=("tavg_avg","max"))
        mm = mm.sort_values("xdate")
        fig.add_trace(go.Scatter(
            x=list(mm["xdate"]) + list(mm["xdate"])[::-1],
            y=list(mm["hi"]) + list(mm["lo"])[::-1],
            fill="toself", fillcolor="rgba(0,0,0,0.05)",
            line=dict(color="rgba(0,0,0,0)"), name="Hist. Range", hoverinfo="skip"))
    for cy in cys_sorted:
        if cy not in selected_cys: continue
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        color = cy_colors.get(cy, INK_4)
        realized = cy_df[cy_df["tag"] == "realized"]
        forecast = cy_df[cy_df["tag"] == "forecast"]
        if not realized.empty:
            fig.add_trace(go.Scatter(x=realized["xdate"], y=realized["tavg_avg"],
                mode="lines", name=cy, legendgroup=cy, showlegend=True,
                line=dict(color=color, width=2), connectgaps=True,
                hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} C<extra></extra>"))
        if not forecast.empty:
            lead = pd.concat([realized.iloc[[-1]], forecast]) if not realized.empty else forecast
            fig.add_trace(go.Scatter(x=lead["xdate"], y=lead["tavg_avg"],
                mode="lines", name=f"{cy} fcst", legendgroup=cy, showlegend=True,
                line=dict(color=color, width=1.5, dash="dot"), connectgaps=True,
                hovertemplate=f"<b>{cy} fcst</b>  %{{x|%b %d}}  %{{y:.1f}} C<extra></extra>"))
    if not df_n.empty:
        fig.add_trace(go.Scatter(x=df_n["xdate"], y=df_n["tavg_avg"],
            mode="lines", name="Normal (Maxar)",
            line=dict(color=INK_4, width=2, dash="dash"), connectgaps=True,
            hovertemplate="<b>Normal</b>  %{x|%b %d}  %{y:.1f} C<extra></extra>"))
    if avg_df is not None:
        d = avg_df[avg_df["region"] == region].sort_values("xdate")
        if not d.empty:
            fig.add_trace(go.Scatter(x=d["xdate"], y=d["tavg_avg"], mode="lines",
                name=avg_label, line=dict(color=avg_color, width=2.5, dash="dashdot"),
                hovertemplate=f"<b>{avg_label}</b>  %{{x|%b %d}}  %{{y:.1f}} C<extra></extra>"))
    layout = _base_layout(f"Average Temperature  —  {region}", "°C")
    layout["xaxis"] = crop_xaxis_dict(sm)
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


# -------------------------------------------------------
# BRAZIL CHART BUILDERS
# -------------------------------------------------------
def build_brazil_cumulative(real_daily, normals_daily, region, crop_years_sorted, crop_year_colors,
                             latest_crop_year, selected_crop_years, sm=9, avg_df=None, avg_label="", avg_color=AVG_5Y_COLOR):
    df_r = real_daily[real_daily["region"] == region].copy()
    df_n = normals_daily[normals_daily["region"] == region].sort_values("xdate")
    fig  = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        color = crop_year_colors.get(cy, INK_4)
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        if cy == latest_crop_year:
            realized = cy_df[cy_df["tag"] == "realized"]
            forecast = cy_df[cy_df["tag"] == "forecast"]
            if not realized.empty:
                fig.add_trace(go.Scatter(x=realized["xdate"], y=realized["cumulative_prcp"],
                    mode="lines", name=cy, legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2, dash="solid"), connectgaps=True,
                    hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
            if not forecast.empty:
                if not realized.empty:
                    forecast = pd.concat([realized.iloc[[-1]], forecast], ignore_index=True)
                fig.add_trace(go.Scatter(x=forecast["xdate"], y=forecast["cumulative_prcp"],
                    mode="lines", name=f"{cy} fcst", legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2, dash="dot"), connectgaps=True,
                    hovertemplate=f"<b>{cy} fcst</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(x=cy_df["xdate"], y=cy_df["cumulative_prcp"],
                mode="lines", name=cy, line=dict(color=color, width=2), connectgaps=True,
                hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
    if not df_n.empty:
        fig.add_trace(go.Scatter(x=df_n["xdate"], y=df_n["cumulative_prcp"],
            mode="lines", name="Normal (Maxar)",
            line=dict(color=INK_4, width=2.5, dash="dash"), connectgaps=True,
            hovertemplate="<b>Normal (Maxar)</b>  %{x|%b %d}  %{y:.1f} mm<extra></extra>"))
    if avg_df is not None:
        d = avg_df[avg_df["region"] == region].sort_values("xdate")
        if not d.empty:
            fig.add_trace(go.Scatter(x=d["xdate"], y=d["cumulative_prcp"],
                mode="lines", name=avg_label,
                line=dict(color=avg_color, width=2.5, dash="dashdot"), connectgaps=True,
                hovertemplate=f"<b>{avg_label}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
    layout = _base_layout(f"Cumulative Precipitation — Crop Year  ({region})", "mm")
    layout["xaxis"] = _brazil_crop_xaxis(sm)
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_temperature(real_daily_temp, normals_daily_temp, region, crop_years_sorted, crop_year_colors,
                              latest_crop_year, selected_crop_years, sm=9, avg_df=None, avg_label="", avg_color=AVG_5Y_COLOR):
    df_r = real_daily_temp[real_daily_temp["region"] == region].copy()
    df_n = normals_daily_temp[normals_daily_temp["region"] == region].sort_values("xdate")
    fig  = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        color = crop_year_colors.get(cy, INK_4)
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        if cy == latest_crop_year:
            realized = cy_df[cy_df["tag"] == "realized"]
            forecast = cy_df[cy_df["tag"] == "forecast"]
            if not realized.empty:
                fig.add_trace(go.Scatter(x=realized["xdate"], y=realized["tavg_avg"],
                    mode="lines", name=cy, legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2, dash="solid"), connectgaps=True,
                    hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} C<extra></extra>"))
            if not forecast.empty:
                if not realized.empty:
                    forecast = pd.concat([realized.iloc[[-1]], forecast], ignore_index=True)
                fig.add_trace(go.Scatter(x=forecast["xdate"], y=forecast["tavg_avg"],
                    mode="lines", name=f"{cy} fcst", legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2, dash="dot"), connectgaps=True,
                    hovertemplate=f"<b>{cy} fcst</b>  %{{x|%b %d}}  %{{y:.1f}} C<extra></extra>"))
        else:
            if not cy_df.empty:
                fig.add_trace(go.Scatter(x=cy_df["xdate"], y=cy_df["tavg_avg"],
                    mode="lines", name=cy, line=dict(color=color, width=2), connectgaps=True,
                    hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} C<extra></extra>"))
    if not df_n.empty:
        fig.add_trace(go.Scatter(x=df_n["xdate"], y=df_n["tavg_avg"],
            mode="lines", name="Normal (Maxar)",
            line=dict(color=INK_4, width=2.5, dash="dash"), connectgaps=True,
            hovertemplate="<b>Normal (Maxar)</b>  %{x|%b %d}  %{y:.1f} C<extra></extra>"))
    if avg_df is not None:
        d = avg_df[avg_df["region"] == region].sort_values("xdate")
        if not d.empty:
            fig.add_trace(go.Scatter(x=d["xdate"], y=d["tavg_avg"],
                mode="lines", name=avg_label,
                line=dict(color=avg_color, width=2.5, dash="dashdot"), connectgaps=True,
                hovertemplate=f"<b>{avg_label}</b>  %{{x|%b %d}}  %{{y:.1f}} C<extra></extra>"))
    layout = _base_layout(f"Average Temperature — Crop Year  ({region})", "°C")
    layout["xaxis"] = _brazil_crop_xaxis(sm)
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_rolling(real_rolled, normals_rolled, region, crop_years_sorted, crop_year_colors,
                          selected_crop_years, sm=9, avg_df=None, avg_label="", avg_color=AVG_5Y_COLOR):
    df_r = real_rolled[real_rolled["region"] == region].copy()
    df_n = normals_rolled[normals_rolled["region"] == region].sort_values("xdate")
    fig  = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        if not cy_df.empty:
            fig.add_trace(go.Scatter(x=cy_df["xdate"], y=cy_df["prcp_30d"],
                mode="lines", name=cy,
                line=dict(color=crop_year_colors.get(cy, INK_4), width=2), connectgaps=True,
                hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
    if not df_n.empty:
        fig.add_trace(go.Scatter(x=df_n["xdate"], y=df_n["prcp_30d"],
            mode="lines", name="Normal (Maxar)",
            line=dict(color=INK_4, width=2.5, dash="dash"), connectgaps=True,
            hovertemplate="<b>Normal (Maxar)</b>  %{x|%b %d}  %{y:.1f} mm<extra></extra>"))
    if avg_df is not None:
        d = avg_df[avg_df["region"] == region].sort_values("xdate")
        if not d.empty:
            fig.add_trace(go.Scatter(x=d["xdate"], y=d["prcp_30d"],
                mode="lines", name=avg_label,
                line=dict(color=avg_color, width=2.5, dash="dashdot"), connectgaps=True,
                hovertemplate=f"<b>{avg_label}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>"))
    layout = _base_layout(f"30-Day Rolling Precipitation — Crop Year  ({region})", "Rolling Sum (mm)")
    layout["xaxis"] = _brazil_crop_xaxis(sm)
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


# -------------------------------------------------------
# ADVANCED ANALYTICS — SHARED
# -------------------------------------------------------
_BRAZIL_MONTH_ORDER  = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
_MONTH_LABELS        = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
_BRAZIL_MONTH_LABELS = [_MONTH_LABELS[m] for m in _BRAZIL_MONTH_ORDER]
_CAL_MONTH_LABELS    = [_MONTH_LABELS[m] for m in range(1, 13)]


# -------------------------------------------------------
# ADVANCED ANALYTICS — BRAZIL
# -------------------------------------------------------
def build_brazil_precip_anomaly(real_daily, normals_daily, region, crop_years_sorted, crop_year_colors, selected_crop_years, sm=9):
    df_r = real_daily[real_daily["region"] == region].copy()
    df_n = normals_daily[normals_daily["region"] == region].copy()
    df_r["month"] = df_r["xdate"].dt.month
    df_n["month"] = df_n["xdate"].dt.month
    normals_m = df_n.groupby("month")["prcp_avg"].sum().reset_index().rename(columns={"prcp_avg": "normal_sum"})
    real_m    = df_r.groupby(["crop_year", "month"])["prcp_avg"].sum().reset_index().rename(columns={"prcp_avg": "real_sum"})
    merged    = real_m.merge(normals_m, on="month")
    merged["anomaly"]     = merged["real_sum"] - merged["normal_sum"]
    merged["month_label"] = merged["month"].map(_MONTH_LABELS)
    mo = _brazil_month_order(sm)
    merged["month_order"] = merged["month"].map({m: i for i, m in enumerate(mo)})
    merged = merged.sort_values("month_order")
    fig = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = merged[merged["crop_year"] == cy]
        if cy_df.empty:
            continue
        fig.add_trace(go.Bar(x=cy_df["month_label"], y=cy_df["anomaly"], name=cy,
            marker_color=crop_year_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y:+.1f}} mm<extra></extra>"))
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    layout = _base_layout(f"Monthly Precipitation Anomaly vs Normal (Maxar) ({region})", "mm above/below normal")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_dry_days(real_daily, region, crop_years_sorted, crop_year_colors, selected_crop_years, threshold=1.0, sm=9):
    mo   = _brazil_month_order(sm)
    df_r = real_daily[real_daily["region"] == region].copy()
    df_r["month"] = df_r["xdate"].dt.month
    results = []
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        for month in mo:
            m_df = cy_df[cy_df["month"] == month]
            if m_df.empty:
                continue
            max_run = curr = 0
            for v in (m_df["prcp_avg"] < threshold):
                curr = curr + 1 if v else 0
                max_run = max(max_run, curr)
            results.append({"crop_year": cy, "month": month, "max_dry_days": max_run})
    if not results:
        return go.Figure()
    res = pd.DataFrame(results)
    res["month_label"] = res["month"].map(_MONTH_LABELS)
    res["month_order"] = res["month"].map({m: i for i, m in enumerate(mo)})
    res = res.sort_values("month_order")
    fig = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty:
            continue
        fig.add_trace(go.Bar(x=cy_df["month_label"], y=cy_df["max_dry_days"], name=cy,
            marker_color=crop_year_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>"))
    layout = _base_layout(f"Max Consecutive Dry Days ({region}, <{threshold} mm/day)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_wet_days(real_daily, region, crop_years_sorted, crop_year_colors, selected_crop_years, threshold=1.0, sm=9):
    mo   = _brazil_month_order(sm)
    df_r = real_daily[real_daily["region"] == region].copy()
    df_r["month"] = df_r["xdate"].dt.month
    results = []
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        for month in mo:
            m_df = cy_df[cy_df["month"] == month]
            if m_df.empty:
                continue
            max_run = curr = 0
            for v in (m_df["prcp_avg"] >= threshold):
                curr = curr + 1 if v else 0
                max_run = max(max_run, curr)
            results.append({"crop_year": cy, "month": month, "max_wet_days": max_run})
    if not results:
        return go.Figure()
    res = pd.DataFrame(results)
    res["month_label"] = res["month"].map(_MONTH_LABELS)
    res["month_order"] = res["month"].map({m: i for i, m in enumerate(mo)})
    res = res.sort_values("month_order")
    fig = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty:
            continue
        fig.add_trace(go.Bar(x=cy_df["month_label"], y=cy_df["max_wet_days"], name=cy,
            marker_color=crop_year_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>"))
    layout = _base_layout(f"Max Consecutive Wet Days ({region}, >={threshold} mm/day)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_heat_stress(real_daily_temp, region, crop_years_sorted, crop_year_colors, selected_crop_years, threshold=28.0, sm=9):
    if real_daily_temp.empty:
        return go.Figure()
    mo   = _brazil_month_order(sm)
    df_r = real_daily_temp[real_daily_temp["region"] == region].copy()
    temp_col = "tmax_avg" if "tmax_avg" in df_r.columns and df_r["tmax_avg"].notna().any() else "tavg_avg"
    df_r["month"] = df_r["xdate"].dt.month
    results = []
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = df_r[df_r["crop_year"] == cy]
        for month in mo:
            m_df = cy_df[cy_df["month"] == month]
            if m_df.empty:
                continue
            results.append({"crop_year": cy, "month": month,
                             "stress_days": int((m_df[temp_col] > threshold).sum())})
    if not results:
        return go.Figure()
    res = pd.DataFrame(results)
    res["month_label"] = res["month"].map(_MONTH_LABELS)
    res["month_order"] = res["month"].map({m: i for i, m in enumerate(mo)})
    res = res.sort_values("month_order")
    fig = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty:
            continue
        fig.add_trace(go.Bar(x=cy_df["month_label"], y=cy_df["stress_days"], name=cy,
            marker_color=crop_year_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>"))
    temp_label = "max" if temp_col == "tmax_avg" else "avg"
    layout = _base_layout(f"Heat Stress Days ({region}, >{threshold}°C {temp_label})", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_frost_risk_days(real_daily_temp, region, crop_years_sorted, crop_year_colors, selected_crop_years, threshold=3.0):
    """Days per month where TMIN <= threshold (frost risk). Brazil only."""
    if real_daily_temp.empty or "tmin_avg" not in real_daily_temp.columns:
        return go.Figure()
    df_r = real_daily_temp[real_daily_temp["region"] == region].copy()
    if df_r["tmin_avg"].isna().all():
        return go.Figure()
    df_r["month"] = df_r["xdate"].dt.month
    # Only show frost-risk months (May–Sep for Brazil southern hemisphere)
    frost_months = [5, 6, 7, 8, 9]
    results = []
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = df_r[df_r["crop_year"] == cy]
        for month in frost_months:
            m_df = cy_df[cy_df["month"] == month]
            if m_df.empty:
                continue
            results.append({"crop_year": cy, "month": month,
                             "frost_days": int((m_df["tmin_avg"] <= threshold).sum())})
    if not results:
        return go.Figure()
    res = pd.DataFrame(results)
    month_labels = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
    res["month_label"] = res["month"].map(month_labels)
    res["month_order"] = res["month"]
    res = res.sort_values("month_order")
    fig = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty:
            continue
        fig.add_trace(go.Bar(x=cy_df["month_label"], y=cy_df["frost_days"], name=cy,
            marker_color=crop_year_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>"))
    layout = _base_layout(f"Frost Risk Days ({region}, TMIN <={threshold}°C)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = ["May", "Jun", "Jul", "Aug", "Sep"]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig



def build_brazil_monthly_boxplot(real_daily, normals_daily, region, crop_years_sorted, crop_year_colors, latest_crop_year, sm=9):
    mo   = _brazil_month_order(sm)
    df_r = real_daily[real_daily["region"] == region].copy()
    df_n = normals_daily[normals_daily["region"] == region].copy()
    df_r["month"] = df_r["xdate"].dt.month
    df_n["month"] = df_n["xdate"].dt.month
    monthly   = df_r.groupby(["crop_year", "month"])["prcp_avg"].sum().reset_index()
    normals_m = df_n.groupby("month")["prcp_avg"].sum().reset_index()
    monthly["month_label"]   = monthly["month"].map(_MONTH_LABELS)
    normals_m["month_label"] = normals_m["month"].map(_MONTH_LABELS)
    monthly["month_order"]   = monthly["month"].map({m: i for i, m in enumerate(mo)})
    normals_m["month_order"] = normals_m["month"].map({m: i for i, m in enumerate(mo)})
    monthly   = monthly.sort_values("month_order")
    normals_m = normals_m.sort_values("month_order")
    fig = go.Figure()
    fig.add_trace(go.Box(x=monthly["month_label"], y=monthly["prcp_avg"],
        name="Historical range", marker_color=INK_4,
        fillcolor="rgba(174,174,178,0.25)", line_color=INK_4,
        boxpoints="outliers", whiskerwidth=0.5))
    if latest_crop_year:
        curr = monthly[monthly["crop_year"] == latest_crop_year]
        fig.add_trace(go.Scatter(x=curr["month_label"], y=curr["prcp_avg"],
            mode="markers+lines", name=latest_crop_year,
            marker=dict(color=RED, size=8), line=dict(color=RED, width=2),
            hovertemplate=f"<b>{latest_crop_year}</b>  %{{x}}  %{{y:.1f}} mm<extra></extra>"))
    fig.add_trace(go.Scatter(x=normals_m["month_label"], y=normals_m["prcp_avg"],
        mode="lines", name="Normal (Maxar)",
        line=dict(color=INK_4, width=2, dash="dash"),
        hovertemplate="<b>Normal (Maxar)</b>  %{x}  %{y:.1f} mm<extra></extra>"))
    layout = _base_layout(f"Monthly Precipitation Distribution ({region})", "mm")
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    layout["legend"]["title"]["text"] = "Series"
    fig.update_layout(**layout)
    return fig


# -------------------------------------------------------
# ADVANCED ANALYTICS — CALENDAR YEAR  (crop-year / xdate)
# -------------------------------------------------------
def build_precip_anomaly(real_daily, normals_daily, region, cys_sorted, cy_colors, selected_cys, sm):
    df_r = real_daily[real_daily["region"] == region].copy()
    df_n = normals_daily[normals_daily["region"] == region].copy()
    mo   = crop_month_order(sm)
    df_r["month"] = df_r["xdate"].dt.month
    df_n["month"] = df_n["xdate"].dt.month
    normals_m = df_n.groupby("month")["prcp_avg"].sum().reset_index().rename(columns={"prcp_avg":"normal_sum"})
    real_m    = df_r.groupby(["crop_year","month"])["prcp_avg"].sum().reset_index().rename(columns={"prcp_avg":"real_sum"})
    merged    = real_m.merge(normals_m, on="month")
    merged["anomaly"]     = merged["real_sum"] - merged["normal_sum"]
    merged["month_label"] = merged["month"].map(_MONTH_LABELS)
    merged["month_order"] = merged["month"].map({m: i for i, m in enumerate(mo)})
    merged = merged.sort_values("month_order")
    fig = go.Figure()
    for cy in selected_cys:
        y_df = merged[merged["crop_year"] == cy]
        if y_df.empty: continue
        fig.add_trace(go.Bar(x=y_df["month_label"], y=y_df["anomaly"], name=cy,
            marker_color=cy_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y:+.1f}} mm<extra></extra>"))
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    layout = _base_layout(f"Monthly Precip Anomaly vs Normal ({region})", "mm above/below normal")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_monthly_boxplot_cal(real_daily, normals_daily, region, latest_cy, sm):
    df_r = real_daily[real_daily["region"] == region].copy()
    df_n = normals_daily[normals_daily["region"] == region].copy()
    mo   = crop_month_order(sm)
    df_r["month"] = df_r["xdate"].dt.month
    df_n["month"] = df_n["xdate"].dt.month
    monthly   = df_r.groupby(["crop_year","month"])["prcp_avg"].sum().reset_index()
    normals_m = df_n.groupby("month")["prcp_avg"].sum().reset_index()
    for d in [monthly, normals_m]:
        d["month_label"] = d["month"].map(_MONTH_LABELS)
        d["month_order"] = d["month"].map({m: i for i, m in enumerate(mo)})
        d.sort_values("month_order", inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Box(x=monthly["month_label"], y=monthly["prcp_avg"],
        name="Historical", marker_color=INK_4, fillcolor="rgba(174,174,178,0.25)",
        line_color=INK_4, boxpoints="outliers", whiskerwidth=0.5))
    curr = monthly[monthly["crop_year"] == latest_cy]
    if not curr.empty:
        fig.add_trace(go.Scatter(x=curr["month_label"], y=curr["prcp_avg"],
            mode="markers+lines", name=latest_cy,
            marker=dict(color=RED, size=8), line=dict(color=RED, width=2)))
    fig.add_trace(go.Scatter(x=normals_m["month_label"], y=normals_m["prcp_avg"],
        mode="lines", name="Normal (Maxar)", line=dict(color=INK_4, width=2, dash="dash")))
    layout = _base_layout(f"Monthly Precip Distribution ({region})", "mm")
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    fig.update_layout(**layout)
    return fig


def build_dry_days(real_daily, region, cys_sorted, cy_colors, selected_cys, threshold, sm):
    df_r = real_daily[real_daily["region"] == region].copy()
    mo   = crop_month_order(sm)
    df_r["month"] = df_r["xdate"].dt.month
    results = []
    for cy in cys_sorted:
        if cy not in selected_cys: continue
        cy_df = df_r[df_r["crop_year"] == cy]
        for month in mo:
            m_df = cy_df[cy_df["month"] == month]
            if m_df.empty: continue
            max_run = curr = 0
            for v in (m_df["prcp_avg"] < threshold):
                curr = curr + 1 if v else 0
                max_run = max(max_run, curr)
            results.append({"crop_year": cy, "month": month, "max_dry_days": max_run})
    if not results: return go.Figure()
    res = pd.DataFrame(results)
    res["month_label"] = res["month"].map(_MONTH_LABELS)
    res["month_order"] = res["month"].map({m: i for i, m in enumerate(mo)})
    res = res.sort_values("month_order")
    fig = go.Figure()
    for cy in selected_cys:
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty: continue
        fig.add_trace(go.Bar(x=cy_df["month_label"], y=cy_df["max_dry_days"], name=cy,
            marker_color=cy_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>"))
    layout = _base_layout(f"Max Consec. Dry Days ({region}, <{threshold} mm/day)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_wet_days(real_daily, region, cys_sorted, cy_colors, selected_cys, threshold, sm):
    df_r = real_daily[real_daily["region"] == region].copy()
    mo   = crop_month_order(sm)
    df_r["month"] = df_r["xdate"].dt.month
    results = []
    for cy in cys_sorted:
        if cy not in selected_cys: continue
        cy_df = df_r[df_r["crop_year"] == cy]
        for month in mo:
            m_df = cy_df[cy_df["month"] == month]
            if m_df.empty: continue
            max_run = curr = 0
            for v in (m_df["prcp_avg"] >= threshold):
                curr = curr + 1 if v else 0
                max_run = max(max_run, curr)
            results.append({"crop_year": cy, "month": month, "max_wet_days": max_run})
    if not results: return go.Figure()
    res = pd.DataFrame(results)
    res["month_label"] = res["month"].map(_MONTH_LABELS)
    res["month_order"] = res["month"].map({m: i for i, m in enumerate(mo)})
    res = res.sort_values("month_order")
    fig = go.Figure()
    for cy in selected_cys:
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty: continue
        fig.add_trace(go.Bar(x=cy_df["month_label"], y=cy_df["max_wet_days"], name=cy,
            marker_color=cy_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>"))
    layout = _base_layout(f"Max Consec. Wet Days ({region}, >={threshold} mm/day)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_heat_stress(real_daily_temp, region, cys_sorted, cy_colors, selected_cys, threshold, sm):
    df_r = real_daily_temp[real_daily_temp["region"] == region].copy()
    temp_col = "tmax_avg" if "tmax_avg" in df_r.columns and df_r["tmax_avg"].notna().any() else "tavg_avg"
    mo = crop_month_order(sm)
    df_r["month"] = df_r["xdate"].dt.month
    results = []
    for cy in cys_sorted:
        if cy not in selected_cys: continue
        cy_df = df_r[df_r["crop_year"] == cy]
        for month in mo:
            m_df = cy_df[cy_df["month"] == month]
            if m_df.empty: continue
            results.append({"crop_year": cy, "month": month,
                             "stress_days": int((m_df[temp_col] > threshold).sum())})
    if not results: return go.Figure()
    res = pd.DataFrame(results)
    res["month_label"] = res["month"].map(_MONTH_LABELS)
    res["month_order"] = res["month"].map({m: i for i, m in enumerate(mo)})
    res = res.sort_values("month_order")
    fig = go.Figure()
    for cy in selected_cys:
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty: continue
        fig.add_trace(go.Bar(x=cy_df["month_label"], y=cy_df["stress_days"], name=cy,
            marker_color=cy_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>"))
    t_label = "max" if temp_col == "tmax_avg" else "avg"
    layout = _base_layout(f"Heat Stress Days ({region}, >{threshold}°C {t_label})", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = [_MONTH_LABELS[m] for m in mo]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_frost_risk_days_cal(real_daily_temp, region, cys_sorted, cy_colors, selected_cys, threshold, sm):
    if real_daily_temp.empty or "tmin_avg" not in real_daily_temp.columns:
        return go.Figure()
    df_r = real_daily_temp[real_daily_temp["region"] == region].copy()
    if df_r["tmin_avg"].isna().all():
        return go.Figure()
    df_r["month"] = df_r["xdate"].dt.month
    frost_months = [5, 6, 7, 8, 9]
    results = []
    for cy in cys_sorted:
        if cy not in selected_cys: continue
        cy_df = df_r[df_r["crop_year"] == cy]
        for month in frost_months:
            m_df = cy_df[cy_df["month"] == month]
            if m_df.empty: continue
            results.append({"crop_year": cy, "month": month,
                             "frost_days": int((m_df["tmin_avg"] <= threshold).sum())})
    if not results: return go.Figure()
    res = pd.DataFrame(results)
    month_labels = {5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep"}
    res["month_label"] = res["month"].map(month_labels)
    res = res.sort_values("month")
    fig = go.Figure()
    for cy in selected_cys:
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty: continue
        fig.add_trace(go.Bar(x=cy_df["month_label"], y=cy_df["frost_days"], name=cy,
            marker_color=cy_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>"))
    layout = _base_layout(f"Frost Risk Days ({region}, TMIN <={threshold}°C)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = ["May","Jun","Jul","Aug","Sep"]
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


# -------------------------------------------------------
# RENDER CALENDAR-YEAR ORIGIN TAB  (Cocoa-style)
# -------------------------------------------------------
def render_cal_tab(origin_name, today, avg_option, sm):
    if not st.session_state.get(f"loaded_{origin_name}", False):
        st.info(f"Click below to load {origin_name} weather data.")
        if st.button(f"Load {origin_name} Data", key=f"btn_{origin_name}"):
            st.session_state[f"loaded_{origin_name}"] = True
            st.rerun()
        return

    c1, c2 = st.columns(2)
    with c1:
        with st.spinner(f"Loading {origin_name} precipitation..."):
            raw_prcp = load_origin_data(origin_name, "PRCP")
    with c2:
        with st.spinner(f"Loading {origin_name} temperature..."):
            raw_temp = load_origin_data(origin_name, "TAVG")

    if raw_prcp.empty:
        st.error(f"No data for {origin_name}. Run backfill.py first.")
        return

    real_daily, normals_daily, cys_sorted, cy_colors, latest_cy = \
        process_prcp(raw_prcp, today, sm)
    real_rolled, normals_rolled = process_rolling(real_daily, normals_daily)
    real_daily_temp, normals_daily_temp = \
        process_temp(raw_temp, today, sm) if not raw_temp.empty \
        else (pd.DataFrame(), pd.DataFrame())

    avg_cum = avg_roll = avg_temp = None
    avg_label = ""
    avg_color = AVG_5Y_COLOR
    if avg_option != "None":
        n         = 5 if "5" in avg_option else 10
        avg_color = AVG_5Y_COLOR if n == 5 else AVG_10Y_COLOR
        avg_label = f"{n}Y Avg"
        avg_cum  = compute_precip_avg(real_daily, n, cys_sorted)
        avg_roll = compute_rolling_avg(real_rolled, n, cys_sorted)
        if not real_daily_temp.empty:
            avg_temp = compute_temp_avg(real_daily_temp, n, cys_sorted)

    fc1, fc2 = st.columns(2)
    with fc1:
        regions_all = sorted(real_daily["region"].unique())
        sel_regions = st.multiselect("Sub-Regions", options=regions_all, default=regions_all,
                                     key=f"reg_{origin_name}")
    with fc2:
        default_cys = cys_sorted[-4:] if len(cys_sorted) >= 4 else cys_sorted
        sel_cys = st.multiselect("Crop Years", options=cys_sorted, default=default_cys,
                                 key=f"cy_{origin_name}")

    if not sel_regions:
        st.warning("Select at least one region.")
        return

    for region in sel_regions:
        st.markdown(f"<h2 class='section-header'>Cumulative Precipitation &nbsp;—&nbsp; {region}</h2>",
                    unsafe_allow_html=True)
        st.plotly_chart(build_cumulative(real_daily, normals_daily, region, cys_sorted,
            cy_colors, latest_cy, sel_cys, sm, avg_cum, avg_label, avg_color),
            use_container_width=True, key=f"cum_{origin_name}_{region}")

        if not real_daily_temp.empty:
            st.markdown(f"<h2 class='section-header'>Average Temperature &nbsp;—&nbsp; {region}</h2>",
                        unsafe_allow_html=True)
            st.plotly_chart(build_temperature(real_daily_temp, normals_daily_temp, region,
                cys_sorted, cy_colors, latest_cy, sel_cys, sm, avg_temp, avg_label, avg_color),
                use_container_width=True, key=f"tmp_{origin_name}_{region}")

        st.markdown(f"<h2 class='section-header'>30-Day Rolling Precipitation &nbsp;—&nbsp; {region}</h2>",
                    unsafe_allow_html=True)
        st.plotly_chart(build_rolling(real_rolled, normals_rolled, region, cys_sorted,
            cy_colors, sel_cys, sm, avg_roll, avg_label, avg_color),
            use_container_width=True, key=f"rol_{origin_name}_{region}")

        with st.expander("Advanced Analytics", expanded=False):
            th_col1, th_col2, th_col3, th_col4 = st.columns(4)
            with th_col1:
                dry_thr = st.number_input("Dry threshold (mm)", min_value=0.0,
                    value=1.0, step=0.5, key=f"dry_thr_{origin_name}_{region}")
            with th_col2:
                wet_thr = st.number_input("Wet threshold (mm)", min_value=0.0,
                    value=1.0, step=0.5, key=f"wet_thr_{origin_name}_{region}")
            with th_col3:
                heat_thr = st.number_input("Heat stress threshold (°C)", min_value=20.0,
                    max_value=45.0, value=32.0, step=0.5, key=f"heat_thr_{origin_name}_{region}")
            with th_col4:
                frost_thr = st.number_input("Frost risk TMIN (°C)", min_value=-10.0,
                    max_value=10.0, value=3.0, step=0.5, key=f"frost_thr_{origin_name}_{region}")

            c_a, c_b = st.columns(2)
            with c_a:
                st.plotly_chart(build_precip_anomaly(real_daily, normals_daily, region,
                    cys_sorted, cy_colors, sel_cys, sm),
                    use_container_width=True, key=f"anom_{origin_name}_{region}")
                st.plotly_chart(build_dry_days(real_daily, region, cys_sorted, cy_colors,
                    sel_cys, dry_thr, sm),
                    use_container_width=True, key=f"dry_{origin_name}_{region}")
            with c_b:
                st.plotly_chart(build_monthly_boxplot_cal(real_daily, normals_daily, region,
                    latest_cy, sm),
                    use_container_width=True, key=f"box_{origin_name}_{region}")
                st.markdown(_BOX_EXPLANATION, unsafe_allow_html=True)
                st.plotly_chart(build_wet_days(real_daily, region, cys_sorted, cy_colors,
                    sel_cys, wet_thr, sm),
                    use_container_width=True, key=f"wet_{origin_name}_{region}")

            if not real_daily_temp.empty:
                c_heat, c_frost = st.columns(2)
                with c_heat:
                    st.plotly_chart(build_heat_stress(real_daily_temp, region, cys_sorted,
                        cy_colors, sel_cys, heat_thr, sm),
                        use_container_width=True, key=f"heat_{origin_name}_{region}")
                with c_frost:
                    st.plotly_chart(build_frost_risk_days_cal(real_daily_temp, region, cys_sorted,
                        cy_colors, sel_cys, frost_thr, sm),
                        use_container_width=True, key=f"frost_{origin_name}_{region}")


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.markdown(
    f"<h1 style='font-family:Helvetica Neue,sans-serif;font-weight:600;"
    f"color:{NAVY};letter-spacing:-.02em;margin-bottom:.15rem'>"
    f"Coffee Weather Dashboard</h1>"
    f"<p style='color:{INK_4};font-size:.78rem;margin-top:0'>"
    f"Maxar XWeather API &nbsp;·&nbsp; Parquet / DuckDB &nbsp;·&nbsp; Refreshed daily</p>",
    unsafe_allow_html=True,
)

today = pd.Timestamp.today().normalize()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        f"<p style='font-size:.65rem;font-weight:700;letter-spacing:.14em;"
        f"text-transform:uppercase;color:{INK_3};margin-bottom:.4rem'>Crop Year Start</p>",
        unsafe_allow_html=True,
    )
    crop_start_name = st.selectbox(
        "Crop Year Start Month", _MONTH_NAMES_LIST, index=0,   # January default
        label_visibility="collapsed",
    )
    sm = _MNUM_MAP[crop_start_name]

    st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:.8rem 0'>",
                unsafe_allow_html=True)
    st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:.8rem 0'>",
                unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:.65rem;font-weight:700;letter-spacing:.14em;"
        f"text-transform:uppercase;color:{INK_3};margin-bottom:.4rem'>Historical Average</p>",
        unsafe_allow_html=True,
    )
    avg_option = st.radio(
        "Show average line",
        options=["None", "Last 5 Years", "Last 10 Years"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:.8rem 0'>",
                unsafe_allow_html=True)
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:.8rem 0'>",
                unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:.65rem;font-weight:700;letter-spacing:.14em;"
        f"text-transform:uppercase;color:{INK_3};margin-bottom:.4rem'>Avg line colors</p>",
        unsafe_allow_html=True,
    )
    for label, col in [("5Y Avg", AVG_5Y_COLOR), ("10Y Avg", AVG_10Y_COLOR)]:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            f"<span style='width:12px;height:12px;border-radius:2px;background:{col};"
            f"display:inline-block'></span>"
            f"<span style='font-size:.82rem;color:{INK_2}'>{label}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown(f"<p style='font-size:.75rem;color:{INK_4};margin-top:.8rem'>"
                f"Today: {today.strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)

# ---------- TABS ----------
tab_brazil, tab_colombia, tab_honduras, tab_super4, tab_vietnam = st.tabs([
    "Brazil", "Colombia", "Honduras", "Super 4", "Vietnam",
])

# ---- BRAZIL ----
with tab_brazil:
    if not st.session_state.get("brazil_loaded", False):
        st.info("Click below to load Brazil weather data.")
        if st.button("Load Brazil Data", key="load_brazil"):
            st.session_state.brazil_loaded = True
            st.rerun()
    else:
        c1, c2 = st.columns(2)
        with c1:
            with st.spinner("Loading Brazil precipitation..."):
                raw_brazil_prcp = load_origin_data("Brazil", "PRCP")
        with c2:
            with st.spinner("Loading Brazil temperature..."):
                raw_brazil_temp = load_origin_data("Brazil", "TAVG")

        if raw_brazil_prcp.empty:
            st.error("No data for Brazil. Run backfill.py first.")
        else:
            real_daily, normals_daily, crop_years_sorted, crop_year_colors, latest_cy = \
                process_brazil(raw_brazil_prcp, today, sm)

            real_daily_temp, normals_daily_temp = \
                process_brazil_temp(raw_brazil_temp, today, sm) if not raw_brazil_temp.empty \
                else (pd.DataFrame(), pd.DataFrame())

            real_rolled, normals_rolled = process_brazil_rolling(real_daily, normals_daily)

            # Average line computation (uses full crop year data before user filter)
            avg_cum_df = avg_roll_df = avg_temp_df = None
            avg_label  = ""
            avg_color  = AVG_5Y_COLOR
            if avg_option != "None":
                n         = 5 if "5" in avg_option else 10
                avg_color = AVG_5Y_COLOR if n == 5 else AVG_10Y_COLOR
                avg_label = f"{n}Y Avg"
                avg_cum_df  = compute_brazil_precip_avg(real_daily, n, crop_years_sorted)
                avg_roll_df = compute_brazil_rolling_avg(real_rolled, n, crop_years_sorted)
                if not real_daily_temp.empty:
                    avg_temp_df = compute_brazil_temp_avg(real_daily_temp, n, crop_years_sorted)

            # Filters row
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                regions_all = sorted(real_daily["region"].unique())
                selected_regions_brazil = st.multiselect(
                    "Sub-Regions", options=regions_all, default=regions_all,
                    key="brazil_region_filter",
                )
            with filter_col2:
                default_cys = crop_years_sorted[-4:] if len(crop_years_sorted) >= 4 else crop_years_sorted
                selected_crop_years = st.multiselect(
                    "Crop Years", options=crop_years_sorted, default=default_cys,
                    key="brazil_cy_filter",
                )

            if not selected_regions_brazil:
                st.warning("Select at least one sub-region.")
            else:
                for region in selected_regions_brazil:
                    st.markdown(f"<h2 class='section-header'>Cumulative Precipitation &nbsp;—&nbsp; {region}</h2>",
                                unsafe_allow_html=True)
                    st.plotly_chart(
                        build_brazil_cumulative(real_daily, normals_daily, region,
                            crop_years_sorted, crop_year_colors, latest_cy, selected_crop_years,
                            sm, avg_cum_df, avg_label, avg_color),
                        use_container_width=True, key=f"bra_cum_{region}")

                    if not real_daily_temp.empty:
                        st.markdown(f"<h2 class='section-header'>Average Temperature &nbsp;—&nbsp; {region}</h2>",
                                    unsafe_allow_html=True)
                        st.plotly_chart(
                            build_brazil_temperature(real_daily_temp, normals_daily_temp, region,
                                crop_years_sorted, crop_year_colors, latest_cy, selected_crop_years,
                                sm, avg_temp_df, avg_label, avg_color),
                            use_container_width=True, key=f"bra_tmp_{region}")

                    st.markdown(f"<h2 class='section-header'>30-Day Rolling Precipitation &nbsp;—&nbsp; {region}</h2>",
                                unsafe_allow_html=True)
                    st.plotly_chart(
                        build_brazil_rolling(real_rolled, normals_rolled, region,
                            crop_years_sorted, crop_year_colors, selected_crop_years,
                            sm, avg_roll_df, avg_label, avg_color),
                        use_container_width=True, key=f"bra_rol_{region}")

                    with st.expander("Advanced Analytics", expanded=False):
                        # Threshold inputs
                        th_col1, th_col2, th_col3, th_col4 = st.columns(4)
                        with th_col1:
                            dry_thr = st.number_input("Dry threshold (mm)", min_value=0.0,
                                value=1.0, step=0.5, key=f"bra_dry_thr_{region}")
                        with th_col2:
                            wet_thr = st.number_input("Wet threshold (mm)", min_value=0.0,
                                value=1.0, step=0.5, key=f"bra_wet_thr_{region}")
                        with th_col3:
                            heat_thr = st.number_input("Heat stress threshold (°C)", min_value=20.0,
                                max_value=45.0, value=32.0, step=0.5, key=f"bra_heat_thr_{region}")
                        with th_col4:
                            frost_thr = st.number_input("Frost risk TMIN (°C)", min_value=-10.0,
                                max_value=10.0, value=3.0, step=0.5, key=f"bra_frost_thr_{region}")

                        c_a, c_b = st.columns(2)
                        with c_a:
                            st.plotly_chart(build_brazil_precip_anomaly(
                                real_daily, normals_daily, region,
                                crop_years_sorted, crop_year_colors, selected_crop_years, sm),
                                use_container_width=True, key=f"bra_anom_{region}")
                            st.plotly_chart(build_brazil_dry_days(
                                real_daily, region, crop_years_sorted,
                                crop_year_colors, selected_crop_years, dry_thr, sm),
                                use_container_width=True, key=f"bra_dry_{region}")
                        with c_b:
                            st.plotly_chart(build_brazil_monthly_boxplot(
                                real_daily, normals_daily, region,
                                crop_years_sorted, crop_year_colors, latest_cy, sm),
                                use_container_width=True, key=f"bra_box_{region}")
                            st.markdown(_BOX_EXPLANATION, unsafe_allow_html=True)
                            st.plotly_chart(build_brazil_wet_days(
                                real_daily, region, crop_years_sorted,
                                crop_year_colors, selected_crop_years, wet_thr, sm),
                                use_container_width=True, key=f"bra_wet_{region}")

                        c_heat, c_frost = st.columns(2)
                        with c_heat:
                            st.plotly_chart(build_brazil_heat_stress(
                                real_daily_temp, region, crop_years_sorted,
                                crop_year_colors, selected_crop_years, heat_thr, sm),
                                use_container_width=True, key=f"bra_heat_{region}")
                        with c_frost:
                            st.plotly_chart(build_brazil_frost_risk_days(
                                real_daily_temp, region, crop_years_sorted,
                                crop_year_colors, selected_crop_years, frost_thr),
                                use_container_width=True, key=f"bra_frost_{region}")

# ---- CALENDAR-YEAR ORIGINS ----
with tab_colombia:
    render_cal_tab("Colombia", today, avg_option, sm)

with tab_honduras:
    render_cal_tab("Honduras", today, avg_option, sm)

with tab_super4:
    render_cal_tab("Super 4", today, avg_option, sm)

with tab_vietnam:
    render_cal_tab("Vietnam", today, avg_option, sm)
