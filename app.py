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

ALL_YEAR_COLORS = {
    "2026": RED,
    "2025": INK,
    "2024": "#2980b9",
    "2023": "#27ae60",
    "2022": "#8e44ad",
    "2021": "#e67e22",
    "2020": "#16a085",
    "2019": "#d35400",
    "2018": "#7f8c8d",
    "2017": "#2c3e50",
    "2016": "#a93226",
    "Normal (Maxar)": INK_4,
}

# Brazil crop year: most recent first
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

CALENDAR_YEARS = [
    "2016", "2017", "2018", "2019", "2020",
    "2021", "2022", "2023", "2024", "2025", "2026",
    "Normal (Maxar)",
]

# -------------------------------------------------------
# DATA LOADING — DuckDB reads parquet files
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_origin_data(origin_name: str, parameter: str) -> pd.DataFrame:
    parquet_path = PARQUET_DIR / FILE_MAP[origin_name]
    if not parquet_path.exists():
        st.error(f"Parquet file missing for {origin_name}. Run backfill.py first.")
        return pd.DataFrame()

    path_str = str(parquet_path).replace("\\", "/")

    if parameter == "PRCP":
        sql = f"""
            SELECT station, region, year, date, prcp, prcp_sum
            FROM read_parquet('{path_str}')
            WHERE prcp IS NOT NULL OR prcp_sum IS NOT NULL
        """
    else:
        sql = f"""
            SELECT station, region, year, date, tavg
            FROM read_parquet('{path_str}')
            WHERE tavg IS NOT NULL
        """
    return duckdb.query(sql).df()


# -------------------------------------------------------
# CALENDAR-YEAR PROCESSING  (Colombia / Honduras / Super4 / Vietnam)
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def process_precipitation(raw_df: pd.DataFrame, today: pd.Timestamp):
    df = raw_df[raw_df["date"] != "02-29"].copy().reset_index(drop=True)
    latest_year = df[df["year"] != "Normal (Maxar)"]["year"].astype(int).max()
    df["full_date"] = pd.to_datetime(df.apply(
        lambda r: f"{latest_year}-{r['date']}" if r["year"] == "Normal (Maxar)" else f"{r['year']}-{r['date']}",
        axis=1,
    ))
    df["tag"] = df.apply(
        lambda r: "realized" if r["year"] == "Normal (Maxar)" or r["full_date"] <= today else "forecast",
        axis=1,
    )
    daily_avg = (
        df.groupby(["region", "year", "date"])
        .agg(prcp_avg=("prcp", "mean"), prcp_sum_avg=("prcp_sum", "mean"))
        .reset_index()
    )
    daily_avg["full_date"] = pd.to_datetime(daily_avg.apply(
        lambda r: f"{latest_year}-{r['date']}" if r["year"] == "Normal (Maxar)" else f"{r['year']}-{r['date']}",
        axis=1,
    ))
    daily_avg["tag"] = daily_avg.apply(
        lambda r: "realized" if r["year"] == "Normal (Maxar)" or r["full_date"] <= today else "forecast",
        axis=1,
    )
    return daily_avg


@st.cache_data(show_spinner=False)
def process_temperature(raw_df: pd.DataFrame, today: pd.Timestamp):
    df = raw_df[raw_df["date"] != "02-29"].copy().reset_index(drop=True)
    daily_avg = (
        df.groupby(["region", "year", "date"])
        .agg(tavg_avg=("tavg", "mean"))
        .reset_index()
    )
    daily_avg["tag"] = daily_avg.apply(
        lambda r: "realized" if r["year"] == "Normal (Maxar)"
        else ("realized" if pd.to_datetime(f"{r['year']}-{r['date']}") <= today else "forecast"),
        axis=1,
    )
    return daily_avg


@st.cache_data(show_spinner=False)
def process_rolling(daily_avg: pd.DataFrame, today: pd.Timestamp):
    df = daily_avg.copy()
    df["full_date"] = pd.to_datetime(df["full_date"])
    df = df[df["date"] != "02-29"].copy().reset_index(drop=True)
    df["rolling_date"] = df.apply(
        lambda r: pd.to_datetime(f"1950-{r['date']}") if r["year"] == "Normal (Maxar)" else r["full_date"],
        axis=1,
    )

    normals_df  = df[df["year"] == "Normal (Maxar)"].copy()
    normals_pre = normals_df.copy()
    normals_pre["rolling_date"] = normals_pre["rolling_date"].apply(
        lambda d: pd.Timestamp(year=1949, month=d.month, day=d.day)
    )
    normals_ext = pd.concat([normals_pre, normals_df], ignore_index=True).sort_values(["region", "rolling_date"])
    agg_normals = normals_ext.groupby(["region", "year", "rolling_date"], as_index=False).agg(
        prcp_avg=("prcp_avg", "mean"), prcp_sum_avg=("prcp_sum_avg", "mean")
    )

    n_parts = []
    for _, grp in agg_normals.groupby("region"):
        grp = grp.sort_values("rolling_date").copy()
        grp["prcp_avg_30d_sum"] = grp.rolling("30D", on="rolling_date", min_periods=1)["prcp_avg"].sum()
        n_parts.append(grp)
    agg_normals = pd.concat(n_parts, ignore_index=True) if n_parts else agg_normals.assign(prcp_avg_30d_sum=pd.NA)
    agg_normals["xdate"] = agg_normals["rolling_date"].apply(
        lambda d: pd.Timestamp(year=2026, month=d.month, day=d.day)
    )
    agg_normals_final = agg_normals[agg_normals["rolling_date"].dt.year != 1949].reset_index(drop=True)

    other_df  = df[df["year"] != "Normal (Maxar)"].copy()
    agg_other = other_df.groupby(["region", "year", "rolling_date"], as_index=False).agg(
        prcp_avg=("prcp_avg", "mean"), prcp_sum_avg=("prcp_sum_avg", "mean")
    )
    o_parts = []
    for _, grp in agg_other.groupby("region"):
        grp = grp.sort_values("rolling_date").copy()
        grp["prcp_avg_30d_sum"] = grp.rolling("30D", on="rolling_date", min_periods=1)["prcp_avg"].sum()
        o_parts.append(grp)
    agg_other = pd.concat(o_parts, ignore_index=True) if o_parts else agg_other.assign(prcp_avg_30d_sum=pd.NA)
    agg_other["xdate"] = agg_other["rolling_date"].apply(
        lambda d: pd.Timestamp(year=2026, month=d.month, day=d.day)
    )
    return (
        pd.concat([agg_other, agg_normals_final], ignore_index=True)
        .sort_values(["region", "rolling_date"])
        .reset_index(drop=True)
    )


# -------------------------------------------------------
# BRAZIL PROCESSING  (crop year Sep-Aug)
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def process_brazil(raw_prcp: pd.DataFrame, today: pd.Timestamp):
    df = raw_prcp[raw_prcp["date"] != "02-29"].copy().reset_index(drop=True)

    df_real    = df[df["year"] != "Normal (Maxar)"].copy()
    df_normals = df[df["year"] == "Normal (Maxar)"].copy()

    df_real["year_int"]  = df_real["year"].astype(int)
    df_real["full_date"] = pd.to_datetime(
        df_real["year_int"].astype(str) + "-" + df_real["date"], errors="coerce"
    )
    df_real = df_real[df_real["full_date"].notna()].copy()

    def _crop_label(dt):
        if dt.month >= 9:
            return f"{dt.year % 100:02d}/{(dt.year + 1) % 100:02d}"
        return f"{(dt.year - 1) % 100:02d}/{dt.year % 100:02d}"

    df_real["crop_year"] = df_real["full_date"].apply(_crop_label)
    df_real["xdate"] = df_real["full_date"].apply(
        lambda dt: pd.Timestamp(2000 if dt.month >= 9 else 2001, dt.month, dt.day)
    )
    df_real["tag"] = df_real["full_date"].apply(
        lambda d: "realized" if d <= today else "forecast"
    )

    real_daily = (
        df_real.groupby(["region", "crop_year", "xdate", "tag"], as_index=False)
        .agg(prcp_avg=("prcp", "mean"))
        .sort_values("xdate")
    )
    real_daily["cumulative_prcp"] = (
        real_daily.groupby(["region", "crop_year"])["prcp_avg"].cumsum()
    )
    # Keep crop years from 15/16 onwards (covers 10 years of data)
    real_daily = real_daily[real_daily["crop_year"] >= "15/16"].copy()

    def _cy_end(cy):
        return int(cy.split("/")[1])

    crop_years_sorted = sorted(real_daily["crop_year"].unique(), key=_cy_end)
    crop_year_colors  = {
        cy: CROP_COLOR_PALETTE[i] if i < len(CROP_COLOR_PALETTE) else INK_4
        for i, cy in enumerate(reversed(crop_years_sorted))
    }
    latest_crop_year = crop_years_sorted[-1] if crop_years_sorted else None

    df_normals["month"] = df_normals["date"].str[:2].astype(int)
    df_normals["day"]   = df_normals["date"].str[3:].astype(int)
    df_normals["xdate"] = df_normals.apply(
        lambda r: pd.Timestamp(2000 if r["month"] >= 9 else 2001, r["month"], r["day"]),
        axis=1,
    )
    normals_daily = (
        df_normals.groupby(["region", "xdate"], as_index=False)
        .agg(prcp_avg=("prcp", "mean"))
        .sort_values("xdate")
    )
    normals_daily["cumulative_prcp"] = (
        normals_daily.groupby("region")["prcp_avg"].cumsum()
    )

    return real_daily, normals_daily, crop_years_sorted, crop_year_colors, latest_crop_year


@st.cache_data(show_spinner=False)
def process_brazil_temp(raw_temp: pd.DataFrame, today: pd.Timestamp):
    df = raw_temp[raw_temp["date"] != "02-29"].copy().reset_index(drop=True)
    df_real    = df[df["year"] != "Normal (Maxar)"].copy()
    df_normals = df[df["year"] == "Normal (Maxar)"].copy()

    df_real["year_int"]  = df_real["year"].astype(int)
    df_real["full_date"] = pd.to_datetime(
        df_real["year_int"].astype(str) + "-" + df_real["date"], errors="coerce"
    )
    df_real = df_real[df_real["full_date"].notna()].copy()

    def _crop_label(dt):
        if dt.month >= 9:
            return f"{dt.year % 100:02d}/{(dt.year + 1) % 100:02d}"
        return f"{(dt.year - 1) % 100:02d}/{dt.year % 100:02d}"

    df_real["crop_year"] = df_real["full_date"].apply(_crop_label)
    df_real["xdate"] = df_real["full_date"].apply(
        lambda dt: pd.Timestamp(2000 if dt.month >= 9 else 2001, dt.month, dt.day)
    )
    df_real["tag"] = df_real["full_date"].apply(
        lambda d: "realized" if d <= today else "forecast"
    )

    real_daily = (
        df_real.groupby(["region", "crop_year", "xdate", "tag"], as_index=False)
        .agg(tavg_avg=("tavg", "mean"))
        .sort_values("xdate")
    )
    real_daily = real_daily[real_daily["crop_year"] >= "15/16"].copy()

    df_normals["month"] = df_normals["date"].str[:2].astype(int)
    df_normals["day"]   = df_normals["date"].str[3:].astype(int)
    df_normals["xdate"] = df_normals.apply(
        lambda r: pd.Timestamp(2000 if r["month"] >= 9 else 2001, r["month"], r["day"]),
        axis=1,
    )
    normals_daily = (
        df_normals.groupby(["region", "xdate"], as_index=False)
        .agg(tavg_avg=("tavg", "mean"))
        .sort_values("xdate")
    )
    return real_daily, normals_daily


@st.cache_data(show_spinner=False)
def process_brazil_rolling(real_daily: pd.DataFrame, normals_daily: pd.DataFrame):
    real = (
        real_daily.groupby(["region", "crop_year", "xdate"], as_index=False)
        .agg(prcp_avg=("prcp_avg", "mean"))
    )
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


_DT_XAXIS_CAL = dict(
    tickformat="%b", dtick="M1", tick0="2001-01-01",
    range=["2001-01-01", "2001-12-31"],
    gridcolor=GRID, tickfont=dict(size=11, color=INK_3),
    linecolor=BORDER, zerolinecolor=BORDER,
)

_DT_XAXIS_BRAZIL = dict(
    tickformat="%b", dtick="M1", tick0="2000-09-01",
    range=["2000-09-01", "2001-08-31"],
    gridcolor=GRID, tickfont=dict(size=11, color=INK_3),
    linecolor=BORDER, zerolinecolor=BORDER,
)


def _add_year_traces(fig, df_year, year, y_col, hover_unit, year_colors):
    color    = year_colors.get(year, INK_4)
    df_all   = df_year.copy()
    df_all["_px"] = pd.to_datetime("2001-" + df_all["date"])
    df_all   = df_all.sort_values("_px")
    realized = df_all[df_all["tag"] == "realized"]
    forecast = df_all[df_all["tag"] == "forecast"]

    if not realized.empty:
        fig.add_trace(go.Scatter(
            x=realized["_px"], y=realized[y_col],
            mode="lines", name=year, legendgroup=year, showlegend=True,
            line=dict(color=color, width=2, dash="solid"), connectgaps=True,
            hovertemplate=f"<b>{year}</b>  %{{x|%b %d}}  %{{y:.1f}} {hover_unit}<extra></extra>",
        ))
    if not forecast.empty:
        if not realized.empty:
            forecast = pd.concat([realized.iloc[[-1]], forecast], ignore_index=True)
        fig.add_trace(go.Scatter(
            x=forecast["_px"], y=forecast[y_col],
            mode="lines", name=f"{year} fcst", legendgroup=year, showlegend=True,
            line=dict(color=color, width=2, dash="dot"), connectgaps=True,
            hovertemplate=f"<b>{year} fcst</b>  %{{x|%b %d}}  %{{y:.1f}} {hover_unit}<extra></extra>",
        ))


# -------------------------------------------------------
# CALENDAR-YEAR CHART BUILDERS
# -------------------------------------------------------
def build_cumulative_precip(daily_avg, region, year_colors):
    df_r = daily_avg[daily_avg["region"] == region].copy()
    fig  = go.Figure()
    for year in sorted(df_r["year"].unique()):
        _add_year_traces(fig, df_r[df_r["year"] == year], year, "prcp_sum_avg", "mm", year_colors)
    layout = _base_layout(f"Cumulative Precipitation  —  {region}", "mm")
    layout["xaxis"] = _DT_XAXIS_CAL.copy()
    fig.update_layout(**layout)
    return fig


def build_temperature(daily_avg_temp, region, year_colors):
    df_r      = daily_avg_temp[daily_avg_temp["region"] == region].copy()
    real_years = [y for y in df_r["year"].unique() if y != "Normal (Maxar)"]
    df_minmax = (
        df_r[df_r["year"].isin(real_years)]
        .groupby("date", as_index=False)
        .agg(tavg_min=("tavg_avg", "min"), tavg_max=("tavg_avg", "max"))
    )
    df_minmax["_px"] = pd.to_datetime("2001-" + df_minmax["date"])
    df_minmax = df_minmax.sort_values("_px")

    fig = go.Figure()
    if not df_minmax.empty:
        fig.add_trace(go.Scatter(
            x=list(df_minmax["_px"]) + list(df_minmax["_px"])[::-1],
            y=list(df_minmax["tavg_max"]) + list(df_minmax["tavg_min"])[::-1],
            fill="toself", fillcolor="rgba(0,0,0,0.05)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Hist. Range", hoverinfo="skip",
        ))
    for year in sorted(df_r["year"].unique()):
        _add_year_traces(fig, df_r[df_r["year"] == year], year, "tavg_avg", "°C", year_colors)
    layout = _base_layout(f"Average Temperature  —  {region}", "°C")
    layout["xaxis"] = _DT_XAXIS_CAL.copy()
    fig.update_layout(**layout)
    return fig


def build_rolling_precip(agg_df, region, today, year_colors):
    latest_yr = str(agg_df[agg_df["year"] != "Normal (Maxar)"]["year"].astype(str).max())
    df_r = agg_df[agg_df["region"] == region].copy()
    df_r = df_r[~((df_r["year"] == latest_yr) & (df_r["xdate"] > today))]
    fig  = go.Figure()
    for year in sorted(df_r["year"].unique()):
        df_y = df_r[df_r["year"] == year].sort_values("xdate")
        fig.add_trace(go.Scatter(
            x=df_y["xdate"], y=df_y["prcp_avg_30d_sum"],
            mode="lines", name=str(year),
            line=dict(color=year_colors.get(str(year), INK_4), width=2),
            connectgaps=True,
            hovertemplate=f"<b>{year}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>",
        ))
    layout = _base_layout(f"30-Day Rolling Precipitation  —  {region}", "Rolling Sum (mm)")
    layout["xaxis"].update(tickformat="%b", dtick="M1")
    fig.update_layout(**layout)
    return fig


# -------------------------------------------------------
# BRAZIL CHART BUILDERS  (crop year, Sep-Aug x-axis)
# -------------------------------------------------------
def build_brazil_cumulative(real_daily, normals_daily, region,
                             crop_years_sorted, crop_year_colors,
                             latest_crop_year, selected_crop_years):
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
                fig.add_trace(go.Scatter(
                    x=realized["xdate"], y=realized["cumulative_prcp"],
                    mode="lines", name=cy, legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2, dash="solid"), connectgaps=True,
                    hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>",
                ))
            if not forecast.empty:
                if not realized.empty:
                    forecast = pd.concat([realized.iloc[[-1]], forecast], ignore_index=True)
                fig.add_trace(go.Scatter(
                    x=forecast["xdate"], y=forecast["cumulative_prcp"],
                    mode="lines", name=f"{cy} fcst", legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2, dash="dot"), connectgaps=True,
                    hovertemplate=f"<b>{cy} fcst</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>",
                ))
        else:
            fig.add_trace(go.Scatter(
                x=cy_df["xdate"], y=cy_df["cumulative_prcp"],
                mode="lines", name=cy,
                line=dict(color=color, width=2, dash="solid"), connectgaps=True,
                hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>",
            ))

    if not df_n.empty:
        fig.add_trace(go.Scatter(
            x=df_n["xdate"], y=df_n["cumulative_prcp"],
            mode="lines", name="Normal (Maxar)",
            line=dict(color=INK_4, width=2.5, dash="dash"), connectgaps=True,
            hovertemplate="<b>Normal (Maxar)</b>  %{x|%b %d}  %{y:.1f} mm<extra></extra>",
        ))

    layout = _base_layout(f"Cumulative Precipitation — Crop Year  ({region})", "mm")
    layout["xaxis"] = _DT_XAXIS_BRAZIL.copy()
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_temperature(real_daily_temp, normals_daily_temp, region,
                              crop_years_sorted, crop_year_colors,
                              latest_crop_year, selected_crop_years):
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
                fig.add_trace(go.Scatter(
                    x=realized["xdate"], y=realized["tavg_avg"],
                    mode="lines", name=cy, legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2, dash="solid"), connectgaps=True,
                    hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} °C<extra></extra>",
                ))
            if not forecast.empty:
                if not realized.empty:
                    forecast = pd.concat([realized.iloc[[-1]], forecast], ignore_index=True)
                fig.add_trace(go.Scatter(
                    x=forecast["xdate"], y=forecast["tavg_avg"],
                    mode="lines", name=f"{cy} fcst", legendgroup=cy, showlegend=True,
                    line=dict(color=color, width=2, dash="dot"), connectgaps=True,
                    hovertemplate=f"<b>{cy} fcst</b>  %{{x|%b %d}}  %{{y:.1f}} °C<extra></extra>",
                ))
        else:
            if not cy_df.empty:
                fig.add_trace(go.Scatter(
                    x=cy_df["xdate"], y=cy_df["tavg_avg"],
                    mode="lines", name=cy,
                    line=dict(color=color, width=2, dash="solid"), connectgaps=True,
                    hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} °C<extra></extra>",
                ))

    if not df_n.empty:
        fig.add_trace(go.Scatter(
            x=df_n["xdate"], y=df_n["tavg_avg"],
            mode="lines", name="Normal (Maxar)",
            line=dict(color=INK_4, width=2.5, dash="dash"), connectgaps=True,
            hovertemplate="<b>Normal (Maxar)</b>  %{x|%b %d}  %{y:.1f} °C<extra></extra>",
        ))

    layout = _base_layout(f"Average Temperature — Crop Year  ({region})", "°C")
    layout["xaxis"] = _DT_XAXIS_BRAZIL.copy()
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_rolling(real_rolled, normals_rolled, region,
                          crop_years_sorted, crop_year_colors, selected_crop_years):
    df_r = real_rolled[real_rolled["region"] == region].copy()
    df_n = normals_rolled[normals_rolled["region"] == region].sort_values("xdate")
    fig  = go.Figure()

    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        color = crop_year_colors.get(cy, INK_4)
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        if not cy_df.empty:
            fig.add_trace(go.Scatter(
                x=cy_df["xdate"], y=cy_df["prcp_30d"],
                mode="lines", name=cy,
                line=dict(color=color, width=2), connectgaps=True,
                hovertemplate=f"<b>{cy}</b>  %{{x|%b %d}}  %{{y:.1f}} mm<extra></extra>",
            ))

    if not df_n.empty:
        fig.add_trace(go.Scatter(
            x=df_n["xdate"], y=df_n["prcp_30d"],
            mode="lines", name="Normal (Maxar)",
            line=dict(color=INK_4, width=2.5, dash="dash"), connectgaps=True,
            hovertemplate="<b>Normal (Maxar)</b>  %{x|%b %d}  %{y:.1f} mm<extra></extra>",
        ))

    layout = _base_layout(f"30-Day Rolling Precipitation — Crop Year  ({region})", "Rolling Sum (mm)")
    layout["xaxis"] = _DT_XAXIS_BRAZIL.copy()
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


# -------------------------------------------------------
# ADVANCED ANALYTICS
# -------------------------------------------------------
_BRAZIL_MONTH_ORDER  = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
_MONTH_LABELS        = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
_BRAZIL_MONTH_LABELS = [_MONTH_LABELS[m] for m in _BRAZIL_MONTH_ORDER]
_CAL_MONTH_LABELS    = [_MONTH_LABELS[m] for m in range(1, 13)]


def build_brazil_precip_anomaly(real_daily, normals_daily, region,
                                 crop_years_sorted, crop_year_colors, selected_crop_years):
    df_r = real_daily[real_daily["region"] == region].copy()
    df_n = normals_daily[normals_daily["region"] == region].copy()
    df_r["month"] = df_r["xdate"].dt.month
    df_n["month"] = df_n["xdate"].dt.month
    normals_m = df_n.groupby("month")["prcp_avg"].sum().reset_index().rename(columns={"prcp_avg": "normal_sum"})
    real_m    = df_r.groupby(["crop_year", "month"])["prcp_avg"].sum().reset_index().rename(columns={"prcp_avg": "real_sum"})
    merged = real_m.merge(normals_m, on="month")
    merged["anomaly"]     = merged["real_sum"] - merged["normal_sum"]
    merged["month_label"] = merged["month"].map(_MONTH_LABELS)
    merged["month_order"] = merged["month"].map({m: i for i, m in enumerate(_BRAZIL_MONTH_ORDER)})
    merged = merged.sort_values("month_order")
    fig = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = merged[merged["crop_year"] == cy]
        if cy_df.empty:
            continue
        fig.add_trace(go.Bar(
            x=cy_df["month_label"], y=cy_df["anomaly"], name=cy,
            marker_color=crop_year_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y:+.1f}} mm<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    layout = _base_layout(f"Monthly Precipitation Anomaly vs Normal (Maxar) ({region})", "mm above/below normal")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = _BRAZIL_MONTH_LABELS
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_dry_days(real_daily, region, crop_years_sorted, crop_year_colors,
                           selected_crop_years, threshold=1.0):
    df_r = real_daily[real_daily["region"] == region].copy()
    df_r["month"] = df_r["xdate"].dt.month
    results = []
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = df_r[df_r["crop_year"] == cy].sort_values("xdate")
        for month in _BRAZIL_MONTH_ORDER:
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
    res["month_order"] = res["month"].map({m: i for i, m in enumerate(_BRAZIL_MONTH_ORDER)})
    res = res.sort_values("month_order")
    fig = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty:
            continue
        fig.add_trace(go.Bar(
            x=cy_df["month_label"], y=cy_df["max_dry_days"], name=cy,
            marker_color=crop_year_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>",
        ))
    layout = _base_layout(f"Max Consecutive Dry Days per Month ({region}, <{threshold}mm/day)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = _BRAZIL_MONTH_LABELS
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_heat_stress(real_daily_temp, region, crop_years_sorted, crop_year_colors,
                              selected_crop_years, threshold=28.0):
    if real_daily_temp.empty:
        return go.Figure()
    df_r = real_daily_temp[real_daily_temp["region"] == region].copy()
    df_r["month"] = df_r["xdate"].dt.month
    results = []
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = df_r[df_r["crop_year"] == cy]
        for month in _BRAZIL_MONTH_ORDER:
            m_df = cy_df[cy_df["month"] == month]
            if m_df.empty:
                continue
            results.append({"crop_year": cy, "month": month,
                             "stress_days": int((m_df["tavg_avg"] > threshold).sum())})
    if not results:
        return go.Figure()
    res = pd.DataFrame(results)
    res["month_label"] = res["month"].map(_MONTH_LABELS)
    res["month_order"] = res["month"].map({m: i for i, m in enumerate(_BRAZIL_MONTH_ORDER)})
    res = res.sort_values("month_order")
    fig = go.Figure()
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df = res[res["crop_year"] == cy]
        if cy_df.empty:
            continue
        fig.add_trace(go.Bar(
            x=cy_df["month_label"], y=cy_df["stress_days"], name=cy,
            marker_color=crop_year_colors.get(cy, INK_4), opacity=0.85,
            hovertemplate=f"<b>{cy}</b>  %{{x}}  %{{y}} days<extra></extra>",
        ))
    layout = _base_layout(f"Heat Stress Days per Month ({region}, >{threshold}°C avg)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = _BRAZIL_MONTH_LABELS
    layout["legend"]["title"]["text"] = "Crop Year"
    fig.update_layout(**layout)
    return fig


def build_brazil_monthly_boxplot(real_daily, normals_daily, region,
                                  crop_years_sorted, crop_year_colors, latest_crop_year):
    df_r = real_daily[real_daily["region"] == region].copy()
    df_n = normals_daily[normals_daily["region"] == region].copy()
    df_r["month"] = df_r["xdate"].dt.month
    df_n["month"] = df_n["xdate"].dt.month
    monthly   = df_r.groupby(["crop_year", "month"])["prcp_avg"].sum().reset_index()
    normals_m = df_n.groupby("month")["prcp_avg"].sum().reset_index()
    monthly["month_label"]   = monthly["month"].map(_MONTH_LABELS)
    normals_m["month_label"] = normals_m["month"].map(_MONTH_LABELS)
    monthly["month_order"]   = monthly["month"].map({m: i for i, m in enumerate(_BRAZIL_MONTH_ORDER)})
    normals_m["month_order"] = normals_m["month"].map({m: i for i, m in enumerate(_BRAZIL_MONTH_ORDER)})
    monthly   = monthly.sort_values("month_order")
    normals_m = normals_m.sort_values("month_order")
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=monthly["month_label"], y=monthly["prcp_avg"],
        name="Historical range", marker_color=INK_4,
        fillcolor="rgba(174,174,178,0.25)", line_color=INK_4,
        boxpoints="outliers", whiskerwidth=0.5,
    ))
    if latest_crop_year:
        curr = monthly[monthly["crop_year"] == latest_crop_year]
        fig.add_trace(go.Scatter(
            x=curr["month_label"], y=curr["prcp_avg"],
            mode="markers+lines", name=latest_crop_year,
            marker=dict(color=RED, size=8, symbol="circle"),
            line=dict(color=RED, width=2),
            hovertemplate=f"<b>{latest_crop_year}</b>  %{{x}}  %{{y:.1f}} mm<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=normals_m["month_label"], y=normals_m["prcp_avg"],
        mode="lines", name="Normal (Maxar)",
        line=dict(color=INK_4, width=2, dash="dash"),
        hovertemplate="<b>Normal (Maxar)</b>  %{x}  %{y:.1f} mm<extra></extra>",
    ))
    layout = _base_layout(f"Monthly Precipitation Distribution ({region})", "mm")
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = _BRAZIL_MONTH_LABELS
    layout["legend"]["title"]["text"] = "Series"
    fig.update_layout(**layout)
    return fig


def build_brazil_ytd_vs_normals(real_daily, normals_daily, region,
                                  crop_years_sorted, crop_year_colors,
                                  selected_crop_years, today):
    df_r = real_daily[real_daily["region"] == region].copy()
    df_n = normals_daily[normals_daily["region"] == region].copy()
    today_xdate = pd.Timestamp(2000 if today.month >= 9 else 2001, today.month, today.day)
    ytd_normal  = df_n[df_n["xdate"] <= today_xdate]["prcp_avg"].sum()
    if ytd_normal == 0:
        return go.Figure()
    results = []
    for cy in crop_years_sorted:
        if cy not in selected_crop_years:
            continue
        cy_df  = df_r[(df_r["crop_year"] == cy) & (df_r["xdate"] <= today_xdate)]
        actual = cy_df["prcp_avg"].sum()
        results.append({"crop_year": cy, "pct": actual / ytd_normal * 100, "actual": actual})
    if not results:
        return go.Figure()
    res = pd.DataFrame(results)
    colors = [crop_year_colors.get(r, INK_4) for r in res["crop_year"]]
    fig = go.Figure(go.Bar(
        x=res["pct"], y=res["crop_year"], orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}%  ({a:.0f} mm)" for p, a in zip(res["pct"], res["actual"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b>  %{x:.1f}% of normal<extra></extra>",
    ))
    fig.add_vline(x=100, line_color=INK_3, line_width=1.5, line_dash="dash")
    layout = _base_layout(
        f"YTD Precip vs Normal (Maxar) — to {today.strftime('%b %d')} ({region})", "% of Normal")
    layout["xaxis"]["ticksuffix"] = "%"
    layout["xaxis"]["range"] = [0, max(res["pct"].max() * 1.25, 150)]
    layout["showlegend"] = False
    layout["height"] = max(260, 70 * len(results))
    fig.update_layout(**layout)
    return fig


# -------------------------------------------------------
# CALENDAR-YEAR ADVANCED ANALYTICS
# -------------------------------------------------------
def build_precip_anomaly(daily_avg, region, selected_years):
    df = daily_avg[daily_avg["region"] == region].copy()
    df["month"] = df["full_date"].dt.month
    normals_m = (df[df["year"] == "Normal (Maxar)"].groupby("month")["prcp_avg"].sum()
                 .reset_index().rename(columns={"prcp_avg": "normal_sum"}))
    real_m    = (df[df["year"] != "Normal (Maxar)"].groupby(["year", "month"])["prcp_avg"].sum()
                 .reset_index().rename(columns={"prcp_avg": "real_sum"}))
    merged = real_m.merge(normals_m, on="month")
    merged["anomaly"]     = merged["real_sum"] - merged["normal_sum"]
    merged["month_label"] = merged["month"].map(_MONTH_LABELS)
    merged = merged.sort_values("month")
    fig = go.Figure()
    for year in selected_years:
        if year == "Normal (Maxar)":
            continue
        y_df = merged[merged["year"] == year]
        if y_df.empty:
            continue
        fig.add_trace(go.Bar(
            x=y_df["month_label"], y=y_df["anomaly"], name=year,
            marker_color=ALL_YEAR_COLORS.get(year, INK_4), opacity=0.85,
            hovertemplate=f"<b>{year}</b>  %{{x}}  %{{y:+.1f}} mm<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    layout = _base_layout(f"Monthly Precipitation Anomaly vs Normal (Maxar) ({region})", "mm above/below normal")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = _CAL_MONTH_LABELS
    layout["legend"]["title"]["text"] = "Year"
    fig.update_layout(**layout)
    return fig


def build_dry_days(daily_avg, region, selected_years, threshold=1.0):
    df = daily_avg[(daily_avg["region"] == region) & (daily_avg["year"] != "Normal (Maxar)")].copy()
    df["month"] = df["full_date"].dt.month
    results = []
    for year in selected_years:
        if year == "Normal (Maxar)":
            continue
        y_df = df[df["year"] == year].sort_values("full_date")
        for month in range(1, 13):
            m_df = y_df[y_df["month"] == month]
            if m_df.empty:
                continue
            max_run = curr = 0
            for v in (m_df["prcp_avg"] < threshold):
                curr = curr + 1 if v else 0
                max_run = max(max_run, curr)
            results.append({"year": year, "month": month, "max_dry_days": max_run})
    if not results:
        return go.Figure()
    res = pd.DataFrame(results)
    res["month_label"] = res["month"].map(_MONTH_LABELS)
    res = res.sort_values("month")
    fig = go.Figure()
    for year in selected_years:
        if year == "Normal (Maxar)":
            continue
        y_df = res[res["year"] == year]
        if y_df.empty:
            continue
        fig.add_trace(go.Bar(
            x=y_df["month_label"], y=y_df["max_dry_days"], name=year,
            marker_color=ALL_YEAR_COLORS.get(year, INK_4), opacity=0.85,
            hovertemplate=f"<b>{year}</b>  %{{x}}  %{{y}} days<extra></extra>",
        ))
    layout = _base_layout(f"Max Consecutive Dry Days per Month ({region}, <{threshold}mm/day)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = _CAL_MONTH_LABELS
    layout["legend"]["title"]["text"] = "Year"
    fig.update_layout(**layout)
    return fig


def build_heat_stress(daily_avg_temp, region, selected_years, threshold=30.0):
    df = daily_avg_temp[daily_avg_temp["region"] == region].copy()
    df = df[df["year"] != "Normal (Maxar)"].copy()
    df["full_date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["date"], errors="coerce")
    df = df[df["full_date"].notna()].copy()
    df["month"] = df["full_date"].dt.month
    results = []
    for year in selected_years:
        if year == "Normal (Maxar)":
            continue
        y_df = df[df["year"] == year]
        for month in range(1, 13):
            m_df = y_df[y_df["month"] == month]
            if m_df.empty:
                continue
            results.append({"year": year, "month": month,
                             "stress_days": int((m_df["tavg_avg"] > threshold).sum())})
    if not results:
        return go.Figure()
    res = pd.DataFrame(results)
    res["month_label"] = res["month"].map(_MONTH_LABELS)
    res = res.sort_values("month")
    fig = go.Figure()
    for year in selected_years:
        if year == "Normal (Maxar)":
            continue
        y_df = res[res["year"] == year]
        if y_df.empty:
            continue
        fig.add_trace(go.Bar(
            x=y_df["month_label"], y=y_df["stress_days"], name=year,
            marker_color=ALL_YEAR_COLORS.get(year, INK_4), opacity=0.85,
            hovertemplate=f"<b>{year}</b>  %{{x}}  %{{y}} days<extra></extra>",
        ))
    layout = _base_layout(f"Heat Stress Days per Month ({region}, >{threshold}°C avg)", "Days")
    layout["barmode"] = "group"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = _CAL_MONTH_LABELS
    layout["legend"]["title"]["text"] = "Year"
    fig.update_layout(**layout)
    return fig


def build_monthly_boxplot(daily_avg, region, selected_years):
    df = daily_avg[daily_avg["region"] == region].copy()
    df["month"] = df["full_date"].dt.month
    monthly   = df[df["year"] != "Normal (Maxar)"].groupby(["year", "month"])["prcp_avg"].sum().reset_index()
    normals_m = df[df["year"] == "Normal (Maxar)"].groupby("month")["prcp_avg"].sum().reset_index()
    monthly["month_label"]   = monthly["month"].map(_MONTH_LABELS)
    normals_m["month_label"] = normals_m["month"].map(_MONTH_LABELS)
    monthly   = monthly.sort_values("month")
    normals_m = normals_m.sort_values("month")
    latest_year = str(df[df["year"] != "Normal (Maxar)"]["year"].astype(str).max())
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=monthly["month_label"], y=monthly["prcp_avg"],
        name="Historical range", marker_color=INK_4,
        fillcolor="rgba(174,174,178,0.25)", line_color=INK_4,
        boxpoints="outliers", whiskerwidth=0.5,
    ))
    curr = monthly[monthly["year"] == latest_year]
    if not curr.empty:
        fig.add_trace(go.Scatter(
            x=curr["month_label"], y=curr["prcp_avg"],
            mode="markers+lines", name=latest_year,
            marker=dict(color=RED, size=8), line=dict(color=RED, width=2),
            hovertemplate=f"<b>{latest_year}</b>  %{{x}}  %{{y:.1f}} mm<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=normals_m["month_label"], y=normals_m["prcp_avg"],
        mode="lines", name="Normal (Maxar)",
        line=dict(color=INK_4, width=2, dash="dash"),
        hovertemplate="<b>Normal (Maxar)</b>  %{x}  %{y:.1f} mm<extra></extra>",
    ))
    layout = _base_layout(f"Monthly Precipitation Distribution ({region})", "mm")
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = _CAL_MONTH_LABELS
    layout["legend"]["title"]["text"] = "Series"
    fig.update_layout(**layout)
    return fig


def build_ytd_vs_normals(daily_avg, region, selected_years, today):
    df = daily_avg[daily_avg["region"] == region].copy()
    ytd_normal = df[(df["year"] == "Normal (Maxar)") & (df["full_date"] <= today)]["prcp_avg"].sum()
    if ytd_normal == 0:
        return go.Figure()
    results = []
    for year in selected_years:
        if year == "Normal (Maxar)":
            continue
        actual = df[(df["year"] == year) & (df["full_date"] <= today)]["prcp_avg"].sum()
        results.append({"year": year, "pct": actual / ytd_normal * 100, "actual": actual})
    if not results:
        return go.Figure()
    res = pd.DataFrame(results)
    colors = [ALL_YEAR_COLORS.get(y, INK_4) for y in res["year"]]
    fig = go.Figure(go.Bar(
        x=res["pct"], y=res["year"], orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}%  ({a:.0f} mm)" for p, a in zip(res["pct"], res["actual"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b>  %{x:.1f}% of normal<extra></extra>",
    ))
    fig.add_vline(x=100, line_color=INK_3, line_width=1.5, line_dash="dash")
    layout = _base_layout(
        f"YTD Precip vs Normal (Maxar) — to {today.strftime('%b %d')} ({region})", "% of Normal")
    layout["xaxis"]["ticksuffix"] = "%"
    layout["xaxis"]["range"] = [0, max(res["pct"].max() * 1.25, 150)]
    layout["showlegend"] = False
    layout["height"] = max(260, 70 * len(results))
    fig.update_layout(**layout)
    return fig


# -------------------------------------------------------
# HELPER: render a full calendar-year origin tab
# -------------------------------------------------------
def render_calendar_tab(origin_name, selected_years, today):
    c1, c2 = st.columns(2)
    with c1:
        with st.spinner(f"Loading {origin_name} precipitation..."):
            raw_prcp = load_origin_data(origin_name, "PRCP")
    with c2:
        with st.spinner(f"Loading {origin_name} temperature..."):
            raw_temp = load_origin_data(origin_name, "TAVG")

    if raw_prcp.empty or raw_temp.empty:
        st.error(f"No data for {origin_name}. Run backfill.py first.")
        return

    daily_avg      = process_precipitation(raw_prcp, today)
    daily_avg_temp = process_temperature(raw_temp, today)
    agg_df         = process_rolling(daily_avg, today)

    daily_avg      = daily_avg[daily_avg["year"].isin(selected_years)]
    daily_avg_temp = daily_avg_temp[daily_avg_temp["year"].isin(selected_years)]
    agg_df         = agg_df[agg_df["year"].isin(selected_years)]

    active_colors = {y: c for y, c in ALL_YEAR_COLORS.items() if y in selected_years}
    regions       = sorted(daily_avg["region"].unique())

    for region in regions:
        st.markdown(
            f"<h2 class='section-header'>Cumulative Precipitation &nbsp;—&nbsp; {region}</h2>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_cumulative_precip(daily_avg, region, active_colors),
            use_container_width=True, key=f"cum_{origin_name}_{region}",
        )
        st.markdown(
            f"<h2 class='section-header'>Temperature &nbsp;—&nbsp; {region}</h2>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_temperature(daily_avg_temp, region, active_colors),
            use_container_width=True, key=f"tmp_{origin_name}_{region}",
        )
        st.markdown(
            f"<h2 class='section-header'>30-Day Rolling Precipitation &nbsp;—&nbsp; {region}</h2>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_rolling_precip(agg_df, region, today, active_colors),
            use_container_width=True, key=f"rol_{origin_name}_{region}",
        )

        with st.expander("Advanced Analytics", expanded=False):
            st.plotly_chart(
                build_ytd_vs_normals(daily_avg, region, selected_years, today),
                use_container_width=True, key=f"ytd_{origin_name}_{region}",
            )
            c_a, c_b = st.columns(2)
            with c_a:
                st.plotly_chart(
                    build_precip_anomaly(daily_avg, region, selected_years),
                    use_container_width=True, key=f"anom_{origin_name}_{region}",
                )
                st.plotly_chart(
                    build_dry_days(daily_avg, region, selected_years),
                    use_container_width=True, key=f"dry_{origin_name}_{region}",
                )
            with c_b:
                st.plotly_chart(
                    build_monthly_boxplot(daily_avg, region, selected_years),
                    use_container_width=True, key=f"box_{origin_name}_{region}",
                )
                st.plotly_chart(
                    build_heat_stress(daily_avg_temp, region, selected_years),
                    use_container_width=True, key=f"heat_{origin_name}_{region}",
                )


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
        f"text-transform:uppercase;color:{INK_3};margin-bottom:.6rem'>Filters</p>",
        unsafe_allow_html=True,
    )
    selected_years = st.multiselect(
        "Years (calendar origins)",
        options=CALENDAR_YEARS,
        default=CALENDAR_YEARS,
        help="Applies to Colombia, Honduras, Super 4, Vietnam",
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
        f"text-transform:uppercase;color:{INK_3};margin-bottom:.6rem'>Year colors</p>",
        unsafe_allow_html=True,
    )
    for yr, col in ALL_YEAR_COLORS.items():
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            f"<span style='width:12px;height:12px;border-radius:2px;background:{col};"
            f"display:inline-block'></span>"
            f"<span style='font-size:.82rem;color:{INK_2}'>{yr}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown(f"<p style='font-size:.75rem;color:{INK_4};margin-top:.8rem'>"
                f"Today: {today.strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)

if not selected_years:
    st.warning("Select at least one year from the sidebar.")
    st.stop()

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
                process_brazil(raw_brazil_prcp, today)

            real_daily_temp, normals_daily_temp = \
                process_brazil_temp(raw_brazil_temp, today) if not raw_brazil_temp.empty \
                else (pd.DataFrame(), pd.DataFrame())

            real_rolled, normals_rolled = process_brazil_rolling(real_daily, normals_daily)

            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                regions_all = sorted(real_daily["region"].unique())
                selected_regions_brazil = st.multiselect(
                    "Sub-Regions",
                    options=regions_all,
                    default=regions_all,
                    key="brazil_region_filter",
                )
            with filter_col2:
                selected_crop_years = st.multiselect(
                    "Crop Years",
                    options=crop_years_sorted,
                    default=crop_years_sorted,
                    key="brazil_cy_filter",
                )

            if not selected_regions_brazil:
                st.warning("Select at least one sub-region.")
            else:
                for region in selected_regions_brazil:
                    st.markdown(
                        f"<h2 class='section-header'>Cumulative Precipitation &nbsp;—&nbsp; {region}</h2>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        build_brazil_cumulative(
                            real_daily, normals_daily, region,
                            crop_years_sorted, crop_year_colors, latest_cy,
                            selected_crop_years,
                        ),
                        use_container_width=True, key=f"bra_cum_{region}",
                    )

                    if not real_daily_temp.empty:
                        st.markdown(
                            f"<h2 class='section-header'>Average Temperature &nbsp;—&nbsp; {region}</h2>",
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            build_brazil_temperature(
                                real_daily_temp, normals_daily_temp, region,
                                crop_years_sorted, crop_year_colors, latest_cy,
                                selected_crop_years,
                            ),
                            use_container_width=True, key=f"bra_tmp_{region}",
                        )

                    st.markdown(
                        f"<h2 class='section-header'>30-Day Rolling Precipitation &nbsp;—&nbsp; {region}</h2>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        build_brazil_rolling(
                            real_rolled, normals_rolled, region,
                            crop_years_sorted, crop_year_colors,
                            selected_crop_years,
                        ),
                        use_container_width=True, key=f"bra_rol_{region}",
                    )

                    with st.expander("Advanced Analytics", expanded=False):
                        st.plotly_chart(
                            build_brazil_ytd_vs_normals(
                                real_daily, normals_daily, region,
                                crop_years_sorted, crop_year_colors,
                                selected_crop_years, today,
                            ),
                            use_container_width=True, key=f"bra_ytd_{region}",
                        )
                        c_a, c_b = st.columns(2)
                        with c_a:
                            st.plotly_chart(
                                build_brazil_precip_anomaly(
                                    real_daily, normals_daily, region,
                                    crop_years_sorted, crop_year_colors, selected_crop_years,
                                ),
                                use_container_width=True, key=f"bra_anom_{region}",
                            )
                            st.plotly_chart(
                                build_brazil_dry_days(
                                    real_daily, region, crop_years_sorted,
                                    crop_year_colors, selected_crop_years,
                                ),
                                use_container_width=True, key=f"bra_dry_{region}",
                            )
                        with c_b:
                            st.plotly_chart(
                                build_brazil_monthly_boxplot(
                                    real_daily, normals_daily, region,
                                    crop_years_sorted, crop_year_colors, latest_cy,
                                ),
                                use_container_width=True, key=f"bra_box_{region}",
                            )
                            st.plotly_chart(
                                build_brazil_heat_stress(
                                    real_daily_temp, region, crop_years_sorted,
                                    crop_year_colors, selected_crop_years,
                                ),
                                use_container_width=True, key=f"bra_heat_{region}",
                            )

# ---- CALENDAR-YEAR ORIGINS ----
with tab_colombia:
    render_calendar_tab("Colombia", selected_years, today)

with tab_honduras:
    render_calendar_tab("Honduras", selected_years, today)

with tab_super4:
    render_calendar_tab("Super 4", selected_years, today)

with tab_vietnam:
    render_calendar_tab("Vietnam", selected_years, today)
