
# Streamlit Dashboard — Switzerland CO₂ Case Study (1980–2014)
# Save this file as: dashboard.py

import io, json, urllib.request, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Switzerland CO₂ (1980–2014)", layout="wide")

START, END = 1980, 2014
ISO = "CHE"
COUNTRY = "Switzerland"
OWID_CO2_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_owid_co2():
    with urllib.request.urlopen(OWID_CO2_URL) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    cols = ["country","iso_code","year","co2","co2_per_capita","population"]
    df = df[cols].dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df

@st.cache_data(show_spinner=False)
def wb_series(country_code: str, indicator: str):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page=20000"
    with urllib.request.urlopen(url) as r:
        js = json.load(r)
    rows = js[1] if isinstance(js, list) and len(js) > 1 else []
    df = pd.DataFrame([{"year": int(x["date"]), "value": x["value"]} for x in rows if x["value"] is not None])
    return df.sort_values("year").reset_index(drop=True)

def annotate_line_end(ax, x, y, label, dx=0.6, fontsize=9):
    s = pd.Series(y.values, index=x.values)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return
    last_year = int(s.index.max())
    ax.text(last_year + dx, float(s.loc[last_year]), label, va="center", fontsize=fontsize)

# ---------- Sidebar ----------
st.sidebar.title("Switzerland CO₂ Case Study")
st.sidebar.markdown("**Window:** 1980–2014 (fixed by assignment).")
emdat_file = st.sidebar.file_uploader("Upload EM-DAT Excel (Hydrological events for Switzerland)", type=["xlsx"])
st.sidebar.markdown("---")
st.sidebar.markdown("Data: OWID CO₂, World Bank (energy & GDP). EM-DAT upload is optional.")

# ---------- Load data ----------
co2_all = load_owid_co2()
co2_8014 = co2_all[(co2_all["year"].between(START, END)) & (co2_all["iso_code"].notna())].copy()
CHE = co2_8014[co2_8014["iso_code"] == ISO].copy()
WORLD = co2_8014[co2_8014["iso_code"] == "OWID_WRL"].copy()
is_owid = co2_8014["iso_code"].astype("string").str.startswith("OWID", na=False)

energy_che = wb_series(ISO, "EG.USE.PCAP.KG.OE")
gdp_che   = wb_series(ISO, "NY.GDP.PCAP.KD.ZG")
energy_wl = wb_series("WLD","EG.USE.PCAP.KG.OE")
energy_che = energy_che[energy_che["year"].between(START, END)]
gdp_che   = gdp_che[gdp_che["year"].between(START, END)]
energy_wl = energy_wl[energy_wl["year"].between(START, END)]

swiss = (
    CHE[["year","co2","co2_per_capita"]]
    .merge(energy_che, on="year", how="left").rename(columns={"value":"energy_pc"})
    .merge(gdp_che,   on="year", how="left").rename(columns={"value":"gdp_pc_growth"})
    .sort_values("year")
)

# ---------- EM-DAT (optional upload) ----------
hyd_yearly = None
if emdat_file is not None:
    emdat = pd.read_excel(emdat_file)
    need = {"Disaster Subgroup","Start Year","DisNo.","Total Deaths","Total Affected"}
    missing = need - set(emdat.columns)
    if missing:
        st.sidebar.error(f"EM-DAT file is missing columns: {missing}")
    else:
        hyd = emdat[emdat["Disaster Subgroup"].astype(str).str.strip().eq("Hydrological")].copy()
        hyd["year"] = pd.to_numeric(hyd["Start Year"], errors="coerce")
        hyd = hyd[hyd["year"].between(START, END)]
        hyd_yearly = (
            hyd.groupby("year", as_index=False)
               .agg(events=("DisNo.","count"),
                    total_killed=("Total Deaths","sum"),
                    total_affected=("Total Affected","sum"))
               .fillna(0)
        )

# ---------- Title ----------
st.title("🇨🇭 Switzerland CO₂ Case Study (1980–2014)")
st.caption("Sources: OWID CO₂ dataset, World Bank indicators, optional EM-DAT upload for hydrological disasters.")

# ---------- 1) “Adding color” line plot ----------
st.subheader("CO₂ per capita over time — highlighting Switzerland")
fig, ax = plt.subplots(figsize=(8,4))
others = co2_8014[(~is_owid) & (co2_8014["iso_code"] != ISO)]
for c, g in others.groupby("country"):
    g = g.dropna(subset=["co2_per_capita"]).sort_values("year")
    if len(g) > 3:
        ax.plot(g["year"], g["co2_per_capita"], linewidth=0.8, alpha=0.6)
ax.plot(CHE["year"], CHE["co2_per_capita"], label=COUNTRY, linewidth=2.5)
ax.plot(WORLD["year"], WORLD["co2_per_capita"], label="World", linewidth=2.0)
ax.set_xlabel("Year"); ax.set_ylabel("tCO₂ per person")
ax.legend()
st.pyplot(fig)
st.caption("Switzerland’s per-capita CO₂ is flat/slightly down; the world mean rises toward Swiss levels by the 2010s.")

# ---------- 2) Top-10 emitters line plot with end labels ----------
st.subheader("Top-10 emitting countries (TOTAL CO₂, 1980–2014)")
panel_win = co2_8014[~is_owid].copy()
top10 = (panel_win.groupby("country", as_index=False)["co2"]
                 .sum().sort_values("co2", ascending=False).head(10))["country"].tolist()
top10_hist = panel_win[panel_win["country"].isin(top10)].copy()
fig, ax = plt.subplots(figsize=(9,5))
for c, g in top10_hist.groupby("country"):
    g = g.dropna(subset=["co2"]).sort_values("year")
    if len(g) > 1:
        ax.plot(g["year"], g["co2"], linewidth=1.8)
        annotate_line_end(ax, g["year"], g["co2"], c, dx=0.6, fontsize=9)
ax.set_xlabel("Year"); ax.set_ylabel("CO₂ (Mt)")
ax.set_xlim(START, END + 2)
st.pyplot(fig)
st.caption("Top global emitters dominate totals; Switzerland isn’t in this top-10 by absolute emissions.")

# ---------- 3) Tile plot (heatmap) for same top-10 (per-capita, row-normalized) ----------
st.subheader("Top-10 emitters — normalized CO₂ per capita (1980–2014)")
pivot = (panel_win[panel_win["country"].isin(top10)]
         .pivot_table(index="country", columns="year", values="co2_per_capita"))
Z = pivot.copy()
row_min = Z.min(axis=1)
row_rng = (Z.max(axis=1) - Z.min(axis=1)).replace(0, np.nan)
Z = (Z.sub(row_min, axis=0)).div(row_rng, axis=0)
fig, ax = plt.subplots(figsize=(9,4.5))
im = ax.imshow(Z.values, aspect="auto", interpolation="nearest")
ax.set_yticks(range(len(Z.index))); ax.set_yticklabels(Z.index)
xt = np.linspace(0, Z.shape[1]-1, 8, dtype=int)
ax.set_xticks(xt); ax.set_xticklabels(pivot.columns[xt])
fig.colorbar(im, ax=ax, label="row-normalized")
st.pyplot(fig)
st.caption("Row normalization shows within-country change patterns, not absolute levels.")

# ---------- 4) 2×2 facet: World energy + Switzerland metrics ----------
st.subheader("World vs Switzerland — focused metrics (1980–2014)")
def _align(df, col, newname, years):
    s = df[["year", col]].dropna(subset=["year"]).copy()
    s["year"] = s["year"].astype(int)
    s = s.rename(columns={col: newname})
    return years.merge(s, on="year", how="left")
years = pd.DataFrame({"year": np.arange(START, END + 1)})
facet = years.copy()
facet = facet.merge(_align(energy_wl.rename(columns={"value":"energy_pc"}), "energy_pc", "world_energy_pc", years), on="year", how="left")
facet = facet.merge(_align(CHE, "co2", "che_co2_total", years), on="year", how="left")
facet = facet.merge(_align(CHE, "co2_per_capita", "che_co2_pc", years), on="year", how="left")
facet = facet.merge(_align(swiss.rename(columns={"energy_pc":"che_energy_pc"}), "che_energy_pc", "che_energy_pc", years), on="year", how="left")
def _plot_or_note(ax, x, y, title):
    if np.isfinite(y).sum() > 0:
        ax.plot(x, y)
    else:
        ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(START, END)
fig, axes = plt.subplots(2,2, figsize=(10,6), sharex=True)
(ax1, ax2), (ax3, ax4) = axes
_plot_or_note(ax1, facet["year"], facet["world_energy_pc"], "World: Energy per capita")
_plot_or_note(ax2, facet["year"], facet["che_co2_total"],   "Switzerland: Total CO₂")
_plot_or_note(ax3, facet["year"], facet["che_co2_pc"],      "Switzerland: CO₂ per capita")
_plot_or_note(ax4, facet["year"], facet["che_energy_pc"],   "Switzerland: Energy per capita")
for ax in axes.flat:
    ax.set_xlabel("Year")
st.pyplot(fig)
st.caption("Global energy per person grows; Switzerland keeps CO₂ flat with stable energy use (clean power & efficiency).")

# ---------- 5) Scatter plots with trend lines ----------
st.subheader("Scatter plots (Switzerland)")
# (a) CO2 vs Energy
df = swiss.dropna(subset=["co2_per_capita","energy_pc"]).copy()
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(df["energy_pc"], df["co2_per_capita"])
if len(df) > 2:
    slope, intercept, r, p, se = stats.linregress(df["energy_pc"], df["co2_per_capita"])
    x = np.linspace(df["energy_pc"].min(), df["energy_pc"].max(), 100)
    ax.plot(x, intercept + slope*x, linewidth=2)
    ax.set_title(f"CO₂/person vs Energy/person (r={r:.2f})")
else:
    ax.set_title("CO₂/person vs Energy/person")
ax.set_xlabel("Energy per cap (kg oe)"); ax.set_ylabel("CO₂ per cap (t)")
st.pyplot(fig)
# (b) CO2 vs GDP growth
df2 = swiss.dropna(subset=["co2_per_capita","gdp_pc_growth"]).copy()
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(df2["gdp_pc_growth"], df2["co2_per_capita"])
if len(df2) > 2:
    slope2, intercept2, r2, p2, se2 = stats.linregress(df2["gdp_pc_growth"], df2["co2_per_capita"])
    x2 = np.linspace(df2["gdp_pc_growth"].min(), df2["gdp_pc_growth"].max(), 100)
    ax.plot(x2, intercept2 + slope2*x2, linewidth=2)
    ax.set_title(f"CO₂/person vs GDP per-cap growth (r={r2:.2f})")
else:
    ax.set_title("CO₂/person vs GDP per-cap growth")
ax.set_xlabel("GDP per-cap growth (%)"); ax.set_ylabel("CO₂ per cap (t)")
st.pyplot(fig)
st.caption("Energy ↗ ↔ CO₂ ↗ (positive association); GDP growth vs CO₂ is weak, indicating partial decoupling.")

# ---------- Optional: show EM-DAT summary ----------
if hyd_yearly is not None:
    st.subheader("Hydrological disasters — yearly counts (EM-DAT upload)")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(hyd_yearly["year"], hyd_yearly["events"])
    ax.set_xlabel("Year"); ax.set_ylabel("Events")
    st.pyplot(fig)
    st.caption("Hydrological disaster counts vary year to year; used descriptively (not causal).")

st.success("Dashboard ready. Deploy to Streamlit Cloud and share the link!")
