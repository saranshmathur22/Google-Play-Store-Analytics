"""
app.py — Google Play Store Analytics: Internship Tasks Dashboard (Streamlit)

ONE file, everything included: data cleaning, all 6 required visualizations,
and IST time-gating. Run with:  streamlit run app.py

Data caveats (documented inline where relevant, and again at the bottom of
the app under "Notes & Limitations"):
  - The dataset has no Country column -> Task 3's "Country" level is a
    single placeholder node, not real geography.
  - The dataset has no install-history field -> "monthly installs" in
    Tasks 4 & 5 is a proxy built from each app's Last-Updated month.
"""
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import zscore

st.set_page_config(page_title="Play Store Analytics — Internship Tasks", layout="wide")

# ----------------------------------------------------------------------
# 1. DATA LOADING & CLEANING  (shared by every task)
# ----------------------------------------------------------------------

def convert_size(size):
    if pd.isna(size):
        return np.nan
    size = str(size).strip()
    if size.endswith("M"):
        try: return float(size[:-1])
        except ValueError: return np.nan
    if size.lower().endswith("k"):
        try: return float(size[:-1]) / 1024.0
        except ValueError: return np.nan
    return np.nan


def convert_android_ver(v):
    if pd.isna(v): return np.nan
    m = re.match(r"(\d+(\.\d+)?)", str(v))
    return float(m.group(1)) if m else np.nan


def clean_installs(x):
    if pd.isna(x): return np.nan
    x = str(x).replace(",", "").replace("+", "").strip()
    try: return int(x)
    except ValueError: return np.nan


def clean_price(x):
    if pd.isna(x): return 0.0
    x = str(x).replace("$", "").strip()
    try: return float(x)
    except ValueError: return 0.0


@st.cache_data
def load_data(apps_path="Play_Store_Data.csv", reviews_path="User_Reviews.csv"):
    df = pd.read_csv(apps_path).drop_duplicates()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df[(df["Rating"].isna()) | (df["Rating"] <= 5)]
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
    df["Installs"] = df["Installs"].apply(clean_installs)
    df["Size"] = df["Size"].apply(convert_size)
    df["Price"] = df["Price"].apply(clean_price)
    df["Android_Ver_Clean"] = df["Android Ver"].apply(convert_android_ver)
    df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")
    df = df.dropna(subset=["App", "Category"])
    df["Name_Len"] = df["App"].str.len()
    df["Has_Digit"] = df["App"].str.contains(r"\d", regex=True, na=False)

    rev = pd.read_csv(reviews_path).dropna(subset=["Translated_Review"])
    sent = rev.groupby("App", as_index=False).agg(
        Sentiment_Subjectivity=("Sentiment_Subjectivity", "mean"))
    df = df.merge(sent, on="App", how="left")
    return df.reset_index(drop=True)


def percentile_normalize(s):
    return s.rank(pct=True) * 100


def weighted_rating(g):
    return (g["Rating"] * g["Reviews"]).sum() / g["Reviews"].sum()


PLOTLY_DARK = dict(paper_bgcolor="#111", plot_bgcolor="#111", font_color="#eee")

# ----------------------------------------------------------------------
# 2. IST TIME GATE
# ----------------------------------------------------------------------

def ist_now():
    return datetime.now(ZoneInfo("Asia/Kolkata"))


def in_window(start_h, end_h, now=None):
    """start_h inclusive, end_h exclusive. `now` is an optional datetime
    override (for automated boundary testing) — defaults to real IST time,
    so runtime behaviour is unchanged when called normally."""
    now = now or ist_now()
    h = now.hour + now.minute / 60
    return start_h <= h < end_h


def task_section(title, start_h, end_h, render_fn):
    """Show a task's title + visualization ONLY inside its IST window.
    Outside the window, render nothing at all for this task (no title,
    no placeholder, no divider) — the task must be fully invisible."""
    if in_window(start_h, end_h):
        st.subheader(f"{title}  ·  visible {start_h:02d}:00–{end_h:02d}:00 IST")
        render_fn()
        st.divider()

# ----------------------------------------------------------------------
# 3. TASK 1 — Radar: Free vs Paid, top 5 categories (1–2 PM IST)
# ----------------------------------------------------------------------

def task1(df):
    metrics = ["Avg_Installs", "Weighted_Rating", "Total_Reviews", "Avg_Size", "Revenue", "Engagement_Rate"]
    labels = {"Avg_Installs": "Avg Installs", "Weighted_Rating": "Weighted Rating",
              "Total_Reviews": "Total Reviews", "Avg_Size": "Avg Size (MB)",
              "Revenue": "Revenue ($)", "Engagement_Rate": "Engagement Rate (%)"}

    d = df[(df.Installs >= 10_000) & (df.Android_Ver_Clean > 4.0) & (df.Size > 15) &
           (df["Content Rating"] == "Everyone") & (df.Name_Len <= 30)].dropna(subset=["Rating", "Reviews", "Size", "Installs"])
    d["Revenue"] = d.Installs * d.Price
    d = d[(d.Type == "Free") | ((d.Type == "Paid") & (d.Revenue > 10_000))]
    if d.empty:
        st.warning("No rows survive Task 6's filters on this dataset.")
        return

    top5 = d.groupby("Category").Installs.sum().nlargest(5).index.tolist()
    d = d[d.Category.isin(top5)]
    d["Engagement_Rate"] = d.Reviews / d.Installs * 100

    def calc(sub):
        if sub.empty: return {m: np.nan for m in metrics}
        return {"Avg_Installs": sub.Installs.mean(), "Weighted_Rating": weighted_rating(sub),
                "Total_Reviews": sub.Reviews.sum(), "Avg_Size": sub.Size.mean(),
                "Revenue": sub.Revenue.sum(), "Engagement_Rate": sub.Engagement_Rate.mean()}

    scope = st.selectbox("Scope", ["Overall (Top 5 Categories)"] + top5, key="t1_scope")
    sub_scope = d if scope.startswith("Overall") else d[d.Category == scope]

    rows = []
    for typ in ["Free", "Paid"]:
        m = calc(sub_scope[sub_scope.Type == typ]); m["Type"] = typ; rows.append(m)
    table = pd.DataFrame(rows)
    norm = table.copy()
    for m in metrics:
        norm[m] = percentile_normalize(pd.concat([df.assign(**{m: np.nan})[m], table[m]]).dropna()) if False else table[m]
    # normalize against the full top5 dataset for stable percentiles
    full_rows = []
    for cat in top5:
        for typ in ["Free", "Paid"]:
            r = calc(d[(d.Category == cat) & (d.Type == typ)]); r["Type"], r["Category"] = typ, cat
            full_rows.append(r)
    full_table = pd.DataFrame(full_rows)
    for m in metrics:
        full_table[m] = percentile_normalize(full_table[m])
    norm = full_table[(full_table.Category == (scope if scope in top5 else full_table.Category.iloc[0]))] if scope in top5 else full_table.groupby("Type")[metrics].mean().reset_index()

    fig = go.Figure()
    for typ, color in [("Free", "#00CC96"), ("Paid", "#EF553B")]:
        row = norm[norm.Type == typ]
        if row.empty: continue
        r = row[metrics].values.flatten().tolist(); r.append(r[0])
        theta = [labels[m] for m in metrics] + [labels[metrics[0]]]
        fig.add_trace(go.Scatterpolar(r=r, theta=theta, name=typ, fill="toself", opacity=0.55,
                                       line=dict(color=color)))
    comp = norm.set_index("Type")[metrics].mean(axis=1) if "Type" in norm else None
    if comp is not None and len(comp) == 2:
        comp_str = " · ".join(f"{t}: {v:.1f}" for t, v in comp.items())
        st.caption(f"**Composite Score Comparison** — {scope}: {comp_str}")
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], color="#eee", gridcolor="#444"),
                                  angularaxis=dict(color="#eee"), bgcolor="#111"),
                       **PLOTLY_DARK, height=450)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# 4. TASK 2 — Hex-style density: Size vs Rating (5–7 PM IST)
# ----------------------------------------------------------------------

def task2(df):
    cats = ["GAME", "BEAUTY", "BUSINESS", "COMICS", "COMMUNICATION", "DATING", "ENTERTAINMENT", "SOCIAL", "EVENTS"]
    translate = {"BEAUTY": "ब्यूटी", "BUSINESS": "வணிகம்", "DATING": "Partnersuche (DE)"}

    d = df[df.Category.isin(cats) & (df.Rating > 3.5) & (df.Installs > 50_000) & (df.Reviews > 500) &
           df.Size.between(10, 100) & (df.Sentiment_Subjectivity > 0.5) &
           ~df.App.str.contains("s", case=False, na=False)].dropna(subset=["Size", "Rating", "Installs"])
    if d.empty:
        st.warning("No rows survive Task 1's filters (the 'exclude names containing S' rule is very strict).")
        return

    outliers = []
    for cat, g in d.groupby("Category"):
        q1, q3 = g.Installs.quantile([.25, .75]); iqr = q3 - q1
        outliers.append(g[(g.Installs < q1 - 1.5 * iqr) | (g.Installs > q3 + 1.5 * iqr)])
    outliers = pd.concat(outliers) if outliers else d.iloc[0:0]

    # True hexagonal binning (matplotlib hexbin) — Plotly's open-source build
    # has no native hexbin trace, so go.Histogram2d (square bins) was
    # previously used as a stand-in and mislabelled. This is a real hexbin.
    # Trade-off: matplotlib is static, so pan/zoom/hover interactivity from
    # the rest of the (Plotly) dashboard is not available on this one chart.
    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=.03, hspace=.03)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    hb = ax_main.hexbin(d.Size, d.Rating, C=d.Installs, reduce_C_function=np.mean,
                         gridsize=22, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax_main, fraction=0.046, pad=0.04, label="Avg Installs")

    game = d[d.Category == "GAME"]
    ax_main.scatter(game.Size, game.Rating, color="deeppink", s=18, alpha=.7, label="Game apps", zorder=3)

    ax_main.scatter(outliers.Size, outliers.Rating, marker="x", color="red", s=60, label="IQR outliers", zorder=4)
    for _, r in outliers.iterrows():
        ax_main.annotate(str(r.App)[:14], (r.Size, r.Rating), fontsize=6, color="red",
                          xytext=(3, 3), textcoords="offset points")

    ax_main.set_xlabel("Size (MB)"); ax_main.set_ylabel("Rating")
    ax_main.legend(loc="lower right", fontsize=7)

    ax_top.hist(d.Size, bins=20, color="#636EFA")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_right.hist(d.Rating, bins=20, orientation="horizontal", color="#EF553B")
    ax_right.set_xticks([]); ax_right.set_yticks([])
    ax_top.set_title("Task 1 — Size vs Rating true hexbin (colour = avg installs)")

    st.pyplot(fig, clear_figure=True)
    present = [c for c in d.Category.unique() if c in translate]
    if present:
        st.caption(" | ".join(f"{c.title()} = {translate[c]}" for c in present))

# ----------------------------------------------------------------------
# 5. TASK 3 — Sunburst: Country→Category→Type→Rating Band (6–8 PM IST)
# ----------------------------------------------------------------------

def rating_band(r):
    if 4.0 <= r < 4.2: return "4.0-4.2"
    if 4.2 <= r < 4.5: return "4.2-4.5"
    if 4.5 <= r < 4.7: return "4.5-4.7"
    if 4.7 <= r <= 5.0: return "4.7-5.0"
    return None


def task3(df):
    translate = {"BUSINESS": "வணிகம்", "TRAVEL_AND_LOCAL": "Voyages/lieux locaux", "PRODUCTIVITY": "Productividad"}
    label = lambda c: translate.get(c, c.replace("_", " ").title())

    d = df[(df.Rating >= 4.0) & (df.Installs > 10_000) & (df.Reviews > 1_000) &
           df.Size.between(15, 80) & ~df.Has_Digit &
           ~df.Category.str.startswith(("A", "C", "G", "S"))].dropna(subset=["Rating", "Installs", "Reviews"])
    if d.empty:
        st.warning("No rows survive Task 2's filters.")
        return

    top5 = d.groupby("Category").Installs.sum().nlargest(5).index.tolist()
    d = d[d.Category.isin(top5)].copy()
    d["Rating_Band"] = d.Rating.apply(rating_band)
    d = d.dropna(subset=["Rating_Band"])
    st.caption("⚠️ Dataset has no Country field — using a single placeholder node "
               "'Global (no country field in dataset)' as the top level.")

    ids, labels, parents, values, colors, hover = [], [], [], [], [], []

    def add_node(node_id, node_label, parent_id, parent_installs, group):
        installs = group.Installs.sum()
        rating = weighted_rating(group)
        reviews = group.Reviews.sum()
        pct = (installs / parent_installs * 100) if parent_installs else 100.0
        ids.append(node_id); labels.append(node_label); parents.append(parent_id)
        values.append(installs); colors.append(rating)
        hover.append(f"Installs: {installs:,}<br>Weighted rating: {rating:.2f}"
                      f"<br>Review count: {reviews:,}<br>% of parent: {pct:.1f}%")

    total_installs = d.Installs.sum()
    root = "World"
    add_node(root, root, "", total_installs, d)  # root: % of parent = 100

    country = "Global (no country field in dataset)"
    c_id = f"{root}/{country}"
    add_node(c_id, country, root, total_installs, d)

    for cat, gcat in d.groupby("Category"):
        cat_id = f"{c_id}/{cat}"
        add_node(cat_id, label(cat), c_id, d.Installs.sum(), gcat)
        for typ, gtyp in gcat.groupby("Type"):
            t_id = f"{cat_id}/{typ}"
            add_node(t_id, typ, cat_id, gcat.Installs.sum(), gtyp)
            for band, gband in gtyp.groupby("Rating_Band"):
                b_id = f"{t_id}/{band}"
                add_node(b_id, band, t_id, gtyp.Installs.sum(), gband)

    fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
                                 marker=dict(colors=colors, colorscale="RdYlGn", cmin=4.0, cmax=5.0),
                                 hovertext=hover, hoverinfo="label+text"))
    # Sunburst click-to-drill-down / click-to-zoom-out is native Plotly
    # behaviour — no extra code is needed to enable it.
    fig.update_layout(**PLOTLY_DARK, height=520, title="Category → Type → Rating Band (size = installs, colour = weighted rating)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# 6. TASK 4 — Calendar heatmap: monthly installs (6–9 PM IST)
# ----------------------------------------------------------------------

def task4(df):
    translate = {"BEAUTY": "ब्यूटी", "BUSINESS": "வணிகம்", "DATING": "Partnersuche (DE)"}
    label = lambda c: translate.get(c, c.title())

    d = df[df.Category.str.startswith(("E", "C", "B")) & (df.Rating >= 4.0) & (df.Installs > 10_000) &
           (df.Reviews > 500) & df.Size.between(15, 80) & (df.Sentiment_Subjectivity > 0.5) &
           ~df.App.str.startswith(("X", "Y", "Z")) & ~df.App.str.contains("s", case=False, na=False)
           ].dropna(subset=["Last Updated", "Installs"])
    if d.empty:
        st.warning("No rows survive Task 3's filters.")
        return
    st.caption("⚠️ No install-history field exists in the dataset. The series below is a "
               "**Monthly Install Proxy**: current Installs bucketed by each app's "
               "Last-Updated month — not real historical install telemetry.")

    top5 = d.groupby("Category").Installs.sum().nlargest(5).index.tolist()
    cat = st.selectbox("Category", sorted(top5), key="t4_cat")
    g = d[d.Category == cat].copy()
    g["Month"] = g["Last Updated"].values.astype("datetime64[M]")
    monthly = g.groupby("Month").Installs.sum().sort_index()
    reviews_monthly = g.groupby("Month").Reviews.sum().sort_index()
    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_idx).fillna(0)
    reviews_monthly = reviews_monthly.reindex(full_idx).fillna(0)
    growth = monthly.pct_change() * 100
    rolling = monthly.rolling(3, min_periods=1).mean()

    # Genuine 3-month forward projection ("3-Month Proxy Trend Projection"):
    # linear slope from the first to the last of the last 3 proxy months,
    # extended 3 months forward. Rendered as a visually distinct trace.
    last3 = monthly.tail(3)
    slope = (last3.iloc[-1] - last3.iloc[0]) / max(len(last3) - 1, 1)
    forecast_idx = pd.date_range(monthly.index.max() + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    forecast_vals = [max(monthly.iloc[-1] + slope * (k + 1), 0) for k in range(3)]

    actual_x = [m.strftime("%Y-%m") for m in monthly.index]
    forecast_x = [m.strftime("%Y-%m") for m in forecast_idx]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=[monthly.values], x=actual_x, y=[f"{label(cat)} (Proxy)"], colorscale="YlOrRd",
        customdata=np.stack([growth.fillna(0), rolling, reviews_monthly,
                              ["Proxy"] * len(monthly)], axis=1)[None, :, :],
        hovertemplate="Month %{x}<br>Installs (proxy) %{z:,.0f}<br>Growth %{customdata[0]:.1f}%"
                      "<br>3mo avg %{customdata[1]:,.0f}<br>Review count %{customdata[2]:,.0f}"
                      "<br>Status: %{customdata[3]}<extra></extra>"))
    fig.add_trace(go.Heatmap(
        z=[forecast_vals], x=forecast_x, y=[f"{label(cat)} (3-Month Proxy Trend Projection)"], colorscale="Greys",
        showscale=False,
        customdata=np.array([["Forecast"]] * len(forecast_vals))[None, :, :],
        hovertemplate="Month %{x}<br>Projected installs %{z:,.0f}<br>Status: %{customdata[0]}<extra></extra>"))
    fig.update_layout(**PLOTLY_DARK, height=280, title=f"Monthly Install Proxy — {label(cat)}")
    # Visually highlight >20% MoM growth months directly on the heatmap
    # (not just listed below) — annotate each qualifying cell.
    for i, (m, gr) in enumerate(growth.items()):
        if pd.notna(gr) and gr > 20:
            fig.add_annotation(x=actual_x[i], y=f"{label(cat)} (Proxy)", text="▲",
                                showarrow=False, font=dict(color="#00FF88", size=14),
                                yshift=14)
    st.plotly_chart(fig, use_container_width=True)
    hi_growth = growth[growth > 20]
    if not hi_growth.empty:
        st.caption("▲ = months with >20% MoM growth (proxy): " + ", ".join(m.strftime("%Y-%m") for m in hi_growth.index))

# ----------------------------------------------------------------------
# 7. TASK 5 — Streamgraph: monthly/cumulative installs (4–6 PM IST)
# ----------------------------------------------------------------------

def task5(df):
    translate = {"TRAVEL_AND_LOCAL": "Voyages/lieux locaux", "PRODUCTIVITY": "Productividad", "PHOTOGRAPHY": "写真"}
    label = lambda c: translate.get(c, c.replace("_", " ").title())

    d = df[df.Category.str.startswith(("T", "P", "B")) & (df.Rating >= 4.2) & (df.Reviews > 1_000) &
           df.Size.between(20, 80) & (df.Installs >= 10_000) & ~df.Has_Digit
           ].dropna(subset=["Last Updated", "Installs"])
    if d.empty:
        st.warning("No rows survive Task 4's filters.")
        return

    d = d.copy(); d["Month"] = d["Last Updated"].values.astype("datetime64[M]")
    monthly = d.groupby(["Category", "Month"]).Installs.sum().reset_index()
    all_months = pd.date_range(monthly.Month.min(), monthly.Month.max(), freq="MS")
    cats = sorted(monthly.Category.unique())
    wide = monthly.pivot(index="Month", columns="Category", values="Installs").reindex(all_months).fillna(0)
    cumulative = wide.cumsum()
    growth = wide.pct_change().fillna(0) * 100

    view = st.radio("View", ["Monthly", "Cumulative", "Growth %"], horizontal=True, key="t5_view")
    data = {"Monthly": wide, "Cumulative": cumulative, "Growth %": growth}[view]

    palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]
    fig = go.Figure()
    for i, cat in enumerate(cats):
        s = wide[cat]
        roll_mean, roll_std = s.rolling(4, min_periods=2).mean(), s.rolling(4, min_periods=2).std().replace(0, np.nan)
        z = (s - roll_mean) / roll_std
        anomaly = (growth[cat] > 25) | (z.abs() > 2)
        fig.add_trace(go.Scatter(x=all_months, y=data[cat], mode="lines", stackgroup="one",
                                  name=label(cat), line=dict(width=.5, color=palette[i % len(palette)])))
        if anomaly.any():
            fig.add_trace(go.Scatter(x=all_months[anomaly], y=data[cat][anomaly], mode="markers",
                                      marker=dict(color="yellow", size=10, symbol="star"),
                                      name=f"{label(cat)} anomaly"))
    fig.update_layout(**PLOTLY_DARK, height=420, title=f"{view} Installs — Categories T/P/B")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# 8. TASK 6 — Clustered heatmap: category performance (3–5 PM IST)
# ----------------------------------------------------------------------

def task6(df):
    metrics = ["Weighted_Rating", "Total_Reviews", "Total_Installs", "Avg_Size", "Engagement_Rate", "Update_Frequency"]
    weights = {"Weighted_Rating": .25, "Total_Reviews": .15, "Total_Installs": .25,
               "Avg_Size": .05, "Engagement_Rate": .15, "Update_Frequency": .15}

    d = df[(df.Rating >= 4.0) & (df.Size > 10) & (df.Installs >= 10_000) & (df.Reviews > 1_000) &
           (df["Last Updated"].dt.month == 1) & ~df.Has_Digit
           ].dropna(subset=["Rating", "Installs", "Reviews", "Size", "Last Updated"])
    if d.empty:
        st.warning("No rows survive Task 5's filters (needs apps last updated in January).")
        return

    top10 = d.groupby("Category").Installs.sum().nlargest(10).index.tolist()
    d = d[d.Category.isin(top10)]
    rows = []
    for cat, g in d.groupby("Category"):
        rows.append({"Category": cat, "Weighted_Rating": weighted_rating(g),
                     "Total_Reviews": g.Reviews.sum(), "Total_Installs": g.Installs.sum(),
                     "Avg_Size": g.Size.mean(), "Engagement_Rate": (g.Reviews / g.Installs).mean() * 100,
                     "Update_Frequency": g["Last Updated"].dt.to_period("M").nunique()})
    table = pd.DataFrame(rows).set_index("Category")
    z = table[metrics].apply(zscore).fillna(0)
    table["Composite"] = sum(z[m] * w for m, w in weights.items())

    order = z.index[leaves_list(linkage(z.values, method="average"))].tolist()
    top3, bot3 = table.Composite.nlargest(3).index.tolist(), table.Composite.nsmallest(3).index.tolist()

    view = st.radio("Values", ["Normalized (Z-score)", "Raw"], horizontal=True, key="t6_view")
    zdata = z.loc[order].values if view.startswith("Normalized") else table.loc[order, metrics].values
    scale = "RdBu" if view.startswith("Normalized") else "Viridis"

    fig = go.Figure(go.Heatmap(z=zdata, x=metrics, y=order, colorscale=scale,
                                zmid=0 if view.startswith("Normalized") else None))
    for c in order:
        if c in top3 or c in bot3:
            fig.add_annotation(x=metrics[-1], y=c, text="TOP" if c in top3 else "BOTTOM", showarrow=False,
                                xshift=60, font=dict(color="gold" if c in top3 else "red", size=10))
    fig.update_layout(**PLOTLY_DARK, height=440, title="Category performance, clustered by similarity")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# 9. PAGE LAYOUT
# ----------------------------------------------------------------------

st.title("📊 Google Play Store Analytics — Internship Tasks")
st.caption(f"Current IST time: {ist_now():%Y-%m-%d %H:%M:%S} — each section below auto shows/hides "
           "based on its required IST display window (re-checked every time the page reruns).")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Play_Store_Data.csv / User_Reviews.csv not found next to app.py — place both CSVs in the same folder.")
    st.stop()

# NOTE: internal function names (task1..task6) were left unchanged to avoid
# an unnecessary rewrite; only the DISPLAYED task numbers/titles below were
# corrected to match the required mapping. Each function already carried the
# correct IST window for its correct task number, so only the labels moved:
#   Task 1 = Hexbin (fn task2)     Task 4 = Streamgraph (fn task5)
#   Task 2 = Sunburst (fn task3)   Task 5 = Clustered Heatmap (fn task6, UNCHANGED)
#   Task 3 = Calendar (fn task4)   Task 6 = Radar (fn task1)
task_section("Task 1 — Size vs Rating Hexbin", 17, 19, lambda: task2(df))
task_section("Task 2 — Country→Category→Type→Rating Band Sunburst", 18, 20, lambda: task3(df))
task_section("Task 3 — Monthly Install Proxy Calendar Heatmap", 18, 21, lambda: task4(df))
task_section("Task 4 — Installs Streamgraph (T/P/B categories)", 16, 18, lambda: task5(df))
task_section("Task 5 — Category Performance Clustered Heatmap", 15, 17, lambda: task6(df))
task_section("Task 6 — Free vs Paid Radar / Composite Score (Top 5 Categories)", 13, 14, lambda: task1(df))

with st.expander("Notes & data limitations (read before submitting)"):
    st.markdown("""
- **No Country field** exists anywhere in the source data, so Task 3 uses a single
  placeholder root node instead of real countries.
- **No install-history field** exists — Tasks 4 & 5 build "monthly installs" from
  each app's current Installs value, bucketed by its Last-Updated month. This is a
  documented proxy, not real historical install telemetry.
- Task 2 excludes any app name containing the letter "S" (per the brief), which
  removes most apps and can leave a small sample.
- IST time-gating is re-checked every time the app reruns (any click, or a manual
  browser refresh) — it isn't a live background timer, since Streamlit scripts only
  run on interaction/refresh.
""")
